"""Post-production analysis: how well did the deployed forecasts do, and what does that mean for the business?

    python post_production_analysis.py

Reads the prediction logs written by the API during simulate_production.py (logs/ = weekly retraining,
logs/frozen/ = models never refreshed), scores them against the revenue that actually happened (the gold
standard) and against the naive baseline, and writes
    reports/post_production_metrics.csv, reports/post_production_summary.json,
    reports/figures/prod_01..03_*.png

Business metric: the forecast is used to plan the next 30 days of revenue, so the question is how often
the forecast is close enough to plan with. "Planning-grade" = within +/-20% of actual 30-day revenue.
"""
import csv
import json
import os

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import config
import ingestion
import model
import monitoring
import viz_style as vs
from features import engineer_features

TOLERANCE = 0.20
COMBINED = os.path.join(config.BASE_DIR, "data", "combined")


def read_log(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def baseline_for(ts_dict, scored):
    cache, out = {}, []
    for c, d in zip(scored["country"], scored["target_date"]):
        if c not in cache:
            X, _, dates = engineer_features(ts_dict[c], training=False)
            cache[c] = (X, {str(x): i for i, x in enumerate(dates)})
        X, idx = cache[c]
        out.append(float(model.baseline_predict(X.iloc[[idx[d]]])[0]))
    return np.array(out)


def metrics(df, col):
    err = df[col] - df["y_true"]
    return {"rmse": float(np.sqrt((err ** 2).mean())), "mae": float(err.abs().mean()),
            "bias": float(err.mean()), "wape": float(err.abs().sum() / df["y_true"].sum()),
            "within_tol": float((err.abs() <= TOLERANCE * df["y_true"]).mean())}


def main():
    ts = ingestion.get_ts(COMBINED)
    logs = {"retrained": os.path.join(config.BASE_DIR, "logs", "predict.log.csv"),
            "frozen": os.path.join(config.BASE_DIR, "logs", "frozen", "predict.log.csv")}
    scored = {k: monitoring.score_predictions(read_log(p), ts) for k, p in logs.items()}
    r, f = scored["retrained"].copy(), scored["frozen"]
    r["baseline"] = baseline_for(ts, r)
    r["frozen"] = f.set_index(["country", "target_date"]).loc[list(zip(r["country"], r["target_date"])), "y_pred"].to_numpy()
    r["date"] = pd.to_datetime(r["target_date"])

    rows = []
    for series, sub in r.groupby("country"):
        row = {"series": series, "n_forecasts": len(sub), "mean_actual_30d_revenue": sub["y_true"].mean()}
        for label, col in [("deployed", "y_pred"), ("baseline", "baseline"), ("frozen", "frozen")]:
            for k, v in metrics(sub, col).items():
                row[f"{label}_{k}"] = v
        row["deployed_vs_baseline_rmse"] = row["deployed_rmse"] / row["baseline_rmse"]
        row["deployed_vs_frozen_rmse"] = row["deployed_rmse"] / row["frozen_rmse"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(config.BASE_DIR, "reports", "post_production_metrics.csv"), index=False)

    # relationship between error and the business metric: error as a share of revenue vs revenue size
    per = table.copy()
    corr = float(np.corrcoef(np.log(per["mean_actual_30d_revenue"]), per["deployed_wape"])[0, 1])
    flagged = r["drift_flag"].astype(str).str.lower() == "true"
    drift_summary = {"flag_rate": float(flagged.mean()),
                     "mape_flagged": float(r.loc[flagged, "abs_pct_error"].mean()) if flagged.any() else None,
                     "mape_unflagged": float(r.loc[~flagged, "abs_pct_error"].mean())}
    summary = {"period": [str(r["date"].min().date()), str(r["date"].max().date())],
               "n_forecasts": int(len(r)), "tolerance": TOLERANCE,
               "corr_log_revenue_vs_wape": corr, "drift": drift_summary}
    json.dump(summary, open(os.path.join(config.BASE_DIR, "reports", "post_production_summary.json"), "w"), indent=2)
    plots(r, table, per)
    print(table.round(3).to_string(index=False))
    print(json.dumps(summary, indent=1))


def plots(r, table, per):
    import matplotlib.pyplot as plt
    # 1 -- all countries: forecast vs actual through production
    a = r[r.country == "all"].sort_values("date")
    fig, ax = vs.new_fig(10.5, 5)
    ax.plot(a["date"], a["y_true"], color=vs.INK, linewidth=2.2, label="Actual")
    ax.plot(a["date"], a["y_pred"], color=vs.BLUE, linewidth=1.8,
            label="Deployed forecast (the naive baseline serves this series)")
    ax.yaxis.set_major_formatter(vs.money)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Revenue in the 30 days starting that day")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    row = table[table.series == "all"].iloc[0]
    vs.headline(fig, f"All countries: the deployed forecast was off by {row.deployed_wape * 100:.0f}% of revenue on "
                     f"average in production", f"Forecasts for {a['date'].min():%d %b} to {a['date'].max():%d %b %Y}, "
                     f"each made using only earlier data; the models are retrained weekly")
    vs.save(fig, "prod_01_all_countries.png")

    # 2 -- deployed vs baseline vs frozen, relative error per series
    t = per.sort_values("deployed_vs_baseline_rmse")
    fig, ax = vs.new_fig(10.5, 5.6)
    y = np.arange(len(t))
    ax.barh(y - 0.2, t["deployed_vs_baseline_rmse"], height=0.38, color=vs.BLUE, label="Deployed (weekly retrain)")
    ax.barh(y + 0.2, t["frozen_rmse"] / t["baseline_rmse"], height=0.38, color="#a9c8ee", label="Never retrained")
    ax.axvline(1.0, color=vs.INK, linewidth=1.2)
    ax.set_yticks(y); ax.set_yticklabels(t["series"]); ax.invert_yaxis()
    ax.set_xlabel("RMSE relative to the naive baseline (below 1.0 = better than baseline)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    learned = t[(t["deployed_vs_baseline_rmse"] - 1).abs() > 1e-9]
    n_better = int((learned["deployed_vs_baseline_rmse"] < 1).sum())
    vs.headline(fig, f"In production a learned model beat the baseline for only {n_better} of the {len(learned)} series "
                     f"where one is deployed",
                "'all' and united_kingdom run the baseline itself (ratio 1.0). Ratio of RMSE to the baseline's RMSE on the same forecasts")
    vs.save(fig, "prod_02_vs_baseline_by_series.png")

    # 3 -- performance vs the business metric: error as a share of revenue, by revenue size
    fig, ax = vs.new_fig(10, 5.4)
    ax.scatter(per["mean_actual_30d_revenue"], per["deployed_wape"] * 100, s=70, color=vs.BLUE, zorder=3)
    for _, q in per.iterrows():
        ax.annotate(q["series"], (q["mean_actual_30d_revenue"], q["deployed_wape"] * 100), xytext=(6, -13) if q["series"] == "all" else (6, 5),
                    textcoords="offset points", fontsize=9, color=vs.INK2)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(vs.money)
    ax.set_xlabel("Average actual revenue per 30 days (log scale)")
    ax.set_ylabel("Forecast error as % of revenue")
    corr = np.corrcoef(np.log(per["mean_actual_30d_revenue"]), per["deployed_wape"])[0, 1]
    vs.headline(fig, "Relative forecast error tends to be lower for the series that carry more revenue"
                if corr < -0.3 else "Larger series are not clearly easier to forecast in relative terms",
                f"One point per series; correlation of log revenue with relative error = {corr:.2f}")
    vs.save(fig, "prod_03_error_vs_revenue.png")


if __name__ == "__main__":
    main()
