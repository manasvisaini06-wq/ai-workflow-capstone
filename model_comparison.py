"""Compare several models with a naive baseline, using leakage-free rolling-origin evaluation.

    python model_comparison.py [data_dir]

For each series, the last 180 labelled days are split into 4 consecutive 45-day test blocks. For each
block the models are fitted ONLY on rows whose 30-day target window had finished before the block
started (a 30-day gap), then scored on the block. Errors from the 4 blocks are pooled.

Only the training data is used here; the production data stay untouched as the final test.
Selection rule per series: the model with the lowest pooled RMSE, but a learned model must beat the
naive baseline by at least 5%, otherwise the baseline is deployed.

Outputs: reports/model_comparison.csv, model_selection.json, reports/figures/model_*.png
"""
import json
import os
import sys
import time

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import config
import ingestion
import model
import viz_style as vs
from features import HORIZON, engineer_features

N_FOLDS, FOLD_DAYS = 4, 45
MIN_IMPROVEMENT = 0.05
CANDIDATES = model.MODEL_NAMES


def rolling_origin(X, y, dates, name):
    """Return (dates, y_true, y_pred) pooled over the test blocks for one model."""
    n = len(X)
    out_d, out_y, out_p = [], [], []
    first_test = n - N_FOLDS * FOLD_DAYS
    for k in range(N_FOLDS):
        t0 = first_test + k * FOLD_DAYS
        test = np.arange(t0, t0 + FOLD_DAYS)
        train = np.arange(0, t0 - (HORIZON - 1))             # targets finished before the block began
        est = model.build_estimator(name).fit(X.iloc[train], y[train])
        out_d.append(dates[test]); out_y.append(y[test]); out_p.append(est.predict(X.iloc[test]))
    return np.concatenate(out_d), np.concatenate(out_y), np.concatenate(out_p)


def run(data_dir=None):
    data_dir = data_dir or os.path.join(config.BASE_DIR, "data", "cs-train")
    ts_all = ingestion.fetch_ts(data_dir)
    rows, preds = [], {}
    start = time.time()
    for series, ts in ts_all.items():
        X, y, dates = engineer_features(ts, training=True)
        for name in CANDIDATES:
            d, yt, yp = rolling_origin(X, y, dates, name)
            preds[(series, name)] = (d, yt, yp)
            rows.append({"series": series, "model": name, "rmse": model.rmse(yt, yp),
                         "mae": float(np.mean(np.abs(yt - yp))), "mean_actual": float(yt.mean())})
        print(f"  {series:15s} done ({time.time() - start:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    base = res[res.model == "baseline_naive"].set_index("series")["rmse"]
    res["baseline_rmse"] = res["series"].map(base)
    res["rel_rmse"] = res["rmse"] / res["baseline_rmse"]
    os.makedirs(os.path.join(config.BASE_DIR, "reports"), exist_ok=True)
    res.to_csv(os.path.join(config.BASE_DIR, "reports", "model_comparison.csv"), index=False)

    # selection
    selection, detail = {}, {}
    for series in ts_all:
        sub = res[(res.series == series) & (res.model != "baseline_naive")].sort_values("rmse")
        best = sub.iloc[0]
        beats = best.rel_rmse <= 1 - MIN_IMPROVEMENT
        selection[series] = best.model if beats else "baseline_naive"
        detail[series] = {"best_learned": best.model, "best_learned_rel_rmse": round(float(best.rel_rmse), 3),
                          "chosen": selection[series], "beats_baseline_by_5pct": bool(beats)}
    json.dump({"rule": f"lowest pooled rolling-origin RMSE; learned model must beat baseline by "
                       f"{MIN_IMPROVEMENT:.0%} else baseline_naive", "selection": selection, "detail": detail},
              open(config.selection_path(), "w"), indent=2)
    plots(res, preds, selection, ts_all)
    return res, selection, detail


LABELS = {"baseline_naive": "Baseline", "ridge": "Ridge", "random_forest": "Random forest",
          "gradient_boosting": "Gradient boosting", "ridge_no_year": "Ridge (no prior-yr)",
          "random_forest_no_year": "Random forest (no prior-yr)",
          "gradient_boosting_no_year": "Gradient boosting (no prior-yr)",
          "ridge_ratio": "Ridge on ratios", "blend_ridge": "50/50 blend: ridge + baseline"}


def plots(res, preds, selection, ts_all):
    import matplotlib.pyplot as plt
    # 1 -- RMSE of every model vs the baseline, for the two series that carry the revenue
    vs.setup()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"wspace": 0.75})
    for ax, series in zip(axes, ["all", "united_kingdom"]):
        sub = res[res.series == series].sort_values("rmse", ascending=False)
        best = sub[sub.model != "baseline_naive"].sort_values("rmse").iloc[0].model
        colors = [vs.GREY if m == "baseline_naive" else (vs.BLUE if m == best else "#a9c8ee") for m in sub.model]
        ax.barh([LABELS[m] for m in sub.model], sub.rmse, color=colors, height=0.62)
        for yy, v in enumerate(sub.rmse):
            ax.text(v * 1.01, yy, vs.money(v), va="center", color=vs.INK2, fontsize=9)
        ax.set_title(f"{series.replace('_', ' ').title()}", loc="left", fontsize=11, color=vs.INK, fontweight="bold")
        ax.set_xlim(0, sub.rmse.max() * 1.18)
        ax.xaxis.set_major_formatter(vs.money)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("RMSE of 30-day revenue forecast (lower is better)")
    b_all = res[(res.series == "all") & (res.model == "baseline_naive")].rmse.iloc[0]
    m_all = res[(res.series == "all") & (res.model != "baseline_naive")].rmse.min()
    word = "worse" if m_all > b_all else "better"
    vs.headline(fig, f"For all countries combined, the best learned model is {abs(m_all / b_all - 1) * 100:.0f}% {word} "
                     f"than the naive baseline", "Rolling-origin evaluation, 4 blocks x 45 days; dark blue = best "
                     "learned model, grey = baseline")
    vs.save(fig, "model_01_rmse_vs_baseline.png")

    # 2 -- chosen model relative to baseline for every series
    chosen = pd.DataFrame([{"series": s, "model": m} for s, m in selection.items()]).merge(
        res[["series", "model", "rel_rmse"]], on=["series", "model"])
    best_learned = res[res.model != "baseline_naive"].sort_values("rel_rmse").groupby("series").first()["rel_rmse"]
    chosen["best_learned_rel"] = chosen["series"].map(best_learned)
    chosen = chosen.sort_values("best_learned_rel", ascending=False)
    fig, ax = vs.new_fig(10, 5.4)
    ax.barh(chosen["series"], chosen["best_learned_rel"], color=[vs.BLUE if v < 1 - MIN_IMPROVEMENT else vs.ORANGE
                                                                  for v in chosen["best_learned_rel"]], height=0.6)
    ax.axvline(1.0, color=vs.INK, linewidth=1.2)
    ax.text(1.0, len(chosen) - 0.4, " baseline = 1.0", color=vs.INK, fontsize=9, va="bottom")
    for yy, v in enumerate(chosen["best_learned_rel"]):
        ax.text(v + 0.02, yy, f"{v:.2f}", va="center", color=vs.INK2, fontsize=9)
    ax.set_xlabel("RMSE of the best learned model relative to the baseline (below 1.0 = better than baseline)")
    ax.grid(axis="y", visible=False)
    n_win = int((chosen["best_learned_rel"] < 1 - MIN_IMPROVEMENT).sum())
    vs.headline(fig, f"A learned model beats the baseline by 5% or more for {n_win} of {len(chosen)} series; the rest "
                     f"are served by the baseline", "Blue = learned model deployed, orange = baseline deployed")
    vs.save(fig, "model_02_relative_error_by_series.png")

    # 3 -- predictions vs actuals over the evaluation blocks (all countries)
    d, yt, _ = preds[("all", "baseline_naive")]
    best_name = selection["all"] if selection["all"] != "baseline_naive" else \
        res[(res.series == "all") & (res.model != "baseline_naive")].sort_values("rmse").iloc[0].model
    fig, ax = vs.new_fig(10.5, 5)
    ax.plot(d, yt, color=vs.INK, label="Actual", linewidth=2.2)
    ax.plot(d, preds[("all", "baseline_naive")][2], color=vs.ORANGE, label="Baseline", linewidth=1.6)
    ax.plot(d, preds[("all", best_name)][2], color=vs.BLUE, label=LABELS[best_name], linewidth=1.8)
    ax.yaxis.set_major_formatter(vs.money)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Revenue in the 30 days starting that day")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    vs.headline(fig, "Forecast vs actual 30-day revenue on the evaluation blocks (all countries)",
                "Each point forecasts the 30 days that start on that date, using only data from before that date")
    vs.save(fig, "model_03_predictions_vs_actual.png")


if __name__ == "__main__":
    res, selection, detail = run(sys.argv[1] if len(sys.argv) > 1 else None)
    print(res[res.series.isin(["all", "united_kingdom"])].round(3).to_string(index=False))
    print(json.dumps(detail, indent=1))
