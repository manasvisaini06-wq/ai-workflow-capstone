"""Exploratory data analysis: every figure that backs a statement in REPORT.md.

    python eda.py [data_dir]

Titles are generated from the computed numbers, so a figure can never disagree with its data.
Outputs: reports/figures/eda_*.png and reports/eda_summary.json
"""
import json
import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import ingestion
import viz_style as vs
from features import engineer_features


def run(data_dir=None):
    data_dir = data_dir or os.path.join(config.BASE_DIR, "data", "cs-train")
    raw = ingestion.load_json_files(data_dir)
    df = ingestion.clean(raw)
    ts = ingestion.build_ts_dict(df)
    allts = ts["all"].set_index("date")
    summary = {"rows_loaded": int(len(raw)), "rows_kept": int(len(df)),
               "adjustments_removed": int(len(raw) - len(df)),
               "first_day": str(allts.index.min().date()), "last_day": str(allts.index.max().date()),
               "n_countries": int(df["country"].nunique())}

    # 1 -- revenue over time (the 30-day total is exactly what the model forecasts)
    r30 = allts["revenue"].rolling(30).sum().dropna()
    q4 = r30[(r30.index >= "2018-11-01") & (r30.index <= "2018-12-31")].mean()
    other = r30[(r30.index < "2018-10-01") | (r30.index > "2019-01-31")].median()
    fig, ax = vs.new_fig(10, 4.8)
    ax.plot(r30.index, r30.values, color=vs.BLUE)
    ax.yaxis.set_major_formatter(vs.money)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Revenue over the preceding 30 days")
    peak_day = r30.idxmax()
    ax.annotate(f"Peak {vs.money(r30.max())}\n{peak_day:%d %b %Y}", xy=(peak_day, r30.max()), xytext=(15, -5),
                textcoords="offset points", color=vs.INK2, fontsize=9, va="top")
    vs.headline(fig, f"Revenue spiked in Q4 2018: Nov-Dec ran at about {q4/other:.1f}x the typical 30-day level",
                "30-day trailing revenue, all countries - the quantity the model forecasts. Only one full Q4 is in the data, "
                "so the seasonal cycle is observed once")
    summary["q4_vs_typical_ratio"] = round(float(q4 / other), 2)
    summary["revenue_30d"] = {"min": float(r30.min()), "median": float(r30.median()), "max": float(r30.max())}
    vs.save(fig, "eda_01_revenue_over_time.png")

    # 2 -- country concentration
    by_country = df.groupby("country")["price"].sum().sort_values(ascending=False)
    share = by_country / by_country.sum() * 100
    top10 = share.head(10)
    fig, ax = vs.new_fig(10, 5)
    ax.barh(top10.index[::-1], top10.values[::-1], color=vs.BLUE, height=0.6)
    for y, v in enumerate(top10.values[::-1]):
        ax.text(v + 0.8, y, f"{v:.1f}%", va="center", color=vs.INK2, fontsize=9)
    ax.set_xlabel("Share of total revenue (%)")
    ax.grid(axis="y", visible=False)
    vs.headline(fig, f"The UK generates {top10.iloc[0]:.0f}% of revenue; the other nine top-10 countries share "
                     f"only {top10.iloc[1:].sum():.0f}%", f"Top 10 of {summary['n_countries']} countries by revenue")
    summary["uk_share_pct"] = round(float(top10.iloc[0]), 1)
    summary["top10_share_pct"] = round(float(top10.sum()), 1)
    vs.save(fig, "eda_02_country_revenue.png")

    # 3 -- weekday pattern
    dow = allts["revenue"].groupby(allts.index.dayofweek).median()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_median = dow.loc[[0, 1, 2, 3, 4]].mean()
    fig, ax = vs.new_fig(8, 4.6)
    colors = [vs.BLUE if i not in (5, 6) else vs.GREY for i in range(7)]
    ax.bar(names, dow.values, color=colors, width=0.6)
    for i, v in enumerate(dow.values):
        ax.text(i, v + 120, vs.money(v) if v else "0", ha="center", color=vs.INK2, fontsize=9)
    ax.yaxis.set_major_formatter(vs.money)
    ax.set_ylabel("Median daily revenue")
    ax.grid(axis="x", visible=False)
    vs.headline(fig, f"There is no trading on Saturdays, and Sunday revenue is only {dow[6]/weekday_median*100:.0f}% "
                     f"of a typical weekday", "Median daily revenue by weekday, all countries")
    summary["weekday_median"] = {n: float(v) for n, v in zip(names, dow.values)}
    vs.save(fig, "eda_03_weekday_pattern.png")

    # 4 -- skewed daily revenue
    traded = allts["revenue"][allts["revenue"] > 0]
    p99 = traded.quantile(0.99)
    fig, ax = vs.new_fig(9, 4.6)
    ax.hist(traded.clip(upper=p99), bins=40, color=vs.BLUE, edgecolor=vs.SURFACE, linewidth=0.8)
    ax.axvline(traded.median(), color=vs.INK2, linewidth=1.2, linestyle="--")
    ax.text(traded.median() + 300, ax.get_ylim()[1] * 0.92, f"median {vs.money(traded.median())}", color=vs.INK2)
    ax.xaxis.set_major_formatter(vs.money)
    ax.set_xlabel(f"Daily revenue (values above the 99th percentile, {vs.money(p99)}, are grouped in the last bar)")
    ax.set_ylabel("Number of trading days")
    ax.grid(axis="x", visible=False)
    vs.headline(fig, f"Daily revenue is skewed: the largest day ({vs.money(traded.max())}) is "
                     f"{traded.max()/traded.median():.0f}x the median", f"Distribution of daily revenue, "
                     f"{len(traded)} trading days")
    summary["largest_day"] = {"date": str(traded.idxmax().date()), "revenue": float(traded.max())}
    vs.save(fig, "eda_04_daily_distribution.png")

    # 5 -- sparsity of small countries
    zero_share = pd.Series({k: (v["revenue"] == 0).mean() * 100 for k, v in ts.items()}).sort_values()
    fig, ax = vs.new_fig(9, 5)
    colors = [vs.BLUE if v < 50 else vs.ORANGE for v in zero_share.values]
    ax.barh(zero_share.index, zero_share.values, color=colors, height=0.6)
    for y, v in enumerate(zero_share.values):
        ax.text(v + 1, y, f"{v:.0f}%", va="center", color=vs.INK2, fontsize=9)
    ax.set_xlabel("Days with zero revenue (%)")
    ax.set_xlim(0, 112)
    ax.grid(axis="y", visible=False)
    n_sparse = int((zero_share >= 80).sum())
    vs.headline(fig, f"{n_sparse} of the top-10 countries had zero revenue on over 80% of days - "
                     f"their 30-day revenue is lumpy and hard to forecast", "Share of days with no revenue, by series "
                     "(orange = at least half of days empty)")
    summary["zero_day_pct"] = {k: round(float(v), 1) for k, v in zero_share.items()}
    vs.save(fig, "eda_05_sparse_series.png")

    # 6 -- which features relate to the target
    X, y, _ = engineer_features(ts["all"], training=True)
    corr = X.corrwith(pd.Series(y)).sort_values()
    fig, ax = vs.new_fig(9, 4.8)
    ax.barh(corr.index, corr.values, color=[vs.BLUE if v >= 0 else vs.GREY for v in corr.values], height=0.6)
    for yy, v in enumerate(corr.values):
        ax.text(v + (0.012 if v >= 0 else -0.012), yy, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", color=vs.INK2, fontsize=9)
    ax.axvline(0, color=vs.INK2, linewidth=0.8)
    ax.set_xlabel("Correlation with the next-30-day revenue (all countries)")
    ax.set_xlim(-0.5, 0.75)
    ax.grid(axis="y", visible=False)
    best = corr.idxmax()
    vs.headline(fig, f"Recent invoice activity ('{best}', r={corr.max():.2f}) predicts next-month revenue better than "
                     f"trailing revenue (r={corr['previous_28']:.2f})", "Correlation of each engineered feature "
                     "with the target, training rows only")
    summary["feature_correlation"] = {k: round(float(v), 3) for k, v in corr.items()}
    vs.save(fig, "eda_06_feature_correlation.png")

    # 7 -- cancellations
    monthly = df.assign(month=df["invoice_date"].dt.to_period("M").astype(str)).groupby(["month", "invoice_type"])[
        "price"].sum().unstack().fillna(0)
    cancel_pct = monthly["cancellation"] / monthly.sum(axis=1) * 100
    fig, ax = vs.new_fig(10, 4.6)
    ax.bar(cancel_pct.index, cancel_pct.values, color=vs.BLUE, width=0.7)
    ax.set_ylabel("Cancellation invoices as % of recorded revenue")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    ax.grid(axis="x", visible=False)
    vs.headline(fig, f"Cancellations make up {cancel_pct.min():.0f}-{cancel_pct.max():.0f}% of monthly recorded revenue, so "
                     f"how they are treated matters", "Cancellation ('C') invoices carry positive prices in the "
                     "source data; they are kept as recorded and flagged for the business to confirm")
    summary["cancellation_pct_overall"] = round(float(monthly["cancellation"].sum() / monthly.sum().sum() * 100), 1)
    vs.save(fig, "eda_07_cancellations.png")

    out = os.path.join(config.BASE_DIR, "reports", "eda_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1] if len(sys.argv) > 1 else None), indent=2)[:1500])
