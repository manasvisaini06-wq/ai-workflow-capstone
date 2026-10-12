"""Performance monitoring: compare logged predictions with the revenue that actually happened.

A prediction for date d is "evaluable" once d+29 has been observed in the data. Only predictions
made OUTSIDE the model's training window are scored, because in-sample predictions would flatter
the model. Drift is monitored two ways: the share of requests whose inputs fall far outside the
training distribution (logged per request) and a rolling error that degrades when behaviour shifts.
"""
import numpy as np
import pandas as pd

import ingestion
import logger
from features import HORIZON


def actual_revenue(ts, date):
    """Revenue of the 30 days starting at ``date``, or None if those days are not all observed yet."""
    days = ts["date"].to_numpy("datetime64[D]")
    start = np.datetime64(date, "D")
    end = start + np.timedelta64(HORIZON - 1, "D")
    if start < days[0] or end > days[-1]:
        return None
    mask = (days >= start) & (days <= end)
    return float(ts["revenue"].to_numpy(float)[mask].sum())


def score_predictions(predictions, ts_dict, only_out_of_sample=True):
    """Join logged predictions with actuals. Returns a DataFrame with y_true, error columns."""
    if not len(predictions):
        return pd.DataFrame(columns=["country", "target_date", "y_pred", "y_true", "error", "abs_pct_error"])
    df = pd.DataFrame(predictions).copy()
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    if only_out_of_sample and "in_training_window" in df:
        df = df[df["in_training_window"].astype(str).str.lower() != "true"]
    df["y_true"] = [actual_revenue(ts_dict[c], d) if c in ts_dict else None
                    for c, d in zip(df["country"], df["target_date"])]
    df = df.dropna(subset=["y_true", "y_pred"]).copy()
    df["error"] = df["y_pred"] - df["y_true"]
    df["abs_pct_error"] = np.where(df["y_true"] > 0, (df["error"].abs() / df["y_true"].where(df["y_true"] > 0)) * 100,
                                   np.nan)
    return df.reset_index(drop=True)


def regression_metrics(scored):
    """RMSE, MAE, MAPE (%), bias (mean signed error) for a scored frame."""
    if not len(scored):
        return {"n_evaluated": 0, "rmse": None, "mae": None, "mape": None, "bias": None}
    err = scored["error"].to_numpy(float)
    return {"n_evaluated": int(len(scored)), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "mape": float(np.nanmean(scored["abs_pct_error"])),
            "bias": float(np.mean(err))}


def monitor_report(test=False, data_dir=None, tail=None):
    """Summary used by the /monitor endpoint."""
    predictions = logger.read_log("predict", test=test, tail=tail)
    report = {"n_predictions": len(predictions), "drift_flag_rate": None, "by_country": {}, "overall": None}
    if not predictions:
        return report
    flags = [str(p.get("drift_flag")).lower() == "true" for p in predictions]
    report["drift_flag_rate"] = float(np.mean(flags))
    try:
        ts_dict = ingestion.get_ts(data_dir)
    except ingestion.IngestionError:
        return report
    scored = score_predictions(predictions, ts_dict)
    report["overall"] = regression_metrics(scored)
    countries = {p["country"] for p in predictions}
    for country in sorted(countries):
        sub = scored[scored["country"] == country] if len(scored) else scored
        cflags = [str(p.get("drift_flag")).lower() == "true" for p in predictions if p["country"] == country]
        report["by_country"][country] = {**regression_metrics(sub), "n_predictions": len(cflags),
                                         "drift_flag_rate": float(np.mean(cflags))}
    return report
