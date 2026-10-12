"""Feature engineering for the 30-day revenue forecast.

For a forecast date ``d`` the target is the revenue of days ``d .. d+29`` (inclusive) and every
feature is computed ONLY from days strictly before ``d``. That rule is what makes the forecast
usable on the day it is requested, and it is covered by a leakage unit test.
"""
import numpy as np
import pandas as pd

HORIZON = 30                      # days of revenue being predicted
WINDOWS = (7, 14, 28, 70)         # trailing revenue windows (days)
MIN_HISTORY = max(WINDOWS)        # first forecastable date needs this much history
FEATURE_NAMES = [f"previous_{w}" for w in WINDOWS] + ["previous_year", "has_previous_year",
                                                      "recent_invoices", "recent_views"]


def engineer_features(ts, training=True):
    """Return ``(X, y, dates)`` for a daily time-series frame (see ``ingestion.convert_to_ts``).

    training=True  : only dates whose full 30-day target is known (safe for fitting).
    training=False : every forecastable date, including the day after the last observation
                     (``y`` is NaN where the target is not yet known).
    """
    n = len(ts)
    if n < MIN_HISTORY + 1:
        raise ValueError(f"need at least {MIN_HISTORY + 1} days of data, got {n}")

    revenue = ts["revenue"].to_numpy(float)
    day_arr = ts["date"].to_numpy("datetime64[D]")
    csum = np.concatenate([[0.0], np.cumsum(revenue)])                       # csum[k] = sum(revenue[:k])
    inv = np.concatenate([[0.0], np.cumsum(ts["unique_invoices"].to_numpy(float))])
    views = np.concatenate([[0.0], np.cumsum(ts["total_views"].to_numpy(float))])

    idx = np.arange(MIN_HISTORY, n + 1)                                      # n == the day after the last day
    feats = {f"previous_{w}": csum[idx] - csum[idx - w] for w in WINDOWS}

    lo, hi = np.clip(idx - 365, 0, None), np.clip(idx - 365 + HORIZON, 0, None)
    has_year = idx >= 365
    feats["previous_year"] = np.where(has_year, csum[hi] - csum[lo], 0.0)
    feats["has_previous_year"] = has_year.astype(float)
    feats["recent_invoices"] = (inv[idx] - inv[idx - 30]) / 30.0
    feats["recent_views"] = (views[idx] - views[idx - 30]) / 30.0

    X = pd.DataFrame(feats)[FEATURE_NAMES]
    end = np.minimum(idx + HORIZON, n + 0)
    y = np.where(idx + HORIZON <= n, csum[np.minimum(idx + HORIZON, n)] - csum[idx], np.nan)
    dates = np.where(idx < n, day_arr[np.minimum(idx, n - 1)], day_arr[-1] + np.timedelta64(1, "D"))
    dates = dates.astype("datetime64[D]")

    if training:
        keep = np.isfinite(y)
        X, y, dates = X[keep].reset_index(drop=True), y[keep], dates[keep]
    return X, y, dates
