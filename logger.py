"""Train and predict logging (CSV files), safe to call from several threads.

Pass ``test=True`` to write to separate ``*-test.log.csv`` files so that unit tests
and dry runs never pollute the production logs.
"""
import csv
import os
import threading

import config

_LOCK = threading.Lock()

TRAIN_FIELDS = ["timestamp", "country", "train_start", "train_end", "rmse", "baseline_rmse", "runtime",
                "model_version", "model_name", "note"]
PREDICT_FIELDS = ["timestamp", "country", "target_date", "y_pred", "runtime", "model_version",
                  "in_training_window", "drift_flag", "drift_max_z", "drift_feature"]
_FIELDS = {"train": TRAIN_FIELDS, "predict": PREDICT_FIELDS}


def _path(kind, test=False):
    if kind not in _FIELDS:
        raise ValueError(f"unknown log kind '{kind}', expected one of {sorted(_FIELDS)}")
    suffix = "-test" if test else ""
    return os.path.join(config.log_dir(), f"{kind}{suffix}.log.csv")


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(kind, row, test):
    path = _path(kind, test)
    with _LOCK:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FIELDS[kind])
            if new_file:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in _FIELDS[kind]})


def update_train_log(country, date_range, metrics, runtime, model_version, note, test=False,
                     model_name=""):
    _append("train", {
        "timestamp": _now(), "country": country, "train_start": date_range[0], "train_end": date_range[1],
        "rmse": metrics.get("rmse"), "baseline_rmse": metrics.get("baseline_rmse", ""), "runtime": runtime,
        "model_version": model_version, "model_name": model_name, "note": note}, test)


def update_predict_log(country, y_pred, target_date, runtime, model_version, drift=None,
                       in_training_window=None, test=False):
    drift = drift or {}
    _append("predict", {
        "timestamp": _now(), "country": country, "target_date": target_date, "y_pred": y_pred,
        "runtime": runtime, "model_version": model_version, "in_training_window": in_training_window,
        "drift_flag": drift.get("flag", ""), "drift_max_z": drift.get("max_z", ""),
        "drift_feature": drift.get("feature", "")}, test)


def read_log(kind, test=False, tail=None):
    """Return the log as a list of dict rows (empty list if nothing has been logged yet)."""
    path = _path(kind, test)
    if not os.path.exists(path):
        return []
    with _LOCK, open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-int(tail):] if tail else rows


def log_path(kind, test=False):
    return _path(kind, test)
