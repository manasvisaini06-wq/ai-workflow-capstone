"""Model training, loading and prediction for the 30-day revenue forecast.

One model is trained per series (``all`` countries plus the top ten by revenue). The model
family and its hyper-parameters were chosen in ``model_comparison.py`` (see REPORT.md).

Training is leakage-safe and re-runnable: ``model_train(as_of=...)`` uses only data observed
before ``as_of`` and only rows whose 30-day target had fully elapsed by then, which lets the
production simulation re-train on a schedule exactly as a live system would.
"""
import os
import time
from datetime import datetime

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
import ingestion
import logger
from features import FEATURE_NAMES, HORIZON, MIN_HISTORY, engineer_features

MODEL_VERSION = "1.0"
MODEL_VERSION_NOTE = "per-country gradient boosting on trailing-revenue features, 30-day horizon"
DEFAULT_MODEL = "gradient_boosting"
MIN_TRAIN_ROWS = 60
DRIFT_Z_THRESHOLD = 4.0
TEST_COUNTRIES = ("all", "united_kingdom")


class ModelNotFoundError(Exception):
    pass


class PredictInputError(ValueError):
    pass


class UnknownCountryError(PredictInputError):
    pass


class DateOutOfRangeError(PredictInputError):
    pass


# ------------------------------------------------------------------ candidates
YEAR_FEATURES = ["previous_year", "has_previous_year"]
MODEL_NAMES = ["baseline_naive", "ridge", "random_forest", "gradient_boosting", "ridge_no_year",
               "random_forest_no_year", "gradient_boosting_no_year", "ridge_ratio", "blend_ridge"]


class NaiveBaseline(BaseEstimator, RegressorMixin):
    """Last 28 days of revenue scaled to 30 days. Deployable, so a series where no learned model
    beats this benchmark is served by the benchmark instead of a worse model."""

    def fit(self, X, y=None):
        self.is_fitted_ = True          # nothing to learn, but sklearn requires a fitted marker
        return self

    def predict(self, X):
        return baseline_predict(X)


class RatioRidge(BaseEstimator, RegressorMixin):
    """Scale-free model: predicts the ratio (next-30-day revenue / baseline) from ratios of the recent
    revenue windows, then multiplies back by the baseline. Because the inputs and target carry no
    absolute level, a series with a few big days cannot dominate the fit, which helps small countries."""

    def __init__(self, alpha=30.0):
        self.alpha = alpha

    @staticmethod
    def _feats(X):
        b = (X["previous_28"] / 28.0).clip(lower=1e-9)
        F = pd.DataFrame({"r7": (X["previous_7"] / 7.0) / b, "r14": (X["previous_14"] / 14.0) / b,
                          "r70": (X["previous_70"] / 70.0) / b, "inv": X["recent_invoices"] / b,
                          "views": X["recent_views"] / b})
        return F.clip(-5, 20)

    def fit(self, X, y):
        base = baseline_predict(X)
        ok = base > 0
        F = self._feats(X[ok])
        self.scaler_ = StandardScaler().fit(F)
        self.reg_ = Ridge(alpha=self.alpha).fit(self.scaler_.transform(F), np.clip(np.asarray(y)[ok] / base[ok], 0, 5))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        ratio = np.clip(self.reg_.predict(self.scaler_.transform(self._feats(X))), 0, 5)
        return baseline_predict(X) * ratio


class BlendWithBaseline(BaseEstimator, RegressorMixin):
    """Shrinks a learned model toward the baseline: w * model + (1 - w) * baseline."""

    def __init__(self, estimator=None, weight=0.5):
        self.estimator = estimator
        self.weight = weight

    def fit(self, X, y):
        self.estimator_ = self.estimator.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.weight * self.estimator_.predict(X) + (1 - self.weight) * baseline_predict(X)


def build_estimator(name=DEFAULT_MODEL, seed=42):
    """A sklearn pipeline (column selection -> scaling -> regressor) for one of the compared models.

    A ``_no_year`` suffix drops the two prior-year features. The data contain only one full seasonal
    cycle, so those features can only be learned from a single year; the comparison tests whether
    they help or hurt.
    """
    base = name.replace("_no_year", "")
    if name == "ridge_ratio":
        return Pipeline([("ratio", RatioRidge())])
    if name == "blend_ridge":
        return Pipeline([("blend", BlendWithBaseline(build_estimator("ridge_no_year"), 0.5))])
    if base == "baseline_naive":
        return Pipeline([("baseline", NaiveBaseline())])
    if base == "ridge":
        reg = Ridge(alpha=10.0)
    elif base == "random_forest":
        reg = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=1)
    elif base == "gradient_boosting":
        reg = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                                        random_state=seed)
    else:
        raise ValueError(f"unknown model '{name}'; choose from {MODEL_NAMES}")
    keep = [f for f in FEATURE_NAMES if not (name.endswith("_no_year") and f in YEAR_FEATURES)]
    select = ColumnTransformer([("keep", "passthrough", keep)])
    return Pipeline([("select", select), ("scaler", StandardScaler()), ("reg", reg)])


def load_selection():
    """Per-series model choice written by model_comparison.py ({} if none, so defaults apply)."""
    path = config.selection_path()
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh).get("selection", {})
    return {}


def baseline_predict(X):
    """Naive benchmark: the last 28 days of revenue scaled to 30 days."""
    return X["previous_28"].to_numpy(float) * HORIZON / 28.0


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# -------------------------------------------------------------------- training
def _prefix(test):
    return "test-" if test else "sl-"


def _model_path(country, test=False):
    return os.path.join(config.model_dir(), f"{_prefix(test)}{country}-{MODEL_VERSION.replace('.', '_')}.joblib")


def _fit_one(ts, country, test, model_name):
    start = time.perf_counter()
    X, y, dates = engineer_features(ts, training=True)
    if len(X) < MIN_TRAIN_ROWS:
        raise ValueError(f"only {len(X)} labelled rows for '{country}' (need {MIN_TRAIN_ROWS})")

    # Honest hold-out score: last 20% of rows, with a 30-day gap so no target window overlaps training.
    n_test = max(20, int(0.2 * len(X)))
    cut = len(X) - n_test
    tr = np.arange(len(X)) < cut - (HORIZON - 1)
    te = np.arange(len(X)) >= cut
    holdout = build_estimator(model_name).fit(X[tr], y[tr])
    holdout_rmse = rmse(y[te], holdout.predict(X[te]))
    baseline_rmse = rmse(y[te], baseline_predict(X[te]))

    final = build_estimator(model_name).fit(X, y)       # refit on everything for deployment
    meta = {
        "country": country, "model_name": model_name, "model_version": MODEL_VERSION,
        "note": MODEL_VERSION_NOTE, "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_start": str(dates[0]), "train_end": str(dates[-1]), "n_rows": int(len(X)),
        "holdout_rmse": holdout_rmse, "baseline_rmse": baseline_rmse, "feature_names": list(FEATURE_NAMES),
        "feature_mean": X.mean().to_dict(), "feature_std": X.std(ddof=0).to_dict(),
    }
    os.makedirs(config.model_dir(), exist_ok=True)
    joblib.dump({"model": final, "meta": meta}, _model_path(country, test))
    runtime = time.perf_counter() - start
    m, s = divmod(runtime, 60)
    h, m = divmod(m, 60)
    logger.update_train_log(country, (meta["train_start"], meta["train_end"]),
                            {"rmse": round(holdout_rmse, 2), "baseline_rmse": round(baseline_rmse, 2)},
                            "%02d:%02d:%02d" % (h, m, s), MODEL_VERSION, MODEL_VERSION_NOTE, test=test,
                            model_name=model_name)
    return meta


def model_train(data_dir=None, countries=None, test=False, as_of=None, model_name=None):
    """Train one model per series from every JSON file in ``data_dir``.

    countries : optional iterable of series ids to train (default: all; test mode: a small subset)
    test      : write ``test-`` model files and ``*-test`` logs, and train only a subset
    as_of     : 'YYYY-MM-DD'; train only on data observed before this date (for scheduled retraining)
    model_name: force one model family for every series (default: the per-series choice made in
                model_comparison.py, falling back to DEFAULT_MODEL)
    Returns {series_id: meta}.
    """
    ts_all = ingestion.get_ts(data_dir)
    selected = list(countries) if countries else (list(TEST_COUNTRIES) if test else list(ts_all))
    unknown = [c for c in selected if c not in ts_all]
    if unknown:
        raise UnknownCountryError(f"no data for: {unknown}")
    cutoff = np.datetime64(as_of, "D") if as_of else None
    selection = load_selection()
    summary = {}
    for country in selected:
        ts = ts_all[country]
        if cutoff is not None:
            ts = ts[ts["date"].to_numpy("datetime64[D]") < cutoff]
        summary[country] = _fit_one(ts, country, test, model_name or selection.get(country, DEFAULT_MODEL))
    clear_model_cache()
    return summary


# --------------------------------------------------------------------- loading
_MODEL_CACHE = {}


def clear_model_cache():
    _MODEL_CACHE.clear()


def model_load(test=False):
    """Load all trained models ({series_id: {"model", "meta"}}); cached until files change."""
    directory, prefix = config.model_dir(), _prefix(test)
    files = sorted(f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".joblib")) \
        if os.path.isdir(directory) else []
    if not files:
        raise ModelNotFoundError(f"no trained models found in {directory}; call /train first")
    sig = tuple((f, os.stat(os.path.join(directory, f)).st_mtime_ns) for f in files)
    key = (directory, prefix)
    if key in _MODEL_CACHE and _MODEL_CACHE[key][0] == sig:
        return _MODEL_CACHE[key][1]
    bundles = {}
    for f in files:
        bundle = joblib.load(os.path.join(directory, f))
        bundles[bundle["meta"]["country"]] = bundle
    _MODEL_CACHE[key] = (sig, bundles)
    return bundles


# ------------------------------------------------------------------- prediction
def drift_check(row, meta, z_threshold=DRIFT_Z_THRESHOLD):
    """Flag a request whose features sit far outside the training distribution (|z| > threshold)."""
    mean = pd.Series(meta["feature_mean"])
    std = pd.Series(meta["feature_std"]).replace(0, 1.0)
    z = ((row[meta["feature_names"]].iloc[0] - mean) / std).abs()
    worst = z.idxmax()
    return {"flag": bool(z.max() > z_threshold), "max_z": round(float(z.max()), 3), "feature": str(worst)}


def _parse_date(date):
    try:
        return np.datetime64(datetime.strptime(str(date), "%Y-%m-%d").date(), "D")
    except ValueError as exc:
        raise PredictInputError("date must be in YYYY-MM-DD format") from exc


def model_predict(country, date, data_dir=None, test=False, bundles=None, log=True):
    """Predict the revenue of the 30 days starting at ``date`` for one country (or ``all``).

    Works for any date from the first forecastable day up to the day after the last observation,
    so users can ask on the 15th or at month end, and the newest date is a true forecast.
    """
    start = time.perf_counter()
    cid = ingestion.country_id(country) if country else ""
    if not cid:
        raise PredictInputError("country is required")
    target = _parse_date(date)
    bundles = bundles or model_load(test=test)
    if cid not in bundles:
        raise UnknownCountryError(f"no model for '{country}'; available: {sorted(bundles)}")

    ts = ingestion.get_ts(data_dir)[cid]
    X, _, dates = engineer_features(ts, training=False)
    where = np.where(dates == target)[0]
    if where.size == 0:
        raise DateOutOfRangeError(f"date {target} not in range {dates[0]} to {dates[-1]}")
    row = X.iloc[[where[0]]]

    bundle = bundles[cid]
    y_pred = max(0.0, float(bundle["model"].predict(row)[0]))
    drift = drift_check(row, bundle["meta"])
    in_window = bool(target <= np.datetime64(bundle["meta"]["train_end"], "D"))
    runtime = time.perf_counter() - start
    result = {"country": cid, "target_date": str(target), "y_pred": y_pred, "horizon_days": HORIZON,
              "model_version": bundle["meta"]["model_version"], "model_name": bundle["meta"]["model_name"],
              "in_training_window": in_window, "drift": drift, "runtime_seconds": round(runtime, 4)}
    if log:
        logger.update_predict_log(cid, y_pred, str(target), "%.4f" % runtime, bundle["meta"]["model_version"],
                                  drift=drift, in_training_window=in_window, test=test)
    return result
