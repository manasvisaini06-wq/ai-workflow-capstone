import os
import numpy as np
import pandas as pd
import pytest

import ingestion
import features
import logger
import model


# ----------------------------------------------------------------- ingestion
def test_fetch_data_normalises_column_names_and_invoices(synthetic_data_dir):
    df = ingestion.fetch_data(synthetic_data_dir)
    assert {"stream_id", "times_viewed", "price", "invoice_date"} <= set(df.columns)
    assert not {"StreamID", "TimesViewed", "total_price"} & set(df.columns)
    assert df["invoice"].str.fullmatch(r"\d+").all()          # letters removed
    assert str(df["invoice_date"].dtype).startswith("datetime64")


def test_fetch_data_excludes_bad_debt_adjustments(synthetic_data_dir):
    df = ingestion.fetch_data(synthetic_data_dir)
    assert (df["price"] < 0).sum() == 0
    raw = ingestion.fetch_data(synthetic_data_dir, exclude_adjustments=False)
    assert (raw["price"] < 0).sum() == 1


def test_fetch_data_flags_cancellations(synthetic_data_dir):
    df = ingestion.fetch_data(synthetic_data_dir)
    assert set(df["invoice_type"].unique()) <= {"sale", "cancellation", "adjustment"}
    assert (df["invoice_type"] == "cancellation").any()


def test_fetch_data_missing_directory(tmp_path):
    with pytest.raises(ingestion.IngestionError):
        ingestion.fetch_data(str(tmp_path / "does-not-exist"))


def test_fetch_data_empty_directory(tmp_path):
    with pytest.raises(ingestion.IngestionError):
        ingestion.fetch_data(str(tmp_path))


def test_fetch_data_rejects_file_with_missing_columns(tmp_path):
    (tmp_path / "bad.json").write_text('[{"country": "United Kingdom", "price": 1.0}]')
    with pytest.raises(ingestion.IngestionError):
        ingestion.fetch_data(str(tmp_path))


def test_convert_to_ts_is_continuous_and_conserves_revenue(synthetic_data_dir):
    df = ingestion.fetch_data(synthetic_data_dir)
    ts = ingestion.convert_to_ts(df)
    days = ts["date"].to_numpy("datetime64[D]")
    assert (np.diff(days) == np.timedelta64(1, "D")).all()      # no gaps: Saturdays filled with 0
    assert ts["revenue"].sum() == pytest.approx(df["price"].sum())
    assert (ts["revenue"] == 0).any()


def test_convert_to_ts_country_uses_same_date_range(synthetic_data_dir):
    df = ingestion.fetch_data(synthetic_data_dir)
    assert len(ingestion.convert_to_ts(df, "Germany")) == len(ingestion.convert_to_ts(df))


def test_convert_to_ts_unknown_country(synthetic_data_dir):
    df = ingestion.fetch_data(synthetic_data_dir)
    with pytest.raises(ingestion.IngestionError):
        ingestion.convert_to_ts(df, "Atlantis")


def test_fetch_ts_returns_all_and_country_series(synthetic_data_dir):
    ts = ingestion.fetch_ts(synthetic_data_dir)
    assert {"all", "united_kingdom", "germany", "france"} <= set(ts)


def test_country_id():
    assert ingestion.country_id("United Kingdom") == "united_kingdom"
    assert ingestion.country_id("united_kingdom") == "united_kingdom"
    assert ingestion.country_id("  EIRE ") == "eire"


# ------------------------------------------------------------------ features
def _flat_ts(n, spike_day=None, spike=1000.0):
    dates = pd.date_range("2018-01-01", periods=n)
    rev = np.zeros(n)
    if spike_day is not None:
        rev[spike_day] = spike
    return pd.DataFrame({"date": dates, "revenue": rev, "unique_invoices": 1.0, "total_views": 1.0,
                         "purchases": 1, "unique_streams": 1})


def test_features_do_not_leak_the_current_or_future_day():
    ts = _flat_ts(200, spike_day=100)
    X, y, dates = features.engineer_features(ts, training=False)
    on_spike = X[dates == np.datetime64("2018-04-11")]            # day index 100
    after = X[dates == np.datetime64("2018-04-12")]
    assert on_spike["previous_7"].iloc[0] == 0                     # spike not visible on its own day
    assert after["previous_7"].iloc[0] == 1000.0                   # visible the next day


def test_target_is_next_30_days_inclusive():
    ts = _flat_ts(200, spike_day=100)
    X, y, dates = features.engineer_features(ts, training=True)
    y_at = dict(zip(dates, y))
    assert y_at[np.datetime64("2018-04-11")] == 1000.0             # day 100 itself is inside its window
    assert y_at[np.datetime64("2018-03-13")] == 1000.0             # day 71: window covers day 100
    assert y_at[np.datetime64("2018-03-12")] == 0.0                # day 70: window ends day 99


def test_training_flag_drops_rows_without_a_complete_target():
    ts = _flat_ts(200)
    Xt, yt, dt = features.engineer_features(ts, training=True)
    Xa, ya, da = features.engineer_features(ts, training=False)
    assert np.isfinite(yt).all()
    assert len(Xa) - len(Xt) == features.HORIZON                   # 29 incomplete days + the next, unseen day
    assert da[-1] == np.datetime64("2018-07-19") + np.timedelta64(1, "D")   # day after the last data day


def test_features_require_minimum_history():
    with pytest.raises(ValueError):
        features.engineer_features(_flat_ts(features.MIN_HISTORY - 1))


def test_feature_columns_are_stable():
    X, _, _ = features.engineer_features(_flat_ts(200), training=False)
    assert list(X.columns) == features.FEATURE_NAMES


# ------------------------------------------------------------ model: training
def test_baseline_prediction_scales_last_28_days():
    X = pd.DataFrame({"previous_28": [280.0]})
    assert model.baseline_predict(X)[0] == pytest.approx(300.0)


def test_train_saves_models_and_logs_inside_isolated_dirs(isolated_env):
    summary = model.model_train(test=True)
    assert set(summary) == {"all", "united_kingdom"}               # test mode trains a subset
    saved = os.listdir(isolated_env["model_dir"])
    assert any(f.startswith("test-all-") for f in saved)
    rows = logger.read_log("train", test=True)
    assert len(rows) == 2
    assert np.isfinite(float(rows[0]["rmse"]))


def test_train_does_not_touch_production_dirs(isolated_env):
    prod_models = os.path.join(os.path.dirname(model.__file__), "models")
    prod_logs = os.path.join(os.path.dirname(model.__file__), "logs")
    before = [sorted(os.listdir(d)) if os.path.isdir(d) else None for d in (prod_models, prod_logs)]
    model.model_train(test=True)
    after = [sorted(os.listdir(d)) if os.path.isdir(d) else None for d in (prod_models, prod_logs)]
    assert before == after


def test_train_as_of_never_uses_labels_from_the_future(isolated_env):
    summary = model.model_train(test=True, as_of="2018-09-01")
    for meta in summary.values():
        # the newest training row's 30-day target window must end before the as_of date
        assert np.datetime64(meta["train_end"]) + np.timedelta64(model.HORIZON - 1, "D") < np.datetime64("2018-09-01")


def test_train_reports_baseline_for_comparison(isolated_env):
    meta = model.model_train(test=True)["all"]
    assert meta["baseline_rmse"] > 0 and meta["holdout_rmse"] > 0


def test_model_load_raises_when_nothing_is_trained(isolated_env):
    with pytest.raises(model.ModelNotFoundError):
        model.model_load()


# ----------------------------------------------------------- model: predict
def test_predict_returns_non_negative_float(trained_env):
    out = model.model_predict("united_kingdom", "2018-10-15", test=True)
    assert isinstance(out["y_pred"], float) and out["y_pred"] >= 0
    assert out["country"] == "united_kingdom"


def test_predict_works_for_all_countries_combined(trained_env):
    assert model.model_predict("all", "2018-10-15", test=True)["y_pred"] > 0


def test_predict_accepts_display_name(trained_env):
    out = model.model_predict("United Kingdom", "2018-10-15", test=True)
    assert out["country"] == "united_kingdom"


def test_predict_can_forecast_the_day_after_the_last_observation(trained_env):
    out = model.model_predict("all", "2019-01-01", test=True)     # data ends 2018-12-31
    assert out["y_pred"] > 0


def test_predict_flags_in_sample_dates(trained_env):
    out = model.model_predict("all", "2018-05-01", test=True)
    assert out["in_training_window"] is True


def test_predict_rejects_bad_date_format(trained_env):
    with pytest.raises(model.PredictInputError):
        model.model_predict("all", "15/10/2018", test=True)


def test_predict_rejects_date_outside_range(trained_env):
    with pytest.raises(model.DateOutOfRangeError):
        model.model_predict("all", "2030-01-01", test=True)
    with pytest.raises(model.DateOutOfRangeError):
        model.model_predict("all", "2018-01-05", test=True)       # before enough history exists


def test_predict_rejects_unknown_country(trained_env):
    with pytest.raises(model.UnknownCountryError):
        model.model_predict("atlantis", "2018-10-15", test=True)


def test_predict_writes_to_the_test_log_only(trained_env):
    model.model_predict("all", "2018-10-15", test=True)
    assert len(logger.read_log("predict", test=True)) == 1
    assert logger.read_log("predict", test=False) == []


# -------------------------------------------------------------------- drift
def test_drift_check_flags_extreme_inputs(trained_env):
    bundle = model.model_load(test=True)["all"]
    meta = bundle["meta"]
    normal = pd.DataFrame([meta["feature_mean"]])[meta["feature_names"]]
    extreme = normal.copy()
    extreme["previous_28"] = meta["feature_mean"]["previous_28"] + 50 * max(meta["feature_std"]["previous_28"], 1)
    assert model.drift_check(normal, meta)["flag"] is False
    flagged = model.drift_check(extreme, meta)
    assert flagged["flag"] is True and flagged["feature"] == "previous_28"


# ------------------------------------------------------- model selection / variants
@pytest.mark.parametrize("name", model.MODEL_NAMES)
def test_every_model_variant_fits_and_predicts(name):
    X = pd.DataFrame({f: np.random.default_rng(1).normal(size=80) for f in features.FEATURE_NAMES})
    y = X["previous_28"] * 3 + 100
    est = model.build_estimator(name).fit(X, y)
    assert est.predict(X.iloc[:3]).shape == (3,)


def test_no_year_variant_ignores_year_features():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({f: rng.normal(size=80) for f in features.FEATURE_NAMES})
    y = X["previous_28"] * 3
    est = model.build_estimator("ridge_no_year").fit(X, y)
    changed = X.copy()
    changed["previous_year"] += 1000
    assert np.allclose(est.predict(X), est.predict(changed))


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError):
        model.build_estimator("magic")


def test_train_uses_per_series_selection(isolated_env, tmp_path, monkeypatch):
    import json
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"selection": {"all": "ridge", "united_kingdom": "random_forest_no_year"}}))
    monkeypatch.setenv("AAVAIL_SELECTION_FILE", str(sel))
    summary = model.model_train(test=True)
    assert summary["all"]["model_name"] == "ridge"
    assert summary["united_kingdom"]["model_name"] == "random_forest_no_year"
