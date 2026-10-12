"""Unit tests for the monitoring functions, on a tiny synthetic series (no files, no production data)."""
import numpy as np
import pandas as pd
import pytest

import monitoring


@pytest.fixture()
def ts_dict():
    days = pd.date_range("2018-01-01", periods=60, freq="D")
    return {"all": pd.DataFrame({"date": days, "revenue": np.full(60, 10.0)})}


def test_actual_revenue_sums_30_days(ts_dict):
    assert monitoring.actual_revenue(ts_dict["all"], "2018-01-01") == 300.0


def test_actual_revenue_none_when_window_not_finished(ts_dict):
    assert monitoring.actual_revenue(ts_dict["all"], "2018-02-15") is None   # needs data to 2018-03-16
    assert monitoring.actual_revenue(ts_dict["all"], "2017-12-01") is None   # before the data


def test_score_excludes_in_sample_predictions(ts_dict):
    preds = [{"country": "all", "target_date": "2018-01-05", "y_pred": "330", "in_training_window": "True"},
             {"country": "all", "target_date": "2018-01-06", "y_pred": "330", "in_training_window": "False"}]
    scored = monitoring.score_predictions(preds, ts_dict)
    assert len(scored) == 1 and scored.loc[0, "target_date"] == "2018-01-06"
    assert len(monitoring.score_predictions(preds, ts_dict, only_out_of_sample=False)) == 2


def test_regression_metrics_values(ts_dict):
    preds = [{"country": "all", "target_date": "2018-01-01", "y_pred": "330", "in_training_window": "False"},
             {"country": "all", "target_date": "2018-01-02", "y_pred": "270", "in_training_window": "False"}]
    m = monitoring.regression_metrics(monitoring.score_predictions(preds, ts_dict))
    assert m["rmse"] == pytest.approx(30.0) and m["bias"] == pytest.approx(0.0) and m["mape"] == pytest.approx(10.0)


def test_unevaluable_predictions_are_dropped_and_empty_is_safe(ts_dict):
    preds = [{"country": "all", "target_date": "2018-03-01", "y_pred": "100", "in_training_window": "False"}]
    assert len(monitoring.score_predictions(preds, ts_dict)) == 0
    assert monitoring.regression_metrics(monitoring.score_predictions([], ts_dict))["n_evaluated"] == 0
