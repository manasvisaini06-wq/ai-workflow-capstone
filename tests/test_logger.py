import csv
import os
import threading

import pytest

import logger


def test_train_log_is_created_with_header_and_row(isolated_env):
    logger.update_train_log("all", ("2018-03-01", "2018-10-01"), {"rmse": 123.4}, "00:00:03", "1.0", "note")
    path = isolated_env["log_dir"] / "train.log.csv"
    assert path.exists()
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 1
    assert rows[0]["country"] == "all"
    assert float(rows[0]["rmse"]) == pytest.approx(123.4)
    assert rows[0]["train_start"] == "2018-03-01"


def test_train_log_appends(isolated_env):
    for i in range(3):
        logger.update_train_log("all", ("a", "b"), {"rmse": i}, "00:00:01", "1.0", "n")
    assert len(logger.read_log("train")) == 3


def test_predict_log_row_contains_expected_fields(isolated_env):
    logger.update_predict_log("united_kingdom", 1000.5, "2018-06-01", "00:00:01", "1.0",
                              drift={"flag": True, "max_z": 5.2, "feature": "previous_7"},
                              in_training_window=False)
    row = logger.read_log("predict")[0]
    assert row["country"] == "united_kingdom"
    assert float(row["y_pred"]) == pytest.approx(1000.5)
    assert row["target_date"] == "2018-06-01"
    assert str(row["drift_flag"]) in ("True", "true", "1")
    assert float(row["drift_max_z"]) == pytest.approx(5.2)


def test_test_flag_uses_separate_files(isolated_env):
    logger.update_train_log("all", ("a", "b"), {"rmse": 1}, "00:00:01", "1.0", "n", test=True)
    logger.update_train_log("all", ("a", "b"), {"rmse": 2}, "00:00:01", "1.0", "n", test=False)
    names = sorted(os.listdir(isolated_env["log_dir"]))
    assert "train-test.log.csv" in names and "train.log.csv" in names
    assert len(logger.read_log("train", test=True)) == 1
    assert len(logger.read_log("train", test=False)) == 1


def test_read_log_tail(isolated_env):
    for i in range(10):
        logger.update_train_log("all", ("a", "b"), {"rmse": i}, "00:00:01", "1.0", "n")
    tail = logger.read_log("train", tail=3)
    assert [float(r["rmse"]) for r in tail] == [7.0, 8.0, 9.0]


def test_read_log_missing_file_returns_empty(isolated_env):
    assert logger.read_log("predict") == []


def test_invalid_kind_raises(isolated_env):
    with pytest.raises(ValueError):
        logger.read_log("nonsense")


def test_logging_writes_only_inside_configured_dir(isolated_env, tmp_path):
    logger.update_predict_log("all", 1.0, "2018-06-01", "00:00:01", "1.0")
    assert os.listdir(isolated_env["log_dir"]) == ["predict.log.csv"]


def test_concurrent_writes_are_not_lost_or_corrupted(isolated_env):
    def work(i):
        logger.update_predict_log("all", float(i), "2018-06-01", "00:00:01", "1.0")
    threads = [threading.Thread(target=work, args=(i,)) for i in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    rows = logger.read_log("predict")
    assert len(rows) == 40
    assert sorted(float(r["y_pred"]) for r in rows) == [float(i) for i in range(40)]
