"""Shared fixtures.

Every test runs against SYNTHETIC data written to a temp directory and against
temp model/log directories, so the unit tests can never read or overwrite the
production models, logs or the real course data.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_month_records(dates, rng, variant, counter):
    """Build invoice records for the given dates using one of the two real schema variants."""
    countries = {"United Kingdom": 30, "Germany": 6, "France": 4}
    records = []
    for day in dates:
        if day.dayofweek == 5:  # the real data has (almost) no Saturday trading
            continue
        trend = 1.0 + (day - pd.Timestamp("2018-01-01")).days / 365.0
        for country, base in countries.items():
            for _ in range(int(base * trend)):
                counter[0] += 1
                invoice = str(500000 + counter[0] // 3)
                if rng.random() < 0.05:
                    invoice = "C" + invoice  # cancellation, as in the real data
                price = round(float(rng.lognormal(1.2, 0.8)), 2)
                rec = {
                    "country": country,
                    "customer_id": None if rng.random() < 0.2 else float(rng.integers(10000, 10500)),
                    "invoice": invoice,
                    "year": str(day.year),
                    "month": f"{day.month:02d}",
                    "day": f"{day.day:02d}",
                }
                views = int(rng.integers(0, 20))
                stream = str(int(rng.integers(20000, 20300)))
                if variant == 0:
                    rec.update({"price": price, "stream_id": stream, "times_viewed": views})
                else:  # the alternative column names found in the real files
                    rec.update({"total_price": price, "StreamID": stream, "TimesViewed": views})
                records.append(rec)
    return records


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory):
    """Twelve monthly JSON files (2018) with both real schema variants and one bad-debt row."""
    data_dir = tmp_path_factory.mktemp("cs-train")
    rng = np.random.default_rng(0)
    counter = [0]
    for m in range(1, 13):
        dates = pd.date_range(f"2018-{m:02d}-01", periods=pd.Period(f"2018-{m:02d}").days_in_month)
        records = _make_month_records(dates, rng, variant=m % 2, counter=counter)
        if m == 1:  # an accounting adjustment that must be excluded from revenue
            adj = {"country": "United Kingdom", "customer_id": None, "invoice": "A999999", "year": "2018",
                   "month": "01", "day": "10", "price": -5000.0, "stream_id": "B", "times_viewed": 1}
            records.append(adj)
        with open(os.path.join(data_dir, f"invoices-2018-{m:02d}.json"), "w") as fh:
            json.dump(records, fh)
    return str(data_dir)


@pytest.fixture()
def isolated_env(tmp_path, monkeypatch, synthetic_data_dir):
    """Point models, logs and data at temp dirs for the duration of one test."""
    model_dir, log_dir = tmp_path / "models", tmp_path / "logs"
    monkeypatch.setenv("AAVAIL_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("AAVAIL_LOG_DIR", str(log_dir))
    monkeypatch.setenv("AAVAIL_DATA_DIR", synthetic_data_dir)
    monkeypatch.setenv("AAVAIL_SELECTION_FILE", str(tmp_path / "no-selection.json"))   # defaults only
    import model
    model.clear_model_cache()
    yield {"model_dir": model_dir, "log_dir": log_dir, "data_dir": synthetic_data_dir}
    model.clear_model_cache()


@pytest.fixture()
def trained_env(isolated_env):
    """An isolated environment in which the models have already been trained (test mode)."""
    import model
    model.model_train(test=True)
    return isolated_env
