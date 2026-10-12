import json
import pytest

import app as app_module
import logger


@pytest.fixture()
def client(isolated_env):
    application = app_module.create_app()
    application.config["TESTING"] = True
    return application.test_client()


@pytest.fixture()
def trained_client(client):
    assert client.post("/train", json={"test": True}).status_code == 200
    return client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200 and resp.get_json()["status"] == "ok"


def test_predict_before_training_returns_503(client):
    resp = client.get("/predict?country=all&date=2018-10-15&test=true")
    assert resp.status_code == 503


def test_train_endpoint_trains_models(client):
    resp = client.post("/train", json={"test": True})
    body = resp.get_json()
    assert resp.status_code == 200
    assert set(body["trained"]) == {"all", "united_kingdom"}


def test_predict_single_country(trained_client):
    resp = trained_client.get("/predict?country=united_kingdom&date=2018-10-15&test=true")
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["country"] == "united_kingdom" and body["y_pred"] > 0


def test_predict_all_countries_combined(trained_client):
    resp = trained_client.get("/predict?country=all&date=2018-10-15&test=true")
    assert resp.status_code == 200 and resp.get_json()["country"] == "all"


def test_predict_accepts_json_post(trained_client):
    resp = trained_client.post("/predict?test=true", json={"country": "all", "date": "2018-10-15"})
    assert resp.status_code == 200


def test_predict_batch(trained_client):
    queries = [{"country": "all", "date": "2018-10-15"}, {"country": "united_kingdom", "date": "2018-10-16"},
               {"country": "atlantis", "date": "2018-10-16"}]
    resp = trained_client.post("/predict?test=true", json={"queries": queries})
    body = resp.get_json()
    assert resp.status_code == 200 and len(body["results"]) == 3
    assert "y_pred" in body["results"][0] and "error" in body["results"][2]   # one bad query does not fail the batch


@pytest.mark.parametrize("query,status", [
    ("country=all&date=15-10-2018", 400),        # bad date format
    ("country=all&date=2030-01-01", 400),        # outside available range
    ("country=atlantis&date=2018-10-15", 404),   # unknown country
    ("date=2018-10-15", 400),                    # missing country
    ("country=all", 400),                        # missing date
])
def test_predict_input_errors(trained_client, query, status):
    resp = trained_client.get(f"/predict?{query}&test=true")
    assert resp.status_code == status
    assert "error" in resp.get_json()


def test_logfile_endpoints(trained_client):
    trained_client.get("/predict?country=all&date=2018-10-15&test=true")
    train = trained_client.get("/logfile/train?test=true").get_json()
    pred = trained_client.get("/logfile/predict?test=true").get_json()
    assert len(train["rows"]) == 2 and len(pred["rows"]) == 1


def test_logfile_tail_and_csv(trained_client):
    for d in ("2018-10-15", "2018-10-16", "2018-10-17"):
        trained_client.get(f"/predict?country=all&date={d}&test=true")
    assert len(trained_client.get("/logfile/predict?test=true&tail=2").get_json()["rows"]) == 2
    csv_resp = trained_client.get("/logfile/predict?test=true&format=csv")
    assert csv_resp.mimetype == "text/csv" and csv_resp.get_data(as_text=True).startswith("timestamp")


def test_logfile_unknown_kind(client):
    assert client.get("/logfile/nonsense").status_code == 404


def test_monitor_reports_performance_and_drift(trained_client):
    for d in ("2018-11-01", "2018-11-02", "2018-11-03"):
        trained_client.get(f"/predict?country=all&date={d}&test=true")
    body = trained_client.get("/monitor?test=true").get_json()
    assert body["n_predictions"] == 3
    assert "drift_flag_rate" in body and "by_country" in body
    assert body["by_country"]["all"]["n_evaluated"] >= 0


def test_models_endpoint_lists_metadata(trained_client):
    body = trained_client.get("/models?test=true").get_json()
    assert {m["country"] for m in body["models"]} == {"all", "united_kingdom"}
    assert "holdout_rmse" in body["models"][0]


def test_concurrent_train_requests_are_serialised(trained_client):
    """A second training request while one is running must be refused, not run in parallel."""
    assert app_module.TRAIN_LOCK.acquire(blocking=False)
    try:
        assert trained_client.post("/train", json={"test": True}).status_code == 409
    finally:
        app_module.TRAIN_LOCK.release()
