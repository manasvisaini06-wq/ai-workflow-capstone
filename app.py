"""Flask API for the 30-day revenue forecaster.

Endpoints
  GET  /health                       liveness probe
  POST /train                        (re)train the models      {"test": bool, "as_of": "YYYY-MM-DD", "countries": [...]}
  GET  /predict?country=&date=       one prediction            (also POST JSON, or {"queries": [...]} for a batch)
  GET  /logfile/<train|predict>      read the logs             ?tail=N  ?format=csv  ?test=true
  GET  /models                       metadata of the deployed models
  GET  /monitor                      performance vs actuals, drift rate

Add ``?test=true`` (or ``"test": true``) to use the isolated test models and test logs.
"""
import threading

import os

from flask import Flask, Response, jsonify, request

import config
import ingestion
import logger
import model
import monitoring

TRAIN_LOCK = threading.Lock()           # only one training run at a time (it replaces model files)
MAX_BATCH = 500


def _flag(value):
    return str(value).lower() in ("1", "true", "yes")


def create_app():
    app = Flask(__name__)

    @app.errorhandler(model.UnknownCountryError)
    def _unknown(exc):
        return jsonify(error=str(exc)), 404

    @app.errorhandler(model.PredictInputError)
    def _bad_input(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(model.ModelNotFoundError)
    def _no_model(exc):
        return jsonify(error=str(exc)), 503

    @app.errorhandler(ingestion.IngestionError)
    def _ingest(exc):
        return jsonify(error=str(exc)), 503

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_version=model.MODEL_VERSION)

    @app.post("/train")
    def train():
        body = request.get_json(silent=True) or {}
        test = _flag(body.get("test", request.args.get("test", False)))
        if not TRAIN_LOCK.acquire(blocking=False):
            return jsonify(error="a training run is already in progress"), 409
        try:
            summary = model.model_train(test=test, as_of=body.get("as_of"), countries=body.get("countries"))
        finally:
            TRAIN_LOCK.release()
        return jsonify(status="trained", test=test, trained=summary)

    def _predict_one(country, date, test):
        return model.model_predict(country, date, test=test)

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        test = _flag(request.args.get("test", False))
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            test = test or _flag(body.get("test", False))
            if "queries" in body:
                queries = body["queries"]
                if not isinstance(queries, list) or len(queries) > MAX_BATCH:
                    return jsonify(error=f"'queries' must be a list of at most {MAX_BATCH} items"), 400
                bundles = model.model_load(test=test)           # load once for the whole batch
                results = []
                for q in queries:
                    try:
                        results.append(model.model_predict(q.get("country"), q.get("date"), test=test,
                                                           bundles=bundles))
                    except model.PredictInputError as exc:
                        results.append({"country": q.get("country"), "date": q.get("date"), "error": str(exc)})
                return jsonify(results=results)
            country, date = body.get("country"), body.get("date")
        else:
            country, date = request.args.get("country"), request.args.get("date")
        if not country or not date:
            return jsonify(error="both 'country' and 'date' (YYYY-MM-DD) are required"), 400
        return jsonify(_predict_one(country, date, test))

    @app.get("/logfile/<kind>")
    def logfile(kind):
        if kind not in ("train", "predict"):
            return jsonify(error=f"unknown log '{kind}', use 'train' or 'predict'"), 404
        test = _flag(request.args.get("test", False))
        tail = request.args.get("tail", type=int)
        if request.args.get("format") == "csv":
            import csv
            import io
            rows = logger.read_log(kind, test=test, tail=tail)
            buf = io.StringIO()
            fields = logger.TRAIN_FIELDS if kind == "train" else logger.PREDICT_FIELDS
            writer = csv.DictWriter(buf, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            return Response(buf.getvalue(), mimetype="text/csv")
        return jsonify(log=kind, test=test, rows=logger.read_log(kind, test=test, tail=tail))

    @app.get("/models")
    def models():
        test = _flag(request.args.get("test", False))
        bundles = model.model_load(test=test)
        keep = ("country", "model_name", "model_version", "trained_at", "train_start", "train_end", "n_rows",
                "holdout_rmse", "baseline_rmse")
        return jsonify(models=[{k: b["meta"][k] for k in keep} for _, b in sorted(bundles.items())])

    @app.get("/monitor")
    def monitor():
        test = _flag(request.args.get("test", False))
        return jsonify(monitoring.monitor_report(test=test, tail=request.args.get("tail", type=int)))

    return app


app = create_app()


def warm_up():
    """Load the data and models once at start-up so the first request is not the slow one (~10 s)."""
    try:
        ingestion.get_ts()
        model.model_load()
    except Exception as exc:                       # an untrained container still starts; /train will fix it
        print(f"warm-up skipped: {exc}")


if os.environ.get("AAVAIL_WARMUP") == "1":
    warm_up()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
