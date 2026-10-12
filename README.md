# AAVAIL revenue forecasting service

Forecasts the revenue of the **next 30 days** for all countries combined and for the ten countries with the
most revenue, behind a small Flask API. Capstone project for the IBM AI Enterprise Workflow course.
The findings are in **[REPORT.md](REPORT.md)**.

## 1. Get the data (not committed, about 160 MB)

Copy the two folders `cs-train/` and `cs-production/` from
<https://github.com/aavail/ai-workflow-capstone> (folder `data/`) into this repository's `data/` folder:

```
data/cs-train/invoices-2017-11.json ...
data/cs-production/invoices-2019-08.json ...
```

## 2. Run everything in Docker

```bash
docker build -t aavail-capstone .        # trains the models and runs all unit tests while building
docker run --rm -p 8080:8080 aavail-capstone
curl "http://localhost:8080/predict?country=all&date=2019-05-01"
```

Run only the tests inside the image: `docker run --rm aavail-capstone python run-tests.py`

## 3. Run without Docker

```bash
pip install -r requirements.txt           # Python 3.13
python run-tests.py                       # all unit tests, one script; exits non-zero on any failure
python -c "import model; model.model_train()"
python app.py                             # http://localhost:8080
```

## API

| Endpoint | What it does |
|---|---|
| `POST /train` | Retrain every model. Optional JSON `{"as_of": "2019-08-01", "countries": ["all"], "test": true}`. Returns 409 if a run is already in progress. |
| `GET /predict?country=all&date=2019-05-01` | Forecast of the 30 days starting at `date`. `country` is `all` or a country id such as `united_kingdom`. |
| `POST /predict` | `{"country": "eire", "date": "2019-05-01"}` or a batch `{"queries": [{"country": ..., "date": ...}, ...]}` (max 500). |
| `GET /logfile/train`, `/logfile/predict` | The training and prediction logs as JSON; add `?format=csv` for CSV, `?tail=N` for the last N rows. |
| `GET /models` | Which model is deployed for each series, with its hold-out error. |
| `GET /monitor` | Performance monitoring: error of logged predictions against the revenue that actually happened, plus the drift-flag rate. |
| `GET /health` | Liveness check. |

Add `?test=true` (or `"test": true`) to any call to use the isolated test models and test logs.

## Repository map

| File | Purpose |
|---|---|
| `ingestion.py` | Data ingestion and cleaning: JSON files to a daily time series. Drop a new monthly file in `data/cs-train/` and every function picks it up. |
| `features.py` | Leakage-free feature engineering (features use only days before the forecast date). |
| `model.py` | Candidate models, training (`model_train`), loading, prediction, drift check. |
| `logger.py`, `monitoring.py` | Train/predict logs and the performance monitor. |
| `app.py` | Flask API. |
| `eda.py`, `model_comparison.py` | EDA figures and the model comparison (writes `model_selection.json`). |
| `simulate_production.py` | Replays the production months through the API with weekly retraining. |
| `post_production_analysis.py` | Scores the production forecasts against actuals and the baseline. |
| `load_test.py` | Latency and throughput test of a running API. |
| `tests/` | Unit tests for the API, model, logging and monitoring. They run against synthetic data and temporary folders, never the production models or logs. |

## Reproduce the results

```bash
python eda.py
python model_comparison.py                # about 1 minute
python simulate_production.py --mode retrain
python simulate_production.py --mode frozen
python post_production_analysis.py
python load_test.py --url http://localhost:8080
```
