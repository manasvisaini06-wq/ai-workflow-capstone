"""Replay the production period through the API, day by day, exactly as a live deployment would be used.

    python simulate_production.py --mode retrain     # weekly retraining (the deployed policy)
    python simulate_production.py --mode frozen      # models trained once, never refreshed (comparison)

The invoice files of ``data/cs-train`` and ``data/cs-production`` are exposed together through
``data/combined`` (symlinks). On every simulated day D the API is asked for the forecast of the 30 days
starting at D, using data up to D-1 (the features never look forward). In ``retrain`` mode the models
are retrained through the ``/train`` endpoint every Monday with ``as_of`` = that day, so each model only
ever sees the past. Forecasts are logged by the API itself (logs/predict.log.csv).

The last day that can be scored is the last data day minus 29 (2019-11-07 for these data); the later
days are real forecasts whose actuals do not exist yet.
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

import config

BASE = config.BASE_DIR
SERIES = None


def link_combined():
    combined = os.path.join(BASE, "data", "combined")
    shutil.rmtree(combined, ignore_errors=True)
    os.makedirs(combined)
    for sub in ("cs-train", "cs-production"):
        src = os.path.join(BASE, "data", sub)
        if not os.path.isdir(src):
            sys.exit(f"missing {src}: copy the course data folders into data/ first (see README)")
        for f in sorted(os.listdir(src)):
            if f.endswith(".json"):
                os.symlink(os.path.join(src, f), os.path.join(combined, f))
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrain", "frozen"], default="retrain")
    ap.add_argument("--start", default="2019-08-01")
    ap.add_argument("--retrain-weekday", type=int, default=0, help="0 = Monday")
    args = ap.parse_args()

    combined = link_combined()
    out = os.path.join(BASE, "logs") if args.mode == "retrain" else os.path.join(BASE, "logs", "frozen")
    models = os.path.join(BASE, "models") if args.mode == "retrain" else os.path.join(BASE, "models_frozen")
    for d in (out, models):
        shutil.rmtree(d, ignore_errors=True)
    os.environ.update(AAVAIL_DATA_DIR=combined, AAVAIL_LOG_DIR=out, AAVAIL_MODEL_DIR=models)

    import ingestion
    from app import app

    ts = ingestion.get_ts(combined)
    last_day = np.datetime64(ts["all"]["date"].max(), "D")
    last_scorable = last_day - np.timedelta64(29, "D")
    days = list(np.arange(np.datetime64(args.start, "D"), last_scorable + np.timedelta64(1, "D")))
    client = app.test_client()
    t0 = time.time()

    print(f"mode={args.mode}  data through {last_day}  simulating {days[0]} .. {days[-1]} ({len(days)} days)")
    r = client.post("/train", json={"as_of": args.start})
    assert r.status_code == 200, r.get_json()
    series = sorted(r.get_json()["trained"])
    n_trains, n_calls, lat = 1, 0, []
    for day in days:
        d = np.datetime64(day, "D")
        if args.mode == "retrain" and str(d) != args.start and d.astype(object).weekday() == args.retrain_weekday:
            r = client.post("/train", json={"as_of": str(d)})
            assert r.status_code == 200, r.get_json()
            n_trains += 1
        s = time.perf_counter()
        r = client.post("/predict", json={"queries": [{"country": c, "date": str(d)} for c in series]})
        lat.append(time.perf_counter() - s)
        assert r.status_code == 200, r.get_json()
        bad = [x for x in r.get_json()["results"] if "error" in x]
        assert not bad, bad[:2]
        n_calls += len(series)
    print(f"done: {n_trains} training runs, {n_calls} forecasts, "
          f"median batch latency {np.median(lat) * 1000:.0f} ms, total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
