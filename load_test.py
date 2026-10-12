"""Simple load test for a running API.

    python load_test.py --url http://localhost:8080 [--requests 300] [--concurrency 1 8 32]

Sends single /predict requests (random country and date) at several concurrency levels and reports
throughput and latency percentiles, plus one 50-query batch request. Writes reports/load_test.json.
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

import config


def one(url, countries, dates):
    t = time.perf_counter()
    r = requests.get(f"{url}/predict", params={"country": random.choice(countries), "date": random.choice(dates)},
                     timeout=60)
    return r.status_code, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    a = ap.parse_args()
    models = requests.get(f"{a.url}/models", timeout=30).json()["models"]
    countries = [m["country"] for m in models]
    end = np.datetime64(min(m["train_end"] for m in models), "D")
    dates = [str(end - np.timedelta64(i, "D")) for i in range(1, 120)]
    out = {"url": a.url, "requests_per_level": a.requests, "levels": {}}
    for c in a.concurrency:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(lambda _: one(a.url, countries, dates), range(a.requests)))
        wall = time.perf_counter() - t0
        lat = np.array([x[1] for x in res]) * 1000
        out["levels"][c] = {"throughput_rps": round(a.requests / wall, 1), "errors": int(sum(x[0] != 200 for x in res)),
                            "p50_ms": round(float(np.percentile(lat, 50))), "p95_ms": round(float(np.percentile(lat, 95))),
                            "max_ms": round(float(lat.max()))}
        print(f"concurrency {c:3d}: {out['levels'][c]}")
    queries = [{"country": random.choice(countries), "date": random.choice(dates)} for _ in range(50)]
    t = time.perf_counter()
    r = requests.post(f"{a.url}/predict", json={"queries": queries}, timeout=120)
    out["batch_50_queries_ms"] = round((time.perf_counter() - t) * 1000)
    out["batch_status"] = r.status_code
    print("batch of 50:", out["batch_50_queries_ms"], "ms, status", r.status_code)
    json.dump(out, open(os.path.join(config.BASE_DIR, "reports", "load_test.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
