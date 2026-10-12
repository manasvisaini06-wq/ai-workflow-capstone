"""Data ingestion: JSON invoice files -> clean invoice table -> daily time-series.

``fetch_ts`` is the single entry point used by training, prediction and monitoring, so
the whole process can be automated by dropping new monthly JSON files in the data folder.

Cleaning decisions (all visible in the returned frame, none silent):
  * the four real column spellings are normalised (StreamID/TimesViewed/total_price),
  * letters are stripped from invoice ids so cancellations (``C...``) match their originals,
  * accounting adjustments (invoice prefix ``A``, stream ``B``: bad-debt write-offs of up to
    -53,594) are excluded from revenue because they are not sales,
  * cancellations are KEPT as recorded: their prices are positive in the source, so their sign
    is ambiguous. They are labelled in ``invoice_type`` so the business can decide.
"""
import os
import re

import numpy as np
import pandas as pd

import config

REQUIRED_COLUMNS = ["country", "customer_id", "day", "invoice", "month", "price", "stream_id",
                    "times_viewed", "year"]
COLUMN_ALIASES = {"StreamID": "stream_id", "TimesViewed": "times_viewed", "total_price": "price"}
TS_COLUMNS = ["date", "purchases", "unique_invoices", "unique_streams", "total_views", "revenue"]


class IngestionError(Exception):
    """Raised for any problem with the input files that the caller should see clearly."""


def country_id(name):
    """'United Kingdom' -> 'united_kingdom' (also accepts an id unchanged)."""
    return re.sub(r"\s+", "_", str(name).strip().lower())


def _normalise_columns(frame):
    """Map alternative column spellings onto the canonical names.

    If a file mixes spellings (some rows ``price``, others ``total_price``) the values are
    combined into one column instead of creating duplicate columns.
    """
    for alias, target in COLUMN_ALIASES.items():
        if alias not in frame.columns:
            continue
        if target in frame.columns:
            frame[target] = frame[target].combine_first(frame[alias])
            frame = frame.drop(columns=[alias])
        else:
            frame = frame.rename(columns={alias: target})
    return frame


def load_json_files(data_dir):
    """Read every ``*.json`` file in ``data_dir`` and normalise the column names."""
    if not data_dir or not os.path.isdir(data_dir):
        raise IngestionError(f"data directory does not exist: {data_dir}")
    files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".json"))
    if not files:
        raise IngestionError(f"no .json files found in {data_dir}")
    frames = []
    for name in files:
        try:
            frame = pd.read_json(os.path.join(data_dir, name), dtype=False)
        except ValueError as exc:
            raise IngestionError(f"could not parse {name}: {exc}") from exc
        frame = _normalise_columns(frame)
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise IngestionError(f"{name} is missing required columns: {missing}")
        frames.append(frame[REQUIRED_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def clean(df, exclude_adjustments=True):
    """Type-convert, build dates, label invoice types and drop rows that cannot be used."""
    df = df.copy()
    raw_invoice = df["invoice"].astype(str)
    df["invoice_type"] = np.select([raw_invoice.str.startswith("C"), raw_invoice.str.startswith("A")],
                                   ["cancellation", "adjustment"], default="sale")
    df["invoice"] = raw_invoice.str.replace(r"\D+", "", regex=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["times_viewed"] = pd.to_numeric(df["times_viewed"], errors="coerce").fillna(0)
    df["stream_id"] = df["stream_id"].astype(str)
    df["country"] = df["country"].astype(str).str.strip()
    df["invoice_date"] = pd.to_datetime({"year": pd.to_numeric(df["year"], errors="coerce"),
                                         "month": pd.to_numeric(df["month"], errors="coerce"),
                                         "day": pd.to_numeric(df["day"], errors="coerce")}, errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["price", "invoice_date"])
    if exclude_adjustments:
        df = df[df["invoice_type"] != "adjustment"]
    df = df.sort_values("invoice_date", kind="stable").reset_index(drop=True)
    df.attrs["rows_dropped"] = n_before - len(df)
    return df


def fetch_data(data_dir=None, exclude_adjustments=True):
    """Load and clean all invoice files in ``data_dir``."""
    return clean(load_json_files(data_dir or config.data_dir()), exclude_adjustments=exclude_adjustments)


def top_countries(df, n=10):
    """The ``n`` countries with the largest total revenue."""
    return df.groupby("country")["price"].sum().sort_values(ascending=False).index[:n].tolist()


def convert_to_ts(df, country=None, start=None, end=None):
    """Aggregate to one row per calendar day. Days with no trading get zeros (no gaps)."""
    if country is not None:
        if country not in set(df["country"]):
            raise IngestionError(f"country not found: {country}")
        start = start if start is not None else df["invoice_date"].min()
        end = end if end is not None else df["invoice_date"].max()
        df = df[df["country"] == country]
    else:
        start = start if start is not None else df["invoice_date"].min()
        end = end if end is not None else df["invoice_date"].max()
    grouped = df.groupby("invoice_date").agg(
        purchases=("price", "size"), unique_invoices=("invoice", "nunique"),
        unique_streams=("stream_id", "nunique"), total_views=("times_viewed", "sum"),
        revenue=("price", "sum"))
    days = pd.date_range(start, end, freq="D")
    ts = grouped.reindex(days).fillna(0.0)
    ts.index.name = "date"
    ts = ts.reset_index()
    ts["year_month"] = ts["date"].dt.strftime("%Y-%m")
    return ts[TS_COLUMNS + ["year_month"]]


def build_ts_dict(df, n_countries=10):
    start, end = df["invoice_date"].min(), df["invoice_date"].max()
    out = {"all": convert_to_ts(df, start=start, end=end)}
    for country in top_countries(df, n_countries):
        out[country_id(country)] = convert_to_ts(df, country, start=start, end=end)
    return out


def fetch_ts(data_dir=None, n_countries=10):
    """Daily time-series for ``all`` countries plus the top ``n_countries`` by revenue."""
    return build_ts_dict(fetch_data(data_dir), n_countries)


# In-memory cache so the API does not re-read ~120 MB of JSON on every request. The key includes
# file names, sizes and modification times, so adding a new month of data refreshes it automatically.
_TS_CACHE = {}


def _signature(data_dir):
    files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".json"))
    return tuple((f, os.stat(os.path.join(data_dir, f)).st_mtime_ns, os.stat(os.path.join(data_dir, f)).st_size)
                 for f in files)


def get_ts(data_dir=None, n_countries=10):
    data_dir = data_dir or config.data_dir()
    if not os.path.isdir(data_dir):
        raise IngestionError(f"data directory does not exist: {data_dir}")
    key = (os.path.abspath(data_dir), n_countries)
    sig = _signature(data_dir)
    cached = _TS_CACHE.get(key)
    if cached and cached[0] == sig:
        return cached[1]
    ts = fetch_ts(data_dir, n_countries)
    _TS_CACHE[key] = (sig, ts)
    return ts


def clear_cache():
    _TS_CACHE.clear()
