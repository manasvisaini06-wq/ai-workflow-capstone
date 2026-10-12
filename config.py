"""Locations of data, models and logs.

Everything is resolved at call time from environment variables, so a test (or a
container) can redirect all reads and writes without touching production files.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def model_dir():
    return os.environ.get("AAVAIL_MODEL_DIR", os.path.join(BASE_DIR, "models"))


def log_dir():
    return os.environ.get("AAVAIL_LOG_DIR", os.path.join(BASE_DIR, "logs"))


def data_dir():
    return os.environ.get("AAVAIL_DATA_DIR", os.path.join(BASE_DIR, "data", "cs-train"))


def selection_path():
    """JSON file with the per-series model choice produced by model_comparison.py."""
    return os.environ.get("AAVAIL_SELECTION_FILE", os.path.join(BASE_DIR, "model_selection.json"))
