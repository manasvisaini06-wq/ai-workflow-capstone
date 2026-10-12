#!/usr/bin/env python
"""Run every unit test (API, model/ingestion/features, logging) with a single command.

    python run-tests.py

Exit status is 0 only if all tests pass. Tests use synthetic data and temporary
model/log directories, so they never touch production models or logs.
"""
import os
import sys

import pytest

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", os.path.join(here, "tests")]))
