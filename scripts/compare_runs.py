#!/usr/bin/env python3
# Compares the forecasts of two harness runs of the same model, snapshot, period and contexts (e.g. the Mac and the
# cluster, gate E): the same (sensor, context, origin, step) rows, the same targets, and every quantile within a
# relative tolerance. Prints the largest differences and exits 1 if the runs differ beyond the tolerance.
#
# Usage (from sensbee_nvp/): venv-bench/bin/python scripts/compare_runs.py <run A> <run B> --rtol 1e-3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["sensor", "context", "origin", "step"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two runs' forecasts")
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-3, help="relative tolerance per quantile")
    args = parser.parse_args()
    records = [json.loads((run / "run.json").read_text()) for run in (args.run_a, args.run_b)]
    for key in ("snapshot", "period", "settings"):
        if records[0][key] != records[1][key]:
            sys.exit(f"RUNS DIFFER IN {key.upper()}")
    a, b = (pd.read_csv(run / "forecasts.csv.gz").set_index(KEYS).sort_index() for run in (args.run_a, args.run_b))
    if not a.index.equals(b.index):
        sys.exit(f"RUNS COVER DIFFERENT ROWS: {len(a)} vs {len(b)}")
    if not np.allclose(a["y"], b["y"], equal_nan=True):
        sys.exit("TARGETS DIFFER")
    quantiles = [c for c in a.columns if c.startswith("q")]
    qa, qb = a[quantiles].to_numpy(float), b[quantiles].to_numpy(float)
    relative = np.abs(qa - qb) / np.maximum(np.abs(qa), 1e-6)
    print(f"{len(a)} rows; max abs diff {np.nanmax(np.abs(qa - qb)):.3g}; max relative diff {np.nanmax(relative):.3g}; "
          f"share of quantiles beyond rtol {args.rtol}: {np.mean(relative > args.rtol):.3g}")
    print(f"code A {records[0]['code']}, code B {records[1]['code']}")
    print(f"platform A {records[0].get('platform')}, platform B {records[1].get('platform')}")
    sys.exit(0 if np.all(np.isclose(qa, qb, rtol=args.rtol, atol=1e-6)) else 1)


if __name__ == "__main__":
    main()
