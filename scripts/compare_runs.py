#!/usr/bin/env python3
# Compares the forecasts of two harness runs of the same model, snapshot, period and contexts (e.g. the Mac and the
# cluster, gate E). Pass (owner decision 2026-10-26; a relative rule breaks on quantiles near 0):
# - the same (sensor, context, origin, step) rows and the same targets;
# - every quantile within --atol in the target's units (default 0.01);
# - every per-cell metric mean (sensor, context, horizon) equal at 4 decimals: |difference| <= 5e-5 (comparing
#   rounded values would fail on means that straddle a rounding boundary); NaN in both runs counts as equal, a cell
#   missing from one run as different.
# --contexts 28d compares only those contexts (e.g. a 28-day-only fine-tuned run with a 1/7/28-day zero-shot run).
# Prints the largest differences and exits 1 if a check fails.
#
# Usage (from sensbee_nvp/): venv-bench/bin/python scripts/compare_runs.py <run A> <run B> [--contexts 28d]

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
    parser.add_argument("--atol", type=float, default=0.01, help="absolute tolerance per quantile (target units)")
    parser.add_argument("--contexts", nargs="+", help="compare only these contexts, e.g. 28d (default: all)")
    args = parser.parse_args()
    records = [json.loads((run / "run.json").read_text()) for run in (args.run_a, args.run_b)]
    for key in ("snapshot", "period", "settings"):
        if records[0][key] != records[1][key]:
            sys.exit(f"RUNS DIFFER IN {key.upper()}")

    # The rows of the compared contexts (all without --contexts).
    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        return frame if args.contexts is None else frame[frame["context"].isin(args.contexts)]

    a, b = (keep(pd.read_csv(run / "forecasts.csv.gz")).set_index(KEYS).sort_index()
            for run in (args.run_a, args.run_b))
    if not len(a):
        sys.exit("NO ROWS TO COMPARE")
    if not a.index.equals(b.index):
        sys.exit(f"RUNS COVER DIFFERENT ROWS: {len(a)} vs {len(b)}")
    if not np.allclose(a["y"], b["y"], equal_nan=True):
        sys.exit("TARGETS DIFFER")
    quantiles = [c for c in a.columns if c.startswith("q")]
    qa, qb = a[quantiles].to_numpy(float), b[quantiles].to_numpy(float)
    largest = float(np.nanmax(np.abs(qa - qb)))
    columns = ["MAE", "RMSE", "MASE", "QL", "SQL", "COV80"]
    means = [keep(pd.read_csv(run / "metrics.csv")).query("status != 'failed'").groupby(
        ["sensor", "context", "horizon_steps"])[columns].mean() for run in (args.run_a, args.run_b)]
    index = means[0].index.union(means[1].index)
    a_means, b_means = means[0].reindex(index), means[1].reindex(index)
    differing = (a_means - b_means).abs().gt(5e-5) | (a_means.isna() != b_means.isna())
    cells_differing = int(differing.any(axis=1).sum())
    print(f"{len(a)} rows; max abs quantile diff {largest:.3g} (atol {args.atol}); "
          f"metric cells differing at 4 decimals: {cells_differing} of {len(index)}")
    print(f"code A {records[0]['code']}, code B {records[1]['code']}")
    print(f"platform A {records[0].get('platform')}, platform B {records[1].get('platform')}")
    sys.exit(0 if largest <= args.atol and cells_differing == 0 else 1)


if __name__ == "__main__":
    main()
