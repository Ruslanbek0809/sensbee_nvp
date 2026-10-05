#!/usr/bin/env python3
# Builds the baseline report (benchmark/report.py) from harness run folders: per-cell means, relative errors and skill
# against the reference, Diebold–Mariano tests, failures, negative-quantile shares and the clipped visitor sensitivity
# table, plus summary.md. Reads only the run folders; writes a new folder under --out. Needs the bench venv
# (statsmodels for the DM tests).
#
# Usage (from sensbee_nvp/):
#   venv-bench/bin/python scripts/make_report.py --runs ../../benchmark_data/results/<run>... \
#       --out ../../benchmark_data/reports

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.models import REFERENCE  # noqa: E402
from benchmark.report import make_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline report from harness runs (no network)")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="run folders (one per model)")
    parser.add_argument("--out", type=Path, required=True, help="parent folder; a new <UTC>_report folder is created")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report_dir = make_report(args.runs, args.out, REFERENCE)
    print((report_dir / "summary.md").read_text())
    print(f"REPORT WRITTEN to {report_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
