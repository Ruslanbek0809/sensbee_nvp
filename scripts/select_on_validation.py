#!/usr/bin/env python3
# Selection on validation (Phase 4 rule E, used again in Phase 6a): per sensor and model family, the candidate run
# with the lowest mean SQL at 24 h over the sensor's validation origins at the 28-day context
# (benchmark.report.validation_selection). Reads only the run folders and prints selection.csv.
#
# Usage (from sensbee_nvp/):
#   venv-bench/bin/python scripts/select_on_validation.py --runs ../../benchmark_data/results/<validation run>... \
#       > <spike folder>/selection.csv

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.report import validation_selection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick each model family's best candidate per sensor on validation")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="validation run folders")
    parser.add_argument("--context", default="28d", help="context the choice is made at (default 28d)")
    args = parser.parse_args()
    validation_selection(args.runs, args.context).to_csv(sys.stdout, index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
