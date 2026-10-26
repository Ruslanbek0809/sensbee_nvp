#!/usr/bin/env python3
# Builds the baseline report (benchmark/report.py) from harness run folders.
# It writes these tables, plus summary.md:
# - per-cell means, relative errors and skill against the reference;
# - Diebold-Mariano tests;
# - failures and negative-quantile shares;
# - the clipped visitor sensitivity table.
# It reads only the run folders and writes a new folder under --out.
# It needs the bench venv (statsmodels for the DM tests).
# --common-origins builds every table on the (sensor, origin) pairs that all runs evaluated.
# Example: a model run on an origin subset, compared with the others on exactly those origins.
# Test families (Phase 6a, frozen protocol F1-F7): the defaults that need no validation choice are always added
# (benchmark.report.default_families, 28-day context); --test adds one chosen on validation, e.g. F5 or F7.
# --snapshot gives the halls' open-slot coverage (COV80_open).
#
# Usage (from sensbee_nvp/):
#   venv-bench/bin/python scripts/make_report.py --runs ../../benchmark_data/results/<run>... \
#       --out ../../benchmark_data/reports
#   venv-bench/bin/python scripts/make_report.py --runs <subset run> <other runs>... --out <dir> --common-origins
#   venv-bench/bin/python scripts/make_report.py --runs <runs>... --out <dir> \
#       --snapshot ../../benchmark_data/snapshots/sensbee-2026-10-01 --test F5 mask0_chronos2_recent_t0.3@28d chronos2@28d

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.models import LLMTIME, REFERENCE, ZERO_SHOT  # noqa: E402
from benchmark.report import FamilyTest, default_families, load_runs, make_report  # noqa: E402


# "model@28d" → ("model", "28d").
def _model_context(text: str) -> tuple[str, str]:
    model, _, context = text.rpartition("@")
    if not model or not context.endswith("d"):
        raise argparse.ArgumentTypeError(f"EXPECTED MODEL@<days>d, GOT {text}")
    return model, context


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline report from harness runs (no network)")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="run folders (one per model)")
    parser.add_argument("--out", type=Path, required=True, help="parent folder; a new <UTC>_report folder is created")
    parser.add_argument("--common-origins", action="store_true",
                        help="only the (sensor, origin) pairs that every run evaluated")
    parser.add_argument("--snapshot", type=Path, help="the runs' snapshot, for the halls' open-slot coverage")
    parser.add_argument("--test", nargs=3, action="append", default=[], metavar=("FAMILY", "MODEL@CTX", "AGAINST@CTX"),
                        help="an extra family test chosen on validation (repeatable)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    records, _ = load_runs(args.runs)
    families = default_families(set(records), REFERENCE, {m: b.contexts for m, b in ZERO_SHOT.items()},
                                {m: b.contexts for m, b in LLMTIME.items()})
    families += [FamilyTest(family, *_model_context(model), *_model_context(against))
                 for family, model, against in args.test]
    report_dir = make_report(args.runs, args.out, REFERENCE, common=args.common_origins, families=families,
                             snapshot_dir=args.snapshot)
    print((report_dir / "summary.md").read_text())
    print(f"REPORT WRITTEN to {report_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
