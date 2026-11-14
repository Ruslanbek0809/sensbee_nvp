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
# --task-set names the runs' tasks (benchmark/tasks.py; default october): the sensor groups and the open-slot halls.
# "january" is built from --snapshot (the January snapshot), so it needs it. It must be the runs' own task set
# (run.json "task_set"; runs without one used october).
# --as NAME RUN... reports several runs (disjoint sensors) as one model NAME, e.g. a tuned model whose config was
# picked per sensor; those runs needn't be listed under --runs again. Runs of one config with different seeds merged
# this way are averaged per origin over the seeds.
# --picks NAME PICKS_CSV RUN... reports the runs as one model NAME, each keeping only the sensors whose validation pick
# (scripts/select_on_validation.py output) is its config: a pooled run's other sensors enter no table. Seeds of one
# config are averaged as with --as.
#
# Usage (from sensbee_nvp/):
#   venv-bench/bin/python scripts/make_report.py --runs ../../benchmark_data/results/<run>... \
#       --out ../../benchmark_data/reports
#   venv-bench/bin/python scripts/make_report.py --runs <subset run> <other runs>... --out <dir> --common-origins
#   venv-bench/bin/python scripts/make_report.py --runs <runs>... --out <dir> \
#       --snapshot ../../benchmark_data/snapshots/sensbee-2026-10-01 --test F5 mask0_chronos2_recent_t0.3@28d chronos2@28d
#   venv-bench/bin/python scripts/make_report.py --runs <runs>... --out <dir> \
#       --picks chronos2_ft_tuned <picks_chronos2_ft.csv> <seed runs of every picked config>... \
#       --test F7 chronos2_ft_tuned@28d chronos2@28d

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.models import LLMTIME, REFERENCE, ZERO_SHOT  # noqa: E402
from benchmark.report import FamilyTest, default_families, load_runs, make_report, picked_runs  # noqa: E402
from benchmark.tasks import TASK_SETS, task_set  # noqa: E402


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
    parser.add_argument("--task-set", choices=TASK_SETS, default="october",
                        help="the runs' task set (january needs --snapshot)")
    parser.add_argument("--as", dest="aliases", nargs="+", action="append", default=[], metavar="NAME RUN",
                        help="report the following run folders as one model NAME (repeatable)")
    parser.add_argument("--picks", nargs="+", action="append", default=[], metavar="NAME PICKS_CSV RUN",
                        help="report the run folders as one model NAME, each keeping only the sensors that picked its "
                             "config in PICKS_CSV (repeatable)")
    parser.add_argument("--test", nargs=3, action="append", default=[], metavar=("FAMILY", "MODEL@CTX", "AGAINST@CTX"),
                        help="an extra family test chosen on validation (repeatable)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.task_set != "october" and args.snapshot is None:
        parser.error(f"--task-set {args.task_set} is built from the snapshot: give --snapshot")
    tasks, _ = task_set(args.task_set, args.snapshot)
    if any(len(group) < 2 for group in args.aliases):
        parser.error("--as needs a NAME and at least one run folder")
    if any(len(group) < 3 for group in args.picks):
        parser.error("--picks needs a NAME, a picks file and at least one run folder")
    aliases = {str(run): group[0] for group in args.aliases for run in group[1:]}
    runs = [*args.runs, *[Path(run) for group in args.aliases for run in group[1:]]]
    sensors, picks = {}, {}
    for name, picks_csv, *picked in args.picks:
        picked_aliases, picked_sensors = picked_runs(Path(picks_csv), name, [Path(run) for run in picked])
        aliases.update(picked_aliases)
        sensors.update(picked_sensors)
        picks[name] = Path(picks_csv)
        runs += [Path(run) for run in picked]
    records, _ = load_runs(runs, aliases, sensors)
    run_set = next(iter(records.values()))["task_set"]  # load_runs refuses runs of different task sets
    if run_set != args.task_set:
        parser.error(f"the runs are of task set {run_set}, not {args.task_set}: give --task-set {run_set}")
    families = default_families(set(records), REFERENCE, {m: b.contexts for m, b in ZERO_SHOT.items()},
                                {m: b.contexts for m, b in LLMTIME.items()})
    families += [FamilyTest(family, *_model_context(model), *_model_context(against))
                 for family, model, against in args.test]
    report_dir = make_report(runs, args.out, REFERENCE, tasks=tasks, common=args.common_origins, families=families,
                             snapshot_dir=args.snapshot, aliases=aliases, sensors=sensors, picks=picks)
    print((report_dir / "summary.md").read_text())
    print(f"REPORT WRITTEN to {report_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
