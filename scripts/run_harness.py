#!/usr/bin/env python3
# Runs the benchmark harness on a frozen snapshot. --dry-run lists the forecast origins per sensor (valid, and why the
# others are not) and writes nothing. Otherwise one forecaster runs over the tasks and writes raw forecasts, metrics
# and a run manifest to a new folder under --out, then the mean metrics per sensor, context and horizon are printed:
# an in-house check forecaster (naive / seasonal naive; harness checks, not thesis baselines) or a baseline from
# benchmark/models (AutoGluon / statsforecast; needs the bench venv), or a zero-shot foundation model (Chronos-2; bench
# venv plus its downloaded weights). A model runs on its own allowed context lengths unless --contexts picks some of
# them. No network; reads only the snapshot (set HF_HUB_OFFLINE=1 so model weights come only from the local cache).
#
# Usage (from sensbee_nvp/):
#   venv/bin/python scripts/run_harness.py --snapshot ../../benchmark_data/snapshots/sensbee-2026-10-01 --dry-run
#   venv/bin/python scripts/run_harness.py --snapshot <dir> --out <dir> --model seasonal_naive_96
#   venv/bin/python scripts/run_harness.py --snapshot <dir> --out <dir> --model seasonal_naive_672 --contexts 28
#   venv-bench/bin/python scripts/run_harness.py --snapshot <dir> --out <dir> --model ag_ets
#   HF_HUB_OFFLINE=1 venv-bench/bin/python scripts/run_harness.py --snapshot <dir> --out <dir> --model chronos2

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import METRICS  # noqa: E402
from benchmark.models import BASELINES, ZERO_SHOT, library_versions  # noqa: E402
from benchmark.runner import CHECK_FORECASTERS, run  # noqa: E402
from benchmark.snapshot import load_snapshot  # noqa: E402
from benchmark.tasks import CONTEXT_DAYS, TASKS, origin_table, target_series  # noqa: E402

INVALID_REASONS = ("excluded", "short_history", "target_gap", "recent_gap", "low_coverage")


# One row per sensor: candidate origins of the period, how many are valid, the count per reason for the others, the
# lowest 7-day context coverage among the valid ones, and how many valid origins have an exclusion inside their
# context.
def dry_run(snapshot_dir: Path, sensors: list[str], period: str) -> pd.DataFrame:
    frames = load_snapshot(snapshot_dir)
    rows = []
    for name in sensors:
        table = origin_table(target_series(frames[name], TASKS[name]), TASKS[name], period)
        valid = table[table["valid"]]
        rows.append({
            "sensor": name,
            "secondary": TASKS[name].secondary,
            "candidates": len(table),
            "valid": len(valid),
            **{reason: int((table["reason"] == reason).sum()) for reason in INVALID_REASONS},
            "min_coverage_7d": valid["coverage_7d"].min(),
            "exclusion_in_context": int(valid["exclusion_in_context"].sum()),
        })
    return pd.DataFrame(rows)


# Mean metrics per sensor, context and horizon over the origins that didn't fail, with the origin counts. Every cell is
# shown, also one where all origins failed. Contexts keep the order of the run.
def summarize(run_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    metrics["context"] = pd.Categorical(metrics["context"], categories=list(dict.fromkeys(metrics["context"])))
    keys = ["sensor", "context", "horizon_steps"]
    ok = metrics[metrics["status"] != "failed"]
    every_cell = metrics.groupby(keys, observed=True).size().index
    summary = ok.groupby(keys, observed=True)[list(METRICS)].mean().reindex(every_cell)
    summary["n_origins"] = ok.groupby(keys, observed=True).size()
    summary["failed"] = metrics[metrics["status"] == "failed"].groupby(keys, observed=True).size()
    return summary.fillna({"n_origins": 0, "failed": 0})


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark harness on a frozen snapshot (no network)")
    parser.add_argument("--snapshot", type=Path, required=True, help="snapshot folder (with manifest.json)")
    parser.add_argument("--out", type=Path, help="parent folder; a new <run id> folder is created (not for --dry-run)")
    parser.add_argument("--model", choices=[*CHECK_FORECASTERS, *BASELINES, *ZERO_SHOT],
                        help="check forecaster, baseline or zero-shot model")
    parser.add_argument("--dry-run", action="store_true", help="list the origins per sensor and write nothing")
    parser.add_argument("--sensors", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--period", choices=["test", "validation"], default="test")
    parser.add_argument("--contexts", nargs="+",
                        help="context lengths in days, or 'all' for the whole history before each origin "
                             "(default: all of CONTEXT_DAYS, or a model's allowed lengths)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dry_run:
        print(dry_run(args.snapshot, args.sensors, args.period).to_string(index=False))
        return
    if args.out is None or args.model is None:
        parser.error("--out and --model are required unless --dry-run")
    model_info = {}
    models = {**BASELINES, **ZERO_SHOT}
    if args.model in models:
        baseline = models[args.model]
        contexts = baseline.contexts if args.contexts is None else tuple(
            None if c == "all" else int(c) for c in args.contexts)
        if not set(contexts) <= set(baseline.contexts):
            parser.error(f"{args.model} runs only on contexts {list(baseline.contexts)} (days)")
        forecaster, params = baseline.build()
        model_info = library_versions()
    else:
        forecaster, params = CHECK_FORECASTERS[args.model]
        contexts = tuple(None if c == "all" else int(c) for c in (args.contexts or CONTEXT_DAYS))
    run_dir = run(forecaster, args.model, args.snapshot, args.out, tasks={name: TASKS[name] for name in args.sensors},
                  period=args.period, context_days=contexts, seed=args.seed, model_params=params,
                  model_info=model_info)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summarize(run_dir).round(4).to_string())
    print(f"RUN WRITTEN to {run_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
