#!/usr/bin/env python3
# Context-length diagnostics (benchmark/diagnostics.py) from harness run folders: per origin, the amplitude ratio and
# shape correlation of each model's median against the reference's median (does the forecast keep the daily cycle?),
# the relative error by origin hour, a check that the reference repeats the last observed day (on the snapshot), and
# figures of a few origins per model and sensor. --runs must include the reference run (Seasonal Naive m = 96, whose
# 28-day context is the reference, D-B6). Reads only the run folders and the snapshot; writes a new folder under --out.
#
# Usage (from sensbee_nvp/; matplotlib comes with the bench venv):
#   venv-bench/bin/python scripts/context_diagnostics.py --runs ../../benchmark_data/results/<run>... \
#       --snapshot ../../benchmark_data/snapshots/sensbee-2026-10-01 --out ../../benchmark_data/reports

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.diagnostics import hour_table, reference_last_day_difference, shape_table  # noqa: E402
from benchmark.models import REFERENCE  # noqa: E402
from benchmark.report import _context_order, load_runs  # noqa: E402
from benchmark.snapshot import git_state, load_snapshot  # noqa: E402
from benchmark.tasks import TASKS, target_series  # noqa: E402

FORECAST_COLUMNS = ["sensor", "context", "origin", "step", "status", "y", "q0.1", "q0.5", "q0.9"]


# Loads the runs (report.load_runs: same snapshot, period and settings, one run per model) and splits off the
# reference at its context. Returns (records by model, model forecasts, model metrics, reference forecasts, reference
# metrics); model rows carry a "model" column.
def _load(run_dirs: list[Path]) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ref_model, ref_context = REFERENCE[0], f"{REFERENCE[1]}d"
    records, metrics = load_runs(run_dirs)
    if ref_model not in records:
        raise ValueError(f"--runs must include the reference run ({ref_model})")
    forecasts, reference = [], None
    for model, record in records.items():
        rows = pd.read_csv(Path(record["dir"]) / "forecasts.csv.gz", usecols=FORECAST_COLUMNS)
        if model == ref_model:
            reference = rows[rows["context"] == ref_context]
        else:
            forecasts.append(rows.assign(model=model))
    is_ref = metrics["model"] == ref_model
    reference_metrics = metrics[is_ref & (metrics["context"] == ref_context)]
    return records, pd.concat(forecasts, ignore_index=True), metrics[~is_ref], reference, reference_metrics


# Per model, sensor and context: the number of origins, how many have a defined shape (a constant reference, e.g. after
# a closed day, has none), and the median amplitude ratio and shape correlation; contexts sorted by length.
def _shape_summary(shape: pd.DataFrame) -> pd.DataFrame:
    summary = shape.groupby(["model", "sensor", "context"]).agg(
        n_origins=("origin", "size"), n_defined=("shape_correlation", "count"),
        median_amplitude_ratio=("amplitude_ratio", "median"),
        median_shape_correlation=("shape_correlation", "median")).reset_index()
    return summary.sort_values(["model", "sensor", "context"], key=_context_order, ignore_index=True)


# A figure per model and sensor from that sensor's rows: rows = 3 origins (at a quarter, half and three quarters of the
# reference's origins), columns = the given contexts; truth, the model's median with its 10–90% band, and the
# reference's median.
def _figure(path: Path, model: str, sensor: str, rows: pd.DataFrame, reference: pd.DataFrame,
            contexts: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    origins = sorted(reference["origin"].unique())
    picks = [origins[int(len(origins) * share)] for share in (0.25, 0.5, 0.75)]
    fig, axes = plt.subplots(len(picks), len(contexts), figsize=(4 * len(contexts), 2.6 * len(picks)), squeeze=False)
    for i, origin in enumerate(picks):
        ref = reference[reference["origin"] == origin].sort_values("step")
        own_origin = rows[rows["origin"] == origin]
        for j, context in enumerate(contexts):
            ax = axes[i][j]
            own = own_origin[own_origin["context"] == context].sort_values("step")
            ax.plot(ref["step"], ref["y"], color="black", lw=1, label="truth")
            ax.plot(ref["step"], ref["q0.5"], color="grey", lw=1, ls="--", label="reference median")
            if len(own):
                ax.fill_between(own["step"], own["q0.1"], own["q0.9"], alpha=0.25, label="model 10–90%")
                ax.plot(own["step"], own["q0.5"], lw=1.2, label="model median")
            ax.set_title(f"{context}, origin {str(origin)[:16]} UTC", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{model} on {sensor}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Context-length diagnostics from harness runs (no network)")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="run folders, including the reference")
    parser.add_argument("--snapshot", type=Path, required=True, help="the snapshot the runs used")
    parser.add_argument("--out", type=Path, required=True, help="parent folder; a new <UTC>_context_diagnostics folder")
    parser.add_argument("--hour-contexts", nargs="+", default=["1d", "2d", "7d"],
                        help="contexts of the by-hour table and the figures")
    parser.add_argument("--figure-sensors", nargs="+", default=["MANEBACH_WEATHER_STATION", "EISHALLE"])
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    out_dir = args.out / f"{started:%Y-%m-%dT%H%M%SZ}_context_diagnostics"
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} EXISTS; outputs are never overwritten")
    records, forecasts, metrics, reference, reference_metrics = _load(args.runs)
    shape = shape_table(forecasts, reference)
    summary = _shape_summary(shape)
    by_hour = hour_table(metrics[metrics["context"].isin(args.hour_contexts)], reference_metrics)
    frames = load_snapshot(args.snapshot)
    series = {name: target_series(frames[name], TASKS[name]) for name in reference["sensor"].unique()}
    last_day = reference_last_day_difference(reference, series)

    out_dir.mkdir(parents=True)
    shape.to_csv(out_dir / "shape.csv", index=False)
    summary.to_csv(out_dir / "shape_summary.csv", index=False)
    by_hour.to_csv(out_dir / "by_hour.csv", index=False)
    for model, rows in forecasts.groupby("model"):
        contexts = [c for c in args.hour_contexts if c in set(rows["context"])]
        for sensor in args.figure_sensors:
            sensor_rows = rows[rows["sensor"] == sensor]
            if contexts and len(sensor_rows):
                _figure(out_dir / f"{model}_{sensor}.png", model, sensor, sensor_rows,
                        reference[reference["sensor"] == sensor], contexts)
    (out_dir / "diagnostics.json").write_text(json.dumps({
        "runs": {model: record["run_id"] for model, record in records.items()}, "reference": list(REFERENCE),
        "reference_last_day_max_difference": last_day, "snapshot": str(args.snapshot),
        "hour_contexts": args.hour_contexts, "code": git_state(), "created_utc": started.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summary.round(3).to_string(index=False))
    print(f"REFERENCE LAST-DAY MAX DIFFERENCE: {last_day}")
    print(f"DIAGNOSTICS WRITTEN to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
