#!/usr/bin/env python3
# Fits the band-widening factor of each service recipe (benchmark/bands.py, thesis Phase 5 step 5.4) on stored
# validation forecasts. No model runs.
# - Each recipe's run is read at the 28-day context, all 96 steps, every origin pooled.
# - The factor is the one on bands.FIT_GRID whose COV80 is closest to 0.80, leaving out [0, 0] bands.
# It writes a new folder <UTC>_phase5_5.4_bands under --out:
# - band_factors.csv: per recipe, the factor and COV80 before and after, on the fitted rows and on all rows;
# - coverage_detail.csv: the same per sensor and at the horizons 4, 24 and 96 steps (descriptive);
# - run.json: the input runs with the SHA-256 of their forecasts, the grid and the code commit.
#
# Usage (from sensbee_nvp/):
#   venv/bin/python scripts/fit_band_factors.py --plain <run> --stations <run> --mask <run> \
#       --out ../../benchmark_data/spikes

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.bands import BAND_FACTORS, FIT_GRID, fit_factor, nonzero_bands, widen  # noqa: E402
from benchmark.metrics import coverage80  # noqa: E402
from benchmark.report import QUANTILE_COLUMNS as COLUMNS  # noqa: E402
from benchmark.snapshot import git_state, sha256_file  # noqa: E402
from benchmark.tasks import QUANTILE_LEVELS, REPORT_STEPS  # noqa: E402

CONTEXT = "28d"


# The 28-day rows of a run's forecasts, without failed origins (as the report leaves them out).
def _load(run: Path) -> pd.DataFrame:
    frame = pd.read_csv(run / "forecasts.csv.gz", float_precision="round_trip")  # exact floats, as written
    frame = frame[(frame["context"] == CONTEXT) & (frame["status"] != "failed")]
    if frame.empty:
        raise ValueError(f"NO {CONTEXT} FORECASTS in {run}")
    return frame


# COV80 of `frame` before and after widening by `factor`, on all rows and on the rows whose band isn't [0, 0].
def _coverages(frame: pd.DataFrame, factor: float) -> dict:
    y, raw = frame["y"].to_numpy(), frame[COLUMNS].to_numpy()
    wide = widen(raw, QUANTILE_LEVELS, factor)
    fitted = nonzero_bands(raw)
    return {"rows": len(frame), "fitted_rows": int(fitted.sum()),
            "cov80_before": coverage80(y[fitted], raw[fitted], QUANTILE_LEVELS),
            "cov80_after": coverage80(y[fitted], wide[fitted], QUANTILE_LEVELS),
            "cov80_all_before": coverage80(y, raw, QUANTILE_LEVELS),
            "cov80_all_after": coverage80(y, wide, QUANTILE_LEVELS)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the band-widening factors on validation runs (no network)")
    for recipe in BAND_FACTORS:
        parser.add_argument(f"--{recipe}", type=Path, required=True, help=f"validation run folder for '{recipe}'")
    parser.add_argument("--out", type=Path, required=True, help="parent folder; a new <UTC> folder is created")
    args = parser.parse_args()

    runs = {recipe: getattr(args, recipe) for recipe in BAND_FACTORS}
    summary, detail, inputs = [], [], {}
    for recipe, run in runs.items():
        meta = json.loads((run / "run.json").read_text())
        if meta.get("period") != "validation":
            raise ValueError(f"{run} is not a validation run (period {meta.get('period')})")
        frame = _load(run)
        factor, _ = fit_factor(frame["y"].to_numpy(), frame[COLUMNS].to_numpy(), QUANTILE_LEVELS)
        summary.append({"recipe": recipe, "factor": factor, "at_grid_edge": factor in (FIT_GRID[0], FIT_GRID[-1]),
                        **_coverages(frame, factor)})
        for sensor, rows in frame.groupby("sensor"):
            for h in REPORT_STEPS:
                detail.append({"recipe": recipe, "sensor": sensor, "horizon_steps": h,
                               **_coverages(rows[rows["step"] <= h], factor)})
        inputs[recipe] = {"run": run.name, "model": meta.get("model"),
                          "forecasts_sha256": sha256_file(run / "forecasts.csv.gz")}

    out = args.out / f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}_phase5_5.4_bands"
    out.mkdir(parents=True)
    pd.DataFrame(summary).to_csv(out / "band_factors.csv", index=False)
    pd.DataFrame(detail).to_csv(out / "coverage_detail.csv", index=False)
    grid = [float(FIT_GRID[0]), float(FIT_GRID[-1]), round(float(FIT_GRID[1] - FIT_GRID[0]), 2)]  # first, last, step
    (out / "run.json").write_text(json.dumps({"inputs": inputs, "context": CONTEXT, "target": 0.80, "grid": grid,
                                              "code": git_state()}, indent=2))
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(summary).round(4).to_string(index=False))
    print(f"WRITTEN to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
