# Widened prediction bands for the service (thesis Phase 5 step 5.4, decision D-E4).
# Chronos-2's 80 % band (q0.1 to q0.9) holds the truth only about 70-74 % of the time on validation. widen() moves
# every quantile away from the median by a factor c: q'_k = q_0.5 + c * (q_k - q_0.5).
# - The median (the point forecast) doesn't change, the order of the quantiles is kept, and c = 1 changes nothing.
# - The quantiles are sorted first, as the runner sorts them for every model, so the service (which sorts, then
#   widens) and a harness run with --band-factor give the same numbers. A widened run therefore counts no crossings.
# BAND_FACTORS holds one c per service recipe, fitted on the validation runs by scripts/fit_band_factors.py.
# The thesis tables keep the raw bands; the widened ones are their own harness variant (run name suffix _w<c>).

from typing import Callable

import numpy as np

# Fitted on validation (28-day context, all 96 steps, bands other than [0, 0]); fit output in the thesis root:
# benchmark_data/spikes/2026-10-22T215830Z_phase5_5.4_bands/band_factors.csv. Validation COV80 goes from
# 0.706 / 0.720 / 0.739 to about 0.80.
BAND_FACTORS = {"plain": 1.20, "stations": 1.19, "mask": 1.12}
FIT_GRID = np.round(np.arange(100, 301) / 100, 2)  # 1.00, 1.01, ..., 3.00 (D-E4 widens only)


# Sorts quantiles [..., levels] along the last axis and widens them around the median by `factor` (>= 1).
def widen(quantiles: np.ndarray, levels: tuple[float, ...], factor: float) -> np.ndarray:
    if factor < 1:
        raise ValueError(f"BAND FACTOR must be at least 1, got {factor}")
    ordered = np.sort(np.asarray(quantiles, dtype=float), axis=-1)
    median = ordered[..., list(levels).index(0.5)][..., None]
    return median + factor * (ordered - median)


# The share of rows with q_0.1 <= y <= q_0.9 (COV80, as in benchmark/metrics.py) for quantiles [rows, levels].
def coverage80(y: np.ndarray, quantiles: np.ndarray, levels: tuple[float, ...]) -> float:
    low, high = quantiles[:, list(levels).index(0.1)], quantiles[:, list(levels).index(0.9)]
    return float(np.mean((low <= y) & (y <= high)))


# The factor on FIT_GRID whose widened COV80 is closest to 0.80 (ties: the smaller factor), with that coverage.
# Rows whose band is [0, 0] (every quantile 0, e.g. a masked closed slot) are left out: no factor changes them.
def fit_factor(y: np.ndarray, quantiles: np.ndarray, levels: tuple[float, ...]) -> tuple[float, float]:
    keep = ~(quantiles == 0).all(axis=1)
    y, quantiles = y[keep], quantiles[keep]
    covered = [coverage80(y, widen(quantiles, levels, c), levels) for c in FIT_GRID]
    best = int(np.argmin(np.abs(np.array(covered) - 0.80)))  # argmin takes the first, i.e. the smaller factor
    return float(FIT_GRID[best]), covered[best]


# Wraps a harness forecaster so that its quantiles come back widened by `factor`.
def widened_forecaster(forecaster: Callable, factor: float) -> Callable:
    def forecast(contexts, horizon, levels, seed, **kwargs):
        quantiles, status = forecaster(contexts, horizon, levels, seed, **kwargs)
        return widen(quantiles, levels, factor), status
    return forecast
