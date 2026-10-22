# Tests for benchmark/bands.py with hand-checked numbers:
# - widen() keeps the median and the order, sorts crossings first, and c = 1 changes nothing;
# - coverage80() counts q0.1 <= y <= q0.9;
# - fit_factor() finds a known factor and ignores [0, 0] bands;
# - widened_forecaster() widens a forecaster's output and passes covariates through.

import numpy as np
import pytest

from benchmark.bands import BAND_FACTORS, FIT_GRID, coverage80, fit_factor, widen, widened_forecaster
from benchmark.tasks import QUANTILE_LEVELS

DECILES = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])  # median 5


def test_widen_moves_each_decile_away_from_the_median():
    out = widen(DECILES, QUANTILE_LEVELS, 1.5)
    assert np.allclose(out, [[-1.0, 0.5, 2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 11.0]])  # 5 + 1.5 * (q - 5)


def test_widen_by_one_only_sorts():
    crossing = np.array([[3.0, 1.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0, 9.0]])
    assert np.array_equal(widen(crossing, QUANTILE_LEVELS, 1.0), np.sort(crossing, axis=-1))


def test_widen_sorts_before_taking_the_median():
    crossing = np.array([[9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])  # unsorted median column holds 4.0
    out = widen(crossing, QUANTILE_LEVELS, 2.0)
    assert out[0, 4] == 5.0 and np.allclose(out[0], 5.0 + 2.0 * (np.arange(1.0, 10.0) - 5.0))


def test_widen_refuses_to_narrow():
    with pytest.raises(ValueError, match="at least 1"):
        widen(DECILES, QUANTILE_LEVELS, 0.9)


def test_coverage80_counts_the_ends_as_inside():
    quantiles = np.tile(DECILES, (4, 1))  # band [1, 9]
    assert coverage80(np.array([1.0, 9.0, 0.5, 9.5]), quantiles, QUANTILE_LEVELS) == 0.5


# Truths at |y - 5| = 4.4, 4.8, 5.2, 5.6, 6.0 (and 2 inside the raw band [1, 9]): 80 % need |y - 5| <= 5.2,
# so c * 4 >= 5.2, c >= 1.30. The two [0, 0] rows (y = 0) are left out of the fit.
def test_fit_factor_finds_the_smallest_factor_that_reaches_80_percent():
    distance = np.array([4.4, 4.8, 5.2, 5.6, 6.0, 1.0, 2.0, 3.0, 0.0, 0.5])
    y = np.concatenate([5.0 + distance, [0.0, 0.0]])
    quantiles = np.concatenate([np.tile(DECILES, (10, 1)), np.zeros((2, 9))])
    factor, covered = fit_factor(y, quantiles, QUANTILE_LEVELS)
    assert factor == 1.30 and covered == 0.8


def test_fit_factor_stays_on_its_grid():
    y = np.array([1000.0])  # no factor up to 3 covers it
    factor, covered = fit_factor(y, DECILES, QUANTILE_LEVELS)
    assert factor == FIT_GRID[0] and covered == 0.0  # every factor misses: the first (smallest) wins the tie


def test_the_shipped_factors_widen():
    assert set(BAND_FACTORS) == {"plain", "stations", "mask"} and all(1 < c <= 3 for c in BAND_FACTORS.values())


def test_widened_forecaster_widens_and_passes_covariates_through():
    calls = []

    def forecaster(contexts, horizon, levels, seed, covariates=None):
        calls.append(covariates)
        return np.tile(DECILES, (len(contexts), horizon, 1)), ["ok"] * len(contexts)

    quantiles, status = widened_forecaster(forecaster, 2.0)([None, None], 3, QUANTILE_LEVELS, 0, covariates=["c"])
    assert calls == [["c"]] and status == ["ok", "ok"] and quantiles.shape == (2, 3, 9)
    assert np.allclose(quantiles[1, 2], 5.0 + 2.0 * (DECILES[0] - 5.0))
