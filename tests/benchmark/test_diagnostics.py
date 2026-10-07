# Tests for benchmark/diagnostics.py on hand-computed examples: the amplitude ratio and shape correlation of a forecast
# median against the reference's median, the error by origin hour, and the check that the reference repeats the last
# observed day.

import math

import numpy as np
import pandas as pd
import pytest

from benchmark.diagnostics import hour_table, reference_last_day_difference, shape_table

ORIGIN = pd.Timestamp("2026-02-02 00:00")


# Forecast rows of one origin in the forecasts.csv.gz layout (only the columns the diagnostics read).
def _rows(model: str, medians: list[float], origin: pd.Timestamp = ORIGIN, context: str = "1d",
          status: str = "ok") -> pd.DataFrame:
    return pd.DataFrame({"model": model, "sensor": "W", "context": context, "origin": origin,
                         "step": np.arange(1, len(medians) + 1), "status": status, "q0.5": medians})


def test_amplitude_and_shape_against_the_reference():
    reference = _rows("ref", [0, 2, 0, 2]).drop(columns="model")  # std 1; real reference rows have no model column
    forecasts = pd.concat([_rows("flat", [1, 1, 1, 1]), _rows("same", [1, 3, 1, 3]), _rows("half", [0, 1, 0, 1]),
                           _rows("flipped", [2, 0, 2, 0]), _rows("broken", [9, 9, 9, 9], status="failed")])

    table = shape_table(forecasts, reference).set_index("model")

    assert "broken" not in table.index  # failed origins are dropped
    assert table.loc["flat", "amplitude_ratio"] == 0 and math.isnan(table.loc["flat", "shape_correlation"])
    assert table.loc["same", "amplitude_ratio"] == pytest.approx(1) and table.loc["same", "shape_correlation"] == 1
    assert table.loc["half", "amplitude_ratio"] == pytest.approx(0.5)
    assert table.loc["half", "shape_correlation"] == pytest.approx(1)
    assert table.loc["flipped", "shape_correlation"] == pytest.approx(-1)


def test_correlation_and_amplitude_equal_numpy_on_irregular_shapes():
    rng = np.random.default_rng(0)
    model, ref = rng.normal(size=96), rng.normal(size=96)

    row = shape_table(_rows("m", list(model)), _rows("ref", list(ref)).drop(columns="model")).iloc[0]

    assert row["shape_correlation"] == pytest.approx(np.corrcoef(model, ref)[0, 1])
    assert row["amplitude_ratio"] == pytest.approx(model.std() / ref.std())


def test_constant_reference_gives_nan():
    table = shape_table(_rows("m", [1, 3, 1, 3]), _rows("ref", [5, 5, 5, 5]).drop(columns="model"))

    assert math.isnan(table.loc[0, "amplitude_ratio"]) and math.isnan(table.loc[0, "shape_correlation"])


def test_error_by_origin_hour():
    six = ORIGIN + pd.Timedelta("6h")
    metrics = pd.DataFrame({"model": "m", "sensor": "W", "context": "1d", "origin": [ORIGIN, six, six],
                            "horizon_steps": [96, 96, 4], "MASE": [1.0, 3.0, np.nan]})
    reference = pd.DataFrame({"sensor": "W", "origin": [ORIGIN, six, six], "horizon_steps": [96, 96, 4],
                              "MASE": [2.0, 2.0, 1.0]})

    table = hour_table(metrics, reference).set_index(["horizon_steps", "hour"])

    assert table.loc[(96, 0), "relative_error"] == 0.5 and table.loc[(96, 6), "relative_error"] == 1.5
    assert (4, 6) not in table.index  # NaN errors are not paired
    assert table.loc[(96, 6), "n_origins"] == 1


def test_reference_repeats_the_last_day():
    index = pd.date_range("2026-02-01 00:00", periods=96, freq="15min")
    series = {"W": pd.Series(np.arange(96, dtype=float), index=index)}
    exact = _rows("ref", list(np.arange(96, dtype=float)))
    shifted = _rows("ref", list(np.arange(96, dtype=float) + 0.5))

    assert reference_last_day_difference(exact, series) == {"W": 0.0}
    assert reference_last_day_difference(shifted, series) == {"W": 0.5}
