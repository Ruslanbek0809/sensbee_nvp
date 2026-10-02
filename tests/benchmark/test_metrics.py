# Tests for benchmark/metrics.py on examples computed by hand: point and quantile metrics on a 4-step window, the
# seasonal scale, crossed quantiles, and the relative skill against a reference (pairing, clipping, geometric mean).
# The last test cross-checks against AutoGluon's metric classes and only runs where autogluon.timeseries is installed.

import math

import numpy as np
import pandas as pd
import pytest

from benchmark.metrics import quantile_loss, relative_errors, relative_skill, seasonal_scale, window_metrics

LEVELS = (0.1, 0.5, 0.9)
DECILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
Y = np.array([1.0, 2.0, 3.0, 4.0])
# Quantiles per step at levels 0.1 / 0.5 / 0.9; the median forecast is [1, 1, 4, 2].
Q = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 4.0, 5.0], [1.0, 2.0, 3.0]])


def test_window_metrics_match_hand_computation():
    # Scale with m = 2: |1 - 0|, |5 - 2|, |3 - 1| → mean 2.
    scale = seasonal_scale(np.array([0.0, 2.0, 1.0, 5.0, 3.0]), 2)
    assert scale == 2.0

    h2, h4 = window_metrics(Y, Q, LEVELS, scale, (2, 4))

    # Errors of the median: 0, 1, -1, 2. Per-step QL (mean over levels of 2·ρ): 0.4/3, 1.4/3, 1.6/3, 4.4/3.
    # Covered by [q0.1, q0.9]: steps 1-3, not step 4 (4 > 3).
    assert h4 == pytest.approx({"horizon_steps": 4, "MAE": 1.0, "RMSE": math.sqrt(1.5), "MASE": 0.5,
                                "QL": 0.65, "SQL": 0.325, "COV80": 0.75})
    assert h2 == pytest.approx({"horizon_steps": 2, "MAE": 0.5, "RMSE": math.sqrt(0.5), "MASE": 0.25,
                                "QL": 0.3, "SQL": 0.15, "COV80": 1.0})


def test_point_forecast_as_all_quantiles_gives_ql_equal_to_mae():
    point = Q[:, 1]
    for levels in (LEVELS, DECILES):
        degenerate = np.repeat(point[:, None], len(levels), axis=1)
        (row,) = window_metrics(Y, degenerate, levels, 1.0, (4,))
        assert row["QL"] == pytest.approx(row["MAE"]) == pytest.approx(1.0)


def test_seasonal_scale_skips_pairs_with_a_gap_and_rejects_zero():
    assert seasonal_scale(np.array([0.0, 2.0, np.nan, 5.0, 3.0]), 2) == 3.0  # only |5 - 2| has both values
    assert math.isnan(seasonal_scale(np.array([4.0, 4.0, 4.0, 4.0]), 2))  # constant: a zero scale can't divide
    assert math.isnan(seasonal_scale(np.array([1.0, 2.0]), 2))  # no pair at all

    (row,) = window_metrics(Y, Q, LEVELS, math.nan, (4,))
    assert math.isnan(row["MASE"]) and math.isnan(row["SQL"])
    assert row["MAE"] == 1.0


def test_sorting_crossed_quantiles_lowers_the_loss_and_restores_coverage():
    y = np.array([2.0])
    crossed = np.array([[3.0, 2.0, 1.0]])
    repaired = np.sort(crossed, axis=-1)

    # Swapping the 0.1 and 0.9 values changes the summed pinball loss by (0.9 - 0.1)·(3 - 1) = 1.6, i.e. by
    # 2 · 1.6 / 3 in QL: 1.2 → 0.4 / 3.
    assert quantile_loss(y, crossed, LEVELS)[0] == pytest.approx(1.2)
    assert quantile_loss(y, repaired, LEVELS)[0] == pytest.approx(0.4 / 3)
    assert window_metrics(y, crossed, LEVELS, 1.0, (1,))[0]["COV80"] == 0.0
    assert window_metrics(y, repaired, LEVELS, 1.0, (1,))[0]["COV80"] == 1.0


# One row per (model, sensor, origin) with an MAE value per origin.
def _per_origin(errors: dict[tuple[str, str], list[float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"model": model, "sensor": sensor, "origin": pd.Timestamp("2026-02-01") + pd.Timedelta(hours=6 * i), "MAE": v}
        for (model, sensor), values in errors.items() for i, v in enumerate(values)
    ])


def test_relative_skill_is_the_geometric_mean_of_clipped_relative_errors():
    per_origin = _per_origin({
        ("snaive", "A"): [2.0, 2.0], ("x", "A"): [1.0, 1.0],      # relative error 0.5
        ("snaive", "B"): [1.0, 1.0], ("x", "B"): [2.0, 4.0],      # 3.0
        ("snaive", "C"): [1.0, 1.0], ("x", "C"): [0.0, 0.0],      # 0 → clipped to 0.01
        ("snaive", "D"): [1.0, 1.0], ("x", "D"): [500.0, 500.0],  # 500 → clipped to 100
    })
    relative = relative_errors(per_origin, "MAE", "snaive")

    x = relative[relative["model"] == "x"].set_index("sensor")["relative_error"]
    assert x.to_dict() == {"A": 0.5, "B": 3.0, "C": 0.0, "D": 500.0}
    assert (relative.loc[relative["model"] == "snaive", "relative_error"] == 1.0).all()

    skill = relative_skill(relative[relative["sensor"].isin(["A", "B"])]).set_index("model")
    assert skill.loc["x", "gmean_relative_error"] == pytest.approx(math.sqrt(1.5))
    assert skill.loc["x", "skill"] == pytest.approx(1 - math.sqrt(1.5))
    assert skill.loc["snaive", "skill"] == pytest.approx(0.0)
    assert skill.loc["x", "n_sensors"] == 2
    assert relative_skill(relative[relative["sensor"] == "C"]).set_index("model").loc["x", "gmean_relative_error"] \
        == pytest.approx(0.01)
    assert relative_skill(relative[relative["sensor"] == "D"]).set_index("model").loc["x", "gmean_relative_error"] \
        == pytest.approx(100.0)


def test_unpaired_origins_raise_unless_imputed():
    per_origin = _per_origin({("snaive", "A"): [2.0, 2.0], ("x", "A"): [1.0, np.nan]})  # x failed on one origin

    with pytest.raises(ValueError, match="UNPAIRED ORIGINS for x on A"):
        relative_errors(per_origin, "MAE", "snaive")

    imputed = relative_errors(per_origin, "MAE", "snaive", missing="impute").set_index("model")
    assert imputed.loc["x", "relative_error"] == 1.0
    assert bool(imputed.loc["x", "imputed"]) is True


def test_mixed_contexts_or_horizons_are_rejected():
    per_origin = _per_origin({("snaive", "A"): [2.0], ("x", "A"): [1.0]})
    with pytest.raises(ValueError, match="ONE ROW PER"):
        relative_errors(pd.concat([per_origin, per_origin]), "MAE", "snaive")


# Cross-check against AutoGluon's own metric classes (API checked against the v1.6.3 source on 2026-10-01: the scorer
# takes prediction_length and seasonal_period, error() is lower-is-better, the point forecast is the "mean" column, and
# the scale uses all data before the last prediction_length steps). Runs only in the AutoGluon venv (task 4).
def test_metrics_match_autogluon():
    pytest.importorskip("autogluon.timeseries")
    from autogluon.timeseries import TimeSeriesDataFrame
    from autogluon.timeseries.metrics import MAE, MASE, RMSE, SQL

    rng = np.random.default_rng(0)
    history, horizon = 4 * 96, 96
    index = pd.date_range("2026-02-01", periods=history + horizon, freq="15min")
    values = 10 + 5 * np.sin(np.arange(history + horizon) * 2 * np.pi / 96) + rng.normal(0, 1, history + horizon)
    quantiles = np.sort(values[history:, None] + rng.normal(0, 2, (horizon, len(DECILES))), axis=1)

    data = TimeSeriesDataFrame.from_data_frame(pd.DataFrame({"item_id": "A", "timestamp": index, "target": values}))
    predictions = TimeSeriesDataFrame.from_data_frame(pd.DataFrame({
        "item_id": "A", "timestamp": index[history:], "mean": quantiles[:, DECILES.index(0.5)],
        **{str(level): quantiles[:, j] for j, level in enumerate(DECILES)},
    }))
    (ours,) = window_metrics(values[history:], quantiles, DECILES, seasonal_scale(values[:history], 96), (horizon,))

    assert MAE(prediction_length=horizon).error(data, predictions) == pytest.approx(ours["MAE"])
    assert RMSE(prediction_length=horizon).error(data, predictions) == pytest.approx(ours["RMSE"])
    assert MASE(prediction_length=horizon, seasonal_period=96).error(data, predictions) == pytest.approx(ours["MASE"])
    assert SQL(prediction_length=horizon, seasonal_period=96).error(data, predictions) == pytest.approx(ours["SQL"])
