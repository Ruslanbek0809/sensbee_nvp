# Tests for the baseline forecasters in benchmark/models on short synthetic series:
# - output shape and finite values
# - batch independence (an origin forecast alone equals the same origin inside a batch)
# - Theta as a point forecaster
# - ETS without the season and without AutoGluon's silent fallback
# - exact seasonal naive on a periodic series
# Run only where AutoGluon and statsforecast are installed (the bench venv).

import numpy as np
import pytest

pytest.importorskip("autogluon.timeseries")
pytest.importorskip("statsforecast")

from benchmark.models import BASELINES  # noqa: E402
from benchmark.tasks import HORIZON, QUANTILE_LEVELS  # noqa: E402


@pytest.mark.parametrize("name", list(BASELINES))
def test_baseline_shape_finite_and_batch_independent(name, make_contexts):
    days = min(d for d in BASELINES[name].contexts if d >= 7)
    forecaster, params = BASELINES[name].build()
    contexts = make_contexts(3, days)

    batch, status = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)
    alone, _ = forecaster([contexts[1]], HORIZON, QUANTILE_LEVELS, 0)

    assert batch.shape == (3, HORIZON, len(QUANTILE_LEVELS)) and status == ["ok"] * 3
    assert np.isfinite(batch).all()
    np.testing.assert_allclose(alone[0], batch[1], rtol=1e-9, atol=1e-9)
    assert "point_only" in params


def test_theta_is_scored_as_a_point_forecaster(make_contexts):
    forecaster, params = BASELINES["ag_theta"].build()
    quantiles, _ = forecaster(make_contexts(2, 7), HORIZON, QUANTILE_LEVELS, 0)
    assert params["point_only"] is True
    assert np.ptp(quantiles, axis=-1).max() == 0  # every quantile is the same value: Theta's own forecast


def test_ets_has_no_season_and_no_silent_fallback(make_contexts):
    forecaster, params = BASELINES["ag_ets"].build()
    hyperparameters = params["hyperparameters"]
    assert hyperparameters["model"] == "ZZN"
    assert hyperparameters["use_fallback_model"] is False and hyperparameters["max_ts_length"] is None

    tiny = make_contexts(1, 1)[0].iloc[-5:]  # ETS can't fit 5 values ("tiny datasets")
    with pytest.raises(Exception):
        forecaster([tiny], HORIZON, QUANTILE_LEVELS, 0)


def test_seasonal_naive_96_repeats_the_last_day_of_a_periodic_series(make_contexts):
    forecaster, _ = BASELINES["ag_snaive96"].build()
    context = make_contexts(1, 8, noise=0.0)[0]
    quantiles, _ = forecaster([context], HORIZON, QUANTILE_LEVELS, 0)
    np.testing.assert_allclose(quantiles[0, :, QUANTILE_LEVELS.index(0.5)], context.to_numpy()[-96:], atol=1e-9)
