# Tests for the trained forecasters in benchmark/models (TRAINED) on short synthetic series: every hyperparameter key
# is one AutoGluon accepts, the fitted models use no date features and the capped lags, the input window fits the
# 14-day context, DirectTabular trained on quantiles, no context sharing an id with a training item, output shape and
# finite values, batch independence, RecursiveTabular as a point forecaster, the same seed giving the same forecasts,
# and no silent fit without training data or when AutoGluon skips the model. Run only where AutoGluon is installed (the
# bench venv).

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("autogluon.timeseries")

from benchmark.models import TRAINED  # noqa: E402
from benchmark.models.trained_models import DIRECT_TABULAR, LAGS, autogluon_trainer  # noqa: E402
from benchmark.tasks import BUCKETS_PER_DAY, HORIZON, QUANTILE_LEVELS  # noqa: E402

MODEL_CLASSES = {"ag_direct_tabular": "DirectTabularModel", "ag_recursive_tabular": "RecursiveTabularModel"}
BUILT: dict = {}


# 30 days of a noisy daily pattern: 16 days of training, then 3 contexts of 14 days, each 6 h later than the last.
def _data() -> tuple[list[pd.Series], list[pd.Series]]:
    index = pd.date_range("2026-01-01", periods=30 * BUCKETS_PER_DAY, freq="15min")
    rng = np.random.default_rng(0)
    y = pd.Series(10 + 5 * np.sin(np.arange(len(index)) * 2 * np.pi / 96) + rng.normal(0, 0.3, len(index)),
                  index=index)
    contexts = [y.iloc[16 * 96 - 1344 + 24 * k: 16 * 96 + 24 * k] for k in range(3)]
    return [y.iloc[: 16 * 96]], contexts


# The trainer, its parameters and a forecaster fitted with seed 0 on _data(), built once per test session.
def _fitted(name: str) -> tuple:
    if name not in BUILT:
        trainer, params = TRAINED[name].build()
        items, _ = _data()
        forecaster, info = trainer("S", items, 0)
        BUILT[name] = (trainer, params, forecaster, info)
    return BUILT[name]


@pytest.mark.parametrize("name", list(TRAINED))
def test_hyperparameters_are_known_values_only_and_fit_the_context(name):
    from autogluon.timeseries import models
    _, params, forecaster, _ = _fitted(name)
    hyperparameters = params["hyperparameters"]
    allowed = getattr(models, MODEL_CLASSES[name])(freq="15min", prediction_length=HORIZON).allowed_hyperparameters
    assert set(hyperparameters) <= set(allowed)
    assert hyperparameters["date_features"] == [] and max(hyperparameters["lags"]) == 672
    assert params["input_steps"] <= min(TRAINED[name].contexts) * BUCKETS_PER_DAY

    model = forecaster.predictor._learner.load_trainer().load_model(forecaster.predictor.model_names()[0])
    assert model._date_features == [] and list(model._target_lags) == list(LAGS)  # what the fitted model uses
    # DirectTabular trains a quantile LightGBM (its deciles are its own); a point regressor would add Gaussian ones.
    assert model.get_tabular_model().model.problem_type == ("regression" if params["point_only"] else "quantile")
    # Training items have ids no context can have, so every context is a series the model didn't see in fit.
    assert all(item.startswith("train") for item in model._residuals_std_per_item.index)


@pytest.mark.parametrize("name", list(TRAINED))
def test_trained_shape_finite_and_batch_independent(name):
    _, params, forecaster, info = _fitted(name)
    _, contexts = _data()

    batch, status = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)
    alone, _ = forecaster([contexts[1]], HORIZON, QUANTILE_LEVELS, 0)

    assert batch.shape == (3, HORIZON, len(QUANTILE_LEVELS)) and status == ["ok"] * 3
    assert np.isfinite(batch).all()
    np.testing.assert_allclose(alone[0], batch[1], rtol=1e-9, atol=1e-9)
    assert info["training_rows"] == 16 * 96
    spread = np.ptp(batch, axis=-1).max()
    assert (spread == 0) if params["point_only"] else (spread > 0)


def test_only_recursive_tabular_is_a_point_forecaster():
    assert {name: TRAINED[name].build()[1]["point_only"] for name in TRAINED} == {
        "ag_direct_tabular": False, "ag_recursive_tabular": True}


@pytest.mark.parametrize("name", list(TRAINED))
def test_the_same_seed_gives_the_same_forecasts(name):
    trainer, _, forecaster, _ = _fitted(name)
    items, contexts = _data()
    again, _ = trainer("S", items, 0)
    np.testing.assert_array_equal(forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)[0],
                                  again(contexts, HORIZON, QUANTILE_LEVELS, 0)[0])


def test_no_training_items_raise_instead_of_fitting_nothing():
    trainer, _ = TRAINED["ag_direct_tabular"].build()
    with pytest.raises(ValueError, match="NO TRAINING ITEMS"):
        trainer("S", [], 0)


def test_a_model_autogluon_skips_raises_instead_of_forecasting_nothing():
    trainer, _ = autogluon_trainer("DirectTabular", {**DIRECT_TABULAR, "model_name": "NO_SUCH_TABULAR_MODEL"})
    items, _ = _data()
    with pytest.raises(RuntimeError, match="TRAINED NO DirectTabular MODEL"):
        trainer("S", items, 0)
