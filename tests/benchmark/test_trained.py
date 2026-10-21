# Tests for the trained forecasters in benchmark/models (TRAINED) on short synthetic series.
# All trained models:
# - every hyperparameter key is one AutoGluon accepts
# - the fitted models use no date features and the capped lags
# - the input window fits the 14-day context
# - no context shares an id with a training item
# - output shape and finite values, batch independence, and the same seed giving the same forecasts
# - no silent fit without training data or when AutoGluon skips the model
# DirectTabular is trained on quantiles. RecursiveTabular is a point forecaster.
# DeepAR:
# - the fitted network has no time features and the capped lags
# - early stopping gets the final days as tuning data
# - the distribution the network computes from a context doesn't depend on the other contexts in the batch
#   (its sampled paths do, through torch's Student-t sampler, whose random draws depend on every element's degrees
#   of freedom)
# - these tests use 2 short epochs (same network)
# PatchTST:
# - the same checks on its fitted network (one-week context, no time features)
# - no training steps held out beyond the tuning days, since its validation loss scores only the validated day
# Both GluonTS models sample training windows only where a full input window of real data exists.
# Known covariates (Phase 4, tabular models):
# - a known column that drives the target reaches the fitted model
# - it changes the forecast when its future changes
# - batch independence holds
# - a forecaster trained with covariates refuses a forecast without them
# Run only where AutoGluon is installed (the bench venv).

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("autogluon.timeseries")

from benchmark import covariates as cov  # noqa: E402
from benchmark.models import TRAINED  # noqa: E402
from benchmark.models.trained_models import (  # noqa: E402
    DEEPAR, DEEPAR_HOLDOUT, DIRECT_TABULAR, LAGS, PATCHTST, TUNING_DAYS, autogluon_trainer, input_steps, tuning_split,
)
from benchmark.tasks import BUCKETS_PER_DAY, HORIZON, QUANTILE_LEVELS  # noqa: E402

TABULAR = ("ag_direct_tabular", "ag_recursive_tabular")
MODEL_CLASSES = {"ag_direct_tabular": "DirectTabularModel", "ag_recursive_tabular": "RecursiveTabularModel"}
SHORT_DEEPAR = {**DEEPAR, "max_epochs": 2, "num_batches_per_epoch": 5}  # same network, 2 short epochs
SHORT_PATCHTST = {**PATCHTST, "max_epochs": 2, "num_batches_per_epoch": 5}
BUILT: dict = {}


# A noisy daily pattern: `days` of training (default 16), then 3 contexts of 14 days ending 0, 6 and 12 h later.
def _data(days: int = 16) -> tuple[list[pd.Series], list[pd.Series]]:
    index = pd.date_range("2026-01-01", periods=(days + 2) * BUCKETS_PER_DAY, freq="15min")
    rng = np.random.default_rng(0)
    y = pd.Series(10 + 5 * np.sin(np.arange(len(index)) * 2 * np.pi / 96) + rng.normal(0, 0.3, len(index)),
                  index=index)
    contexts = [y.iloc[days * 96 - 1344 + 24 * k: days * 96 + 24 * k] for k in range(3)]
    return [y.iloc[: days * 96]], contexts


# The fitted AutoGluon model behind a trained forecaster, loaded from the predictor's folder.
def _fitted_model(forecaster):
    return forecaster.predictor._learner.load_trainer().load_model(forecaster.predictor.model_names()[0])


# The trainer, its parameters and a forecaster fitted with seed 0 on _data(), built once per test session.
def _fitted(name: str) -> tuple:
    if name not in BUILT:
        trainer, params = TRAINED[name].build()
        items, _ = _data()
        forecaster, info = trainer("S", items, 0)
        BUILT[name] = (trainer, params, forecaster, info)
    return BUILT[name]


@pytest.mark.parametrize("name", TABULAR)
def test_hyperparameters_are_known_values_only_and_fit_the_context(name):
    from autogluon.timeseries import models
    _, params, forecaster, _ = _fitted(name)
    hyperparameters = params["hyperparameters"]
    allowed = getattr(models, MODEL_CLASSES[name])(freq="15min", prediction_length=HORIZON).allowed_hyperparameters
    assert set(hyperparameters) <= set(allowed)
    assert hyperparameters["date_features"] == [] and max(hyperparameters["lags"]) == 672

    model = _fitted_model(forecaster)
    assert model._date_features == [] and list(model._target_lags) == list(LAGS)  # what the fitted model uses
    # DirectTabular trains a quantile LightGBM (its deciles are its own); a point regressor would add Gaussian ones.
    assert model.get_tabular_model().model.problem_type == ("regression" if params["point_only"] else "quantile")
    # Training items have ids no context can have, so every context is a series the model didn't see in fit.
    assert all(item.startswith("train") for item in model._residuals_std_per_item.index)


@pytest.mark.parametrize("name", TABULAR)
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
        "ag_direct_tabular": False, "ag_recursive_tabular": True, "ag_deepar": False, "ag_patchtst": False}


@pytest.mark.parametrize("name", TABULAR)
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



def test_every_trained_input_window_fits_the_shortest_context():
    for name, baseline in TRAINED.items():
        assert baseline.build()[1]["input_steps"] <= min(baseline.contexts) * BUCKETS_PER_DAY, name
    assert input_steps(DEEPAR) == 2 * 96 + 672 - 1 and input_steps(PATCHTST) == 672


def test_tuning_split_holds_out_the_final_days_one_window_each():
    index = pd.date_range("2026-01-01", periods=25 * BUCKETS_PER_DAY, freq="15min")
    first, last = pd.Series(1.0, index=index[:5 * 96]), pd.Series(np.arange(20 * 96.0), index=index[5 * 96:])
    train, tune = tuning_split([first, last], 7, 863, holdout=191)

    assert train[0] is first
    assert train[1].index[-1] == last.index[-1] - pd.Timedelta(days=7) - 191 * pd.Timedelta("15min")
    assert [t.index[-1] for t in tune] == [last.index[-1] - pd.Timedelta(days=k) for k in range(7)]
    # Every step a GluonTS validation loss scores (the last 96 of a tuning item and the 191 before them) comes after
    # the last training step.
    assert all(t.index[-(96 + 191)] > train[1].index[-1] for t in tune)
    with pytest.raises(ValueError, match="TOO SHORT"):  # 1,057 steps left < a 1,055-step window plus a horizon
        tuning_split([first, last], 7, 863 + 192, holdout=191)


def test_registered_deepar_values_only_with_early_stopping_on_the_final_week():
    _, params = TRAINED["ag_deepar"].build()
    hyperparameters = params["hyperparameters"]
    assert hyperparameters["time_features"] == [] and max(hyperparameters["lags_seq"]) == 672
    assert hyperparameters["disable_static_features"] and hyperparameters["disable_known_covariates"]
    assert params["tuning_days"] == TUNING_DAYS == 7 and params["point_only"] is False
    assert params["holdout_steps"] == DEEPAR_HOLDOUT == 191 and params["train_min_past"] == 863


def test_registered_patchtst_one_week_context_with_early_stopping_on_the_final_week():
    _, params = TRAINED["ag_patchtst"].build()
    hyperparameters = params["hyperparameters"]
    assert hyperparameters["context_length"] == 672 and hyperparameters["patch_len"] == 16
    assert hyperparameters["disable_static_features"] and hyperparameters["disable_known_covariates"]
    assert params["tuning_days"] == TUNING_DAYS and params["holdout_steps"] == 0 and params["point_only"] is False
    assert params["train_min_past"] == 672 and TRAINED["ag_direct_tabular"].build()[1]["train_min_past"] == 0


# The distribution parameters the network's projection (`layer`, e.g. "param_proj") computes over each context in
# one forecast call: the first call of the projection, on the context window. AutoGluon loads the model from disk for
# each prediction, so the projection is wrapped at the class level.
def _projected(monkeypatch, forecaster, network, layer: str, contexts: list[pd.Series]) -> list[np.ndarray]:
    captured = []
    projection = type(getattr(network, layer))
    original = projection.forward
    monkeypatch.setattr(projection, "forward", lambda self, x: captured.append(original(self, x)) or captured[-1])
    forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)
    monkeypatch.setattr(projection, "forward", original)
    return [t.detach().clone().numpy() for t in captured[0]]


# No path between series: a context's distribution parameters are the same whatever the other contexts are, and a
# changed context changes its own. The sampled paths may still differ, because torch's Student-t sampler consumes
# random numbers depending on every element's degrees of freedom.
def _assert_batch_independent(monkeypatch, forecaster, network, layer: str, contexts: list[pd.Series]) -> None:
    alone = _projected(monkeypatch, forecaster, network, layer, contexts)
    mixed = _projected(monkeypatch, forecaster, network, layer,
                       [contexts[0] + 100.0, contexts[1], contexts[2] - 50.0])
    assert alone[0].shape[0] == 3
    for a, b in zip(alone, mixed):
        np.testing.assert_allclose(a[1], b[1], rtol=1e-6, atol=1e-6)
        assert not np.allclose(a[0], b[0])


def test_deepar_network_early_stopping_seed_and_sampling(monkeypatch):
    trainer, _ = autogluon_trainer("DeepAR", SHORT_DEEPAR, tuning_days=TUNING_DAYS, holdout=DEEPAR_HOLDOUT)
    items, contexts = _data(days=20)  # 13 days of training after the 7 tuning days
    forecaster, info = trainer("S", items, 0)

    model = _fitted_model(forecaster)
    network = model.gts_predictor.prediction_net.model
    assert network.context_length == 192 and max(network.lags_seq) == 671  # GluonTS stores lag - 1
    assert network._past_length == input_steps(DEEPAR)
    assert model._get_estimator().train_sampler.min_past == input_steps(DEEPAR)  # no zero-padded training windows
    assert network.num_feat_dynamic_real == 1  # only the built-in age feature: no time features
    # Early stopping ran on the tuning data: a validation loss was logged every epoch, and the fit record has it.
    assert info["tuning_days"] == 7 and info["holdout_steps"] == 191
    assert 1 <= info["best_epoch"] <= info["epochs_run"] <= SHORT_DEEPAR["max_epochs"]
    assert np.isfinite(info["best_val_loss"])
    assert info["training_rows"] == 13 * 96 - 191

    batch, status = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)
    assert batch.shape == (3, HORIZON, len(QUANTILE_LEVELS)) and status == ["ok"] * 3 and np.isfinite(batch).all()
    np.testing.assert_array_equal(batch, forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)[0])

    assert _projected(monkeypatch, forecaster, network, "param_proj", contexts)[0].shape[1] >= 192  # context window
    _assert_batch_independent(monkeypatch, forecaster, network, "param_proj", contexts)


def test_patchtst_network_early_stopping_seed_and_batch_independence(monkeypatch):
    trainer, _ = autogluon_trainer("PatchTST", SHORT_PATCHTST, tuning_days=TUNING_DAYS)
    items, contexts = _data(days=20)
    forecaster, info = trainer("S", items, 0)

    model = _fitted_model(forecaster)
    network = model.gts_predictor.prediction_net.model
    assert network.context_length == 672 and network.patch_len == 16 and network.stride == 8
    assert network.num_feat_dynamic_real == 0  # no time features, no covariates
    assert model._get_estimator().train_sampler.min_past == 672  # no zero-padded training windows
    assert info["tuning_days"] == 7 and info["holdout_steps"] == 0 and info["training_rows"] == 13 * 96
    assert 1 <= info["best_epoch"] <= info["epochs_run"] <= SHORT_PATCHTST["max_epochs"]
    assert np.isfinite(info["best_val_loss"])

    batch, status = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)
    assert batch.shape == (3, HORIZON, len(QUANTILE_LEVELS)) and status == ["ok"] * 3 and np.isfinite(batch).all()
    np.testing.assert_array_equal(batch, forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)[0])
    _assert_batch_independent(monkeypatch, forecaster, network, "args_proj", contexts)


# _data() plus a known on/off column "k" (random 2-h blocks) that adds 8 to the target, the trainer's frame of it
# before the cutoff, and each context's covariate window as the runner builds it.
def _covariate_data() -> tuple[list[pd.Series], list[pd.Series], pd.DataFrame, list[pd.DataFrame]]:
    items, contexts = _data()
    index = pd.date_range("2026-01-01", periods=18 * BUCKETS_PER_DAY, freq="15min")
    k = np.repeat(np.random.default_rng(1).integers(0, 2, len(index) // 8), 8).astype(float)
    frame = pd.DataFrame({"k": k}, index=index)
    shift = pd.Series(8 * k, index=index)
    items = [item + shift.reindex(item.index) for item in items]
    contexts = [c + shift.reindex(c.index) for c in contexts]
    kinds = {"k": "known"}
    windows = [cov.covariate_window(frame, kinds, c.index[-1] + pd.Timedelta("15min"), c.index) for c in contexts]
    return items, contexts, cov.training_covariates(frame, kinds, items[0].index[-1] + pd.Timedelta("15min")), windows


@pytest.mark.parametrize("name", TABULAR)
def test_known_covariates_reach_the_tabular_model_and_its_forecast(name):
    trainer, _ = TRAINED[name].build()
    items, contexts, frame, windows = _covariate_data()
    forecaster, info = trainer("S", items, 0, covariates=frame)

    model = _fitted_model(forecaster)
    assert model.covariate_metadata.known_covariates == ["k"]
    assert info["known_covariates"] == ["k"] and info["covariate_scaler"] is None
    batch, status = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0, covariates=windows)
    alone, _ = forecaster([contexts[1]], HORIZON, QUANTILE_LEVELS, 0, covariates=[windows[1]])
    assert status == ["ok"] * 3 and np.isfinite(batch).all()
    np.testing.assert_allclose(alone[0], batch[1], rtol=1e-9, atol=1e-9)
    flipped = []
    for c, w in zip(contexts, windows):
        f = w.copy()
        f.iloc[len(c):, 0] = 1 - f.iloc[len(c):, 0]
        flipped.append(f)
    other, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0, covariates=flipped)
    # where k flips from 0 to 1 the forecast should rise by about 8 (the model learned the effect)
    up = np.array([w["k"].iloc[len(c):].to_numpy() == 0 for c, w in zip(contexts, windows)])
    rise = (other[..., 4] - batch[..., 4])[up]
    print(f"{name}: MEAN RISE WHERE k FLIPS 0 -> 1 = {rise.mean():.2f}")
    assert rise.mean() > 4
    with pytest.raises(ValueError, match="EVERY FORECAST NEEDS THEM"):
        forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)
