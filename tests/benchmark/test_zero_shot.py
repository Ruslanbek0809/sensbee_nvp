# Integration tests shared by every zero-shot forecaster in benchmark/models (ZERO_SHOT): output shape, batch
# independence (an origin forecast alone equals the same origin inside a batch), determinism for the same seed, an
# all-zero context (closed facility) and a smoke run on the RP temperature fixture. A model is skipped when its package
# doesn't import or its weight file isn't in the local Hugging Face cache, so these tests never download anything. A
# model that samples (params "num_samples") is checked for seed effects here; its batch independence is tested in its
# own module, because its random draws depend on the batch layout. Run in the bench venv with
#   HF_HUB_OFFLINE=1 venv-bench/bin/python -m pytest tests/benchmark/test_zero_shot.py -m integration -q -s

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("torch")

from huggingface_hub import try_to_load_from_cache  # noqa: E402

from benchmark.models import ZERO_SHOT  # noqa: E402
from benchmark.tasks import HORIZON, QUANTILE_LEVELS  # noqa: E402
from src.data_access.data_loader import load_sensor_series_from_json  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "temp_14day_sensbee_data.json"
PATTERN = 10 + 5 * np.sin(np.arange(96) * 2 * np.pi / 96)
SHAPE = (HORIZON, len(QUANTILE_LEVELS))
BUILT: dict = {}


# n daily-periodic contexts of `days` days with a little noise, each with its own noise and end time.
def _contexts(n: int, days: int, noise: float = 0.3) -> list[pd.Series]:
    out = []
    for i in range(n):
        rng = np.random.default_rng(i)
        index = pd.date_range("2026-01-01", periods=days * 96, freq="15min") + pd.Timedelta(hours=6 * i)
        values = np.tile(np.roll(PATTERN, -24 * i), days) + rng.normal(0, noise, days * 96)
        out.append(pd.Series(values, index=index))
    return out


# The forecaster and parameters of one zero-shot model, built once per test session; skips the test when the model's
# weights aren't cached or its package is missing.
@pytest.fixture(params=sorted(ZERO_SHOT))
def model(request):
    name = request.param
    model_id, revision, filename = ZERO_SHOT[name].weights
    if not isinstance(try_to_load_from_cache(model_id, filename, revision=revision), str):
        pytest.skip(f"{model_id} at {revision} is not in the local Hugging Face cache")
    if name not in BUILT:
        try:
            BUILT[name] = ZERO_SHOT[name].build()
        except ImportError as exc:
            pytest.skip(f"{name}: {exc}")
    return name, *BUILT[name]


def test_shape_independence_and_determinism(model):
    name, forecaster, params = model
    contexts = _contexts(3, 7)

    batch, status = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)
    again, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)

    assert batch.shape == (3, *SHAPE) and status == ["ok"] * 3
    assert np.isfinite(batch).all()
    np.testing.assert_array_equal(again, batch)
    assert params["revision"] == ZERO_SHOT[name].weights[1] and params["point_only"] is False
    if "num_samples" in params:
        other, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 1)
        assert not np.array_equal(other, batch)
    else:
        alone, _ = forecaster([contexts[1]], HORIZON, QUANTILE_LEVELS, 0)
        print(f"{name}: MAX |ALONE - BATCH| = {np.abs(alone[0] - batch[1]).max():.3g}")
        np.testing.assert_allclose(alone[0], batch[1], rtol=1e-5, atol=1e-4)


# A closed day: the models fall back to a tiny scale (e.g. 1e-5) when the context has no spread, so a scaling bug
# would show as huge values. The bound is loose on purpose: Chronos-Bolt's q0.9 climbs to about 2 by step 64 after a
# single all-zero day (2026-10-07); the largest value is printed for every model.
def test_all_zero_context_gives_finite_forecasts_near_zero(model):
    name, forecaster, _ = model
    closed = pd.Series(0.0, index=pd.date_range("2026-01-01", periods=96, freq="15min"))

    quantiles, status = forecaster([closed], HORIZON, QUANTILE_LEVELS, 0)

    print(f"{name}: MAX |Q| AFTER ONE ALL-ZERO DAY = {np.abs(quantiles).max():.3g}")
    assert quantiles.shape == (1, *SHAPE) and status == ["ok"]
    assert np.isfinite(quantiles).all() and np.abs(quantiles).max() < 5


def test_rp_temperature_fixture_smoke(model):
    name, forecaster, _ = model
    series = load_sensor_series_from_json(str(FIXTURE), column_name="temperature")

    quantiles, status = forecaster([series.iloc[-7 * 96:]], HORIZON, QUANTILE_LEVELS, 0)

    assert quantiles.shape == (1, *SHAPE) and status == ["ok"]
    assert np.isfinite(quantiles).all()
    print(f"{name}: CROSSED STEPS ON THE FIXTURE = {int(np.any(np.diff(quantiles, axis=-1) < 0, axis=-1).sum())}")


# The forecaster of one named model, built once; skips when its weights aren't cached.
def _named(name: str):
    model_id, revision, filename = ZERO_SHOT[name].weights
    if not isinstance(try_to_load_from_cache(model_id, filename, revision=revision), str):
        pytest.skip(f"{model_id} at {revision} is not in the local Hugging Face cache")
    if name not in BUILT:
        BUILT[name] = ZERO_SHOT[name].build()
    return BUILT[name]


# Deciles of AutoGluon's own wrapper for one model on the given contexts (no fitting: a zero-shot model only loads).
def _autogluon_deciles(model: str, hyperparameters: dict, contexts: list[pd.Series], path: Path) -> np.ndarray:
    pytest.importorskip("autogluon.timeseries")
    from autogluon.timeseries import TimeSeriesPredictor

    from benchmark.models.autogluon_models import _batch_frame
    from benchmark.snapshot import BUCKET_FREQ
    data = _batch_frame(contexts)
    predictor = TimeSeriesPredictor(prediction_length=HORIZON, freq=BUCKET_FREQ, quantile_levels=list(QUANTILE_LEVELS),
                                    path=str(path), verbosity=0, log_to_file=False)
    predictor.fit(data, hyperparameters={model: hyperparameters}, enable_ensemble=False, skip_model_selection=True)
    predictions = predictor.predict(data)
    return np.stack([predictions.loc[f"{i:06d}", [str(q) for q in QUANTILE_LEVELS]].to_numpy(dtype=float)
                     for i in range(len(contexts))])


def test_chronos_bolt_equals_autogluon_chronos(tmp_path):
    forecaster, _ = _named("chronos_bolt_base")
    model_id, revision, _ = ZERO_SHOT["chronos_bolt_base"].weights
    contexts = _contexts(3, 7)

    reference = _autogluon_deciles("Chronos", {"model_path": model_id, "revision": revision, "device": "cpu",
                                               "torch_dtype": "float32", "context_length": None, "batch_size": 64},
                                   contexts, tmp_path / "predictor")
    ours, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)

    print(f"chronos_bolt_base: MAX |OURS - AUTOGLUON| = {np.abs(ours - reference).max():.3g}")
    np.testing.assert_allclose(ours, reference, rtol=1e-5, atol=1e-4)


def test_chronos_bolt_cuts_contexts_at_2048_steps():
    forecaster, params = _named("chronos_bolt_base")
    context = _contexts(1, 28)[0]  # 2,688 steps

    full, _ = forecaster([context], HORIZON, QUANTILE_LEVELS, 0)
    cut, _ = forecaster([context.iloc[-2048:]], HORIZON, QUANTILE_LEVELS, 0)

    assert params["model_context_length"] == 2048
    np.testing.assert_array_equal(full, cut)
