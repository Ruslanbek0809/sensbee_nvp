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

from benchmark.models import ZERO_SHOT  # noqa: E402
from benchmark.tasks import HORIZON, QUANTILE_LEVELS  # noqa: E402
from src.data_access.data_loader import load_sensor_series_from_json  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "temp_14day_sensbee_data.json"
SHAPE = (HORIZON, len(QUANTILE_LEVELS))


# (name, forecaster, parameters) of every zero-shot model in turn.
@pytest.fixture(params=sorted(ZERO_SHOT))
def model(request, zero_shot):
    return request.param, *zero_shot(request.param)


def test_shape_independence_and_determinism(model, make_contexts):
    name, forecaster, params = model
    contexts = make_contexts(3, 7)

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


def test_chronos_bolt_equals_autogluon_chronos(tmp_path, make_contexts, autogluon_deciles, zero_shot):
    forecaster, _ = zero_shot("chronos_bolt_base")
    model_id, revision, _ = ZERO_SHOT["chronos_bolt_base"].weights
    contexts = make_contexts(3, 7)

    reference = autogluon_deciles("Chronos", {"model_path": model_id, "revision": revision, "device": "cpu",
                                              "torch_dtype": "float32", "context_length": None, "batch_size": 64},
                                  contexts, tmp_path / "predictor")
    ours, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)

    print(f"chronos_bolt_base: MAX |OURS - AUTOGLUON| = {np.abs(ours - reference).max():.3g}")
    np.testing.assert_allclose(ours, reference, rtol=1e-5, atol=1e-4)


# Chronos-Bolt and TiRex read at most 2,048 steps, so a 28-day context gives the forecast of its last 2,048 steps.
@pytest.mark.parametrize("name", ["chronos_bolt_base", "tirex"])
def test_models_cut_contexts_at_2048_steps(name, make_contexts, zero_shot):
    forecaster, params = zero_shot(name)
    context = make_contexts(1, 28)[0]  # 2,688 steps

    full, _ = forecaster([context], HORIZON, QUANTILE_LEVELS, 0)
    cut, _ = forecaster([context.iloc[-2048:]], HORIZON, QUANTILE_LEVELS, 0)

    assert params["model_context_length"] == 2048
    np.testing.assert_array_equal(full, cut)


# TimesFM clips at 0 by default when a context is never negative (infer_is_positive); the adapter turns that off, so a
# non-negative series that falls to 0 at the end may get negative low quantiles, as the other models' can.
def test_timesfm_does_not_clip_non_negative_contexts(zero_shot):
    forecaster, params = zero_shot("timesfm25")
    index = pd.date_range("2026-01-01", periods=7 * 96, freq="15min")
    falling = pd.Series(np.maximum(0, 10 - np.arange(len(index)) * 10 / len(index)) + 2 * np.abs(np.sin(
        np.arange(len(index)) * np.pi / 96)), index=index)

    quantiles, _ = forecaster([falling], HORIZON, QUANTILE_LEVELS, 0)

    assert params["infer_is_positive"] is False and params["fix_quantile_crossing"] is False
    print(f"timesfm25: MIN q0.1 ON A NON-NEGATIVE CONTEXT = {quantiles[0, :, 0].min():.3g}")
    assert quantiles[0, :, 0].min() < 0


# The longest context TimesFM runs on (169 days = 16,224 steps) fits its limit and gives a full forecast.
def test_timesfm_runs_the_169_day_context(make_contexts, zero_shot):
    forecaster, params = zero_shot("timesfm25")
    context = make_contexts(1, 169)[0]

    quantiles, status = forecaster([context], HORIZON, QUANTILE_LEVELS, 0)

    assert len(context) <= params["model_context_length"] == 16256
    assert quantiles.shape == (1, *SHAPE) and status == ["ok"] and np.isfinite(quantiles).all()


# Each TiRex call runs on one torch thread (so a forecast alone equals it in a batch) without frequency resampling;
# the caller's thread count is restored afterwards.
def test_tirex_runs_on_one_thread_and_restores_the_count(make_contexts, zero_shot):
    import torch
    forecaster, params = zero_shot("tirex")
    threads = torch.get_num_threads()

    forecaster(make_contexts(1, 7), HORIZON, QUANTILE_LEVELS, 0)

    assert params["torch_threads"] == 1 and params["resample_strategy"] is None
    assert torch.get_num_threads() == threads
