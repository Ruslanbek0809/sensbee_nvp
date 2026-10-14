# Tests for the LLMTime harness forecaster (benchmark/models/llmtime_models.py) with a mocked completion function
# (no model load): point-only output (every quantile the median), a visible "fallback" when every sample is filtered,
# no clipping of negative forecasts, one origin failing without failing the others, the seed set before every origin,
# filter counts drained per call, and the registry entry. Needs torch (the bench venv).

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from benchmark.models import LLMTIME, MISTRAL_7B  # noqa: E402
from benchmark.models.llmtime_models import llmtime_forecaster  # noqa: E402
from benchmark.tasks import HORIZON, QUANTILE_LEVELS  # noqa: E402
from src.models.serialize import create_scaler  # noqa: E402

STEPS = np.arange(HORIZON)
WAVE = 500 + np.round(300 * np.sin(STEPS / 5)).astype(int)
LOW = -1500 + np.round(300 * np.sin(STEPS / 5)).astype(int)  # maps below 0 for a context starting at 0


# Serializes integers like the local LLMTime settings.
def _text(values) -> str:
    return ",".join(str(int(v)) for v in values)


# A 14-day context with a daily cycle between `low` and `low` + 20, ending at `end`.
def _context(low: float = 0.0, end: str = "2026-02-15") -> pd.Series:
    index = pd.date_range(end=pd.Timestamp(end) - pd.Timedelta("15min"), periods=1344, freq="15min")
    return pd.Series(low + 10 + 10 * np.sin(np.arange(1344) * 2 * np.pi / 96), index=index)


# A completion function giving the n-th answer on its n-th call (one call per origin): "fail" raises, "flat" gives
# only filtered samples, "low" puts a sample far below the context, anything else the wave. Records the torch seed seen.
def _complete(answers: list, seeds: list):
    calls = iter(answers)

    def complete(prompt, settings, num_forecasts, **_):
        import torch
        seeds.append(torch.initial_seed())
        answer = next(calls)
        if answer == "fail":
            raise RuntimeError("MPS out of memory")
        if answer == "flat":
            return [_text([500] * HORIZON)] * num_forecasts
        return [_text(WAVE), _text(WAVE + 10), _text(LOW if answer == "low" else WAVE + 20)][:num_forecasts]
    return complete


def test_point_only_fallback_failure_and_counts():
    contexts = [_context(), _context(low=5.0), _context(low=-3.0)]
    seeds = []
    forecast, params = llmtime_forecaster(*MISTRAL_7B, num_forecasts=3,
                                          complete=_complete(["wave", "flat", "fail"], seeds))

    out, status = forecast(contexts, HORIZON, QUANTILE_LEVELS, 7)

    assert status == ["ok", "fallback", "failed"] and out.shape == (3, HORIZON, len(QUANTILE_LEVELS))
    assert np.ptp(out[:2], axis=-1).max() == 0  # point-only: every quantile is the median
    scaler = create_scaler(contexts[0].to_numpy(), 0.95, 0.3)
    np.testing.assert_allclose(out[0, :, 0], scaler.inverse_transform((WAVE + 10) / 1000))
    np.testing.assert_array_equal(out[1, :, 4], np.full(HORIZON, contexts[1].iloc[-1]))
    assert np.isnan(out[2]).all() and seeds == [7, 7, 7]
    assert forecast.drain_diagnostics() == {"completions": 6, "parse_failed": 0, "filtered_a": 0, "filtered_b": 3,
                                            "filtered_c": 0, "filtered_d": 0, "valid": 3}
    assert forecast.drain_diagnostics() == {}
    assert params["point_only"] and params["clipping"] is None and params["batch_size"] == 1
    assert (params["model_id"], params["revision"]) == MISTRAL_7B and params["dtype"] == "float16"
    assert params["multi_gpu"] is False and params["device_backend"] is None  # mocked: no device used


def test_negative_forecasts_are_not_clipped_and_an_origin_does_not_depend_on_its_batch():
    context = _context()
    low, _ = llmtime_forecaster(*MISTRAL_7B, num_forecasts=3, complete=_complete(["low"], []))[0](
        [context], HORIZON, QUANTILE_LEVELS, 0)
    assert low.min() < 0  # the median of WAVE, WAVE + 10 and LOW, mapped back without clipping at 0

    alone, _ = llmtime_forecaster(*MISTRAL_7B, complete=_complete(["wave"], []))[0](
        [context], HORIZON, QUANTILE_LEVELS, 0)
    batch, _ = llmtime_forecaster(*MISTRAL_7B, complete=_complete(["flat", "wave"], []))[0](
        [_context(low=5.0), context], HORIZON, QUANTILE_LEVELS, 0)
    np.testing.assert_array_equal(alone[0], batch[1])


def test_registered_llmtime_runs_on_14_days_with_the_pinned_mistral():
    baseline = LLMTIME["llmtime_mistral7b"]
    assert baseline.contexts == (14,)
    assert baseline.weights == (*MISTRAL_7B, "model-00001-of-00002.safetensors")


# Local generation: one device, one sample at a time, at the pinned revision; no GPU backend fails the build.
def test_local_generation_is_single_device_and_needs_a_gpu(monkeypatch):
    from src.models import local_llm
    from src.models.serialize import get_serializer_settings
    calls = []
    monkeypatch.setattr(local_llm, "get_device_backend", lambda: "mps")
    monkeypatch.setattr(local_llm, "local_llm_completion", lambda **kwargs: calls.append(kwargs) or [_text(WAVE)])
    forecast, params = llmtime_forecaster(*MISTRAL_7B, num_forecasts=1)
    forecast([_context()], HORIZON, QUANTILE_LEVELS, 0)
    assert params["device_backend"] == "mps"
    assert {k: calls[0][k] for k in ("model_name", "revision", "dtype", "batch_size", "multi_gpu")} == {
        "model_name": MISTRAL_7B[0], "revision": MISTRAL_7B[1], "dtype": "float16", "batch_size": 1, "multi_gpu": False}
    assert calls[0]["settings"] == get_serializer_settings(model_name=MISTRAL_7B[0], provider="local", is_local=True)

    monkeypatch.setattr(local_llm, "get_device_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="NEEDS A GPU"):
        llmtime_forecaster(*MISTRAL_7B)

