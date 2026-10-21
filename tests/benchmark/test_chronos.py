# Integration tests for the Chronos-2 forecaster in benchmark/models:
# - output shape, determinism, and batch independence (an origin forecast alone equals the same origin inside a batch)
# - an all-zero context (closed facility)
# - a smoke run on the RP temperature fixture
# - a cross-check against AutoGluon's Chronos2 wrapper with cross-learning off
# - covariate windows (Phase 4): batch independence, future covariates are used, and the future of a past-only column
#   never reaches the model
# They need the model weights in the local Hugging Face cache and never download them. Run in the bench venv with
#   HF_HUB_OFFLINE=1 venv-bench/bin/python -m pytest tests/benchmark/test_chronos.py -m integration -q

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("chronos")

from huggingface_hub import try_to_load_from_cache  # noqa: E402

from benchmark import covariates as cov  # noqa: E402
from benchmark.models import CHRONOS2, ZERO_SHOT  # noqa: E402
from benchmark.tasks import HORIZON, QUANTILE_LEVELS  # noqa: E402
from src.data_access.data_loader import load_sensor_series_from_json  # noqa: E402

MODEL_ID, REVISION = CHRONOS2
if not isinstance(try_to_load_from_cache(MODEL_ID, "model.safetensors", revision=REVISION), str):
    pytest.skip(f"{MODEL_ID} at {REVISION} is not in the local Hugging Face cache", allow_module_level=True)

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "temp_14day_sensbee_data.json"


# The forecaster is built once: loading the weights takes seconds.
@pytest.fixture(scope="module")
def chronos2():
    return ZERO_SHOT["chronos2"].build()


def test_shape_batch_independence_and_determinism(chronos2, make_contexts):
    forecaster, params = chronos2
    contexts = make_contexts(3, 7)

    batch, status = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)
    again, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)
    alone, _ = forecaster([contexts[1]], HORIZON, QUANTILE_LEVELS, 0)

    assert batch.shape == (3, HORIZON, len(QUANTILE_LEVELS)) and status == ["ok"] * 3
    assert np.isfinite(batch).all()
    np.testing.assert_array_equal(again, batch)
    print(f"MAX |ALONE - BATCH| = {np.abs(alone[0] - batch[1]).max():.3g}")
    np.testing.assert_allclose(alone[0], batch[1], rtol=1e-5, atol=1e-4)
    assert params["cross_learning"] is False and params["revision"] == REVISION and params["point_only"] is False


def test_all_zero_context_gives_finite_forecasts_near_zero(chronos2):
    forecaster, _ = chronos2
    closed = pd.Series(0.0, index=pd.date_range("2026-01-01", periods=96, freq="15min"))

    quantiles, _ = forecaster([closed], HORIZON, QUANTILE_LEVELS, 0)

    assert np.isfinite(quantiles).all() and np.abs(quantiles).max() < 1


def test_rp_temperature_fixture_smoke(chronos2):
    forecaster, _ = chronos2
    series = load_sensor_series_from_json(str(FIXTURE), column_name="temperature")

    quantiles, status = forecaster([series.iloc[-7 * 96:]], HORIZON, QUANTILE_LEVELS, 0)

    assert quantiles.shape == (1, HORIZON, len(QUANTILE_LEVELS)) and status == ["ok"]
    assert np.isfinite(quantiles).all()
    assert (np.diff(quantiles, axis=-1) >= 0).all()


def test_equals_autogluon_chronos2_without_cross_learning(chronos2, tmp_path, make_contexts, autogluon_deciles):
    forecaster, _ = chronos2
    contexts = make_contexts(3, 7)

    reference = autogluon_deciles("Chronos2", {"model_path": MODEL_ID, "revision": REVISION, "device": "cpu",
                                               "cross_learning": False, "context_length": None, "batch_size": 256},
                                  contexts, tmp_path / "predictor")
    ours, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)

    print(f"MAX |OURS - AUTOGLUON| = {np.abs(ours - reference).max():.3g}")
    np.testing.assert_allclose(ours, reference, rtol=1e-5, atol=1e-4)


# Covariate windows for the contexts: a known column "k" (a daily on/off pulse that the model can use) and a past-only
# column "p", built by covariates.covariate_window like in a run.
def _windows(contexts: list[pd.Series]) -> list[pd.DataFrame]:
    index = pd.date_range("2026-01-01", "2026-01-12", freq="15min")
    frame = pd.DataFrame({"k": ((index.hour >= 8) & (index.hour < 16)).astype(float),
                          "p": np.random.default_rng(0).normal(size=len(index))}, index=index)
    kinds = {"k": "known", "p": "past"}
    return [cov.covariate_window(frame, kinds, c.index[-1] + pd.Timedelta("15min"), c.index) for c in contexts]


def test_covariate_forecasts_are_batch_independent_and_deterministic(chronos2, make_contexts):
    forecaster, _ = chronos2
    contexts = make_contexts(3, 7)
    windows = _windows(contexts)

    batch, status = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0, covariates=windows)
    again, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0, covariates=windows)
    alone, _ = forecaster([contexts[1]], HORIZON, QUANTILE_LEVELS, 0, covariates=[windows[1]])

    assert batch.shape == (3, HORIZON, len(QUANTILE_LEVELS)) and status == ["ok"] * 3
    assert np.isfinite(batch).all()
    np.testing.assert_array_equal(again, batch)
    print(f"MAX |ALONE - BATCH| WITH COVARIATES = {np.abs(alone[0] - batch[1]).max():.3g}")
    np.testing.assert_allclose(alone[0], batch[1], rtol=1e-5, atol=1e-4)


def test_future_covariates_are_used_and_a_past_only_future_is_not(chronos2, make_contexts):
    forecaster, _ = chronos2
    contexts = make_contexts(2, 7)
    windows = _windows(contexts)
    shifted, leaked = [], []
    for c, w in zip(contexts, windows):
        n = len(c)
        k = w.copy()
        k.iloc[n:, k.columns.get_loc("k")] = 1 - k.iloc[n:, k.columns.get_loc("k")]  # a different known future
        k.attrs = dict(w.attrs)
        p = w.copy()
        p.iloc[n:, p.columns.get_loc("p")] = 100.0  # a past-only future that the runner would have blanked
        p.attrs = dict(w.attrs)
        shifted.append(k)
        leaked.append(p)

    base, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0, covariates=windows)
    other, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0, covariates=shifted)
    ignored, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0, covariates=leaked)

    print(f"MAX |BASE - OTHER KNOWN FUTURE| = {np.abs(base - other).max():.3g}")
    assert np.abs(base - other).max() > 1e-3
    np.testing.assert_array_equal(ignored, base)
