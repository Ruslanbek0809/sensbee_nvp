# Integration tests for the Chronos-2 forecaster in benchmark/models: output shape, batch independence (an origin
# forecast alone equals the same origin inside a batch), determinism, an all-zero context (closed facility), a smoke run
# on the RP temperature fixture, and a cross-check against AutoGluon's Chronos2 wrapper with cross-learning off. They
# need the model weights in the local Hugging Face cache and never download them: run in the bench venv with
#   HF_HUB_OFFLINE=1 venv-bench/bin/python -m pytest tests/benchmark/test_chronos.py -m integration -q

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("chronos")

from huggingface_hub import try_to_load_from_cache  # noqa: E402

from benchmark.models import CHRONOS2, ZERO_SHOT  # noqa: E402
from benchmark.tasks import HORIZON, QUANTILE_LEVELS  # noqa: E402
from src.data_access.data_loader import load_sensor_series_from_json  # noqa: E402

MODEL_ID, REVISION = CHRONOS2
if not isinstance(try_to_load_from_cache(MODEL_ID, "model.safetensors", revision=REVISION), str):
    pytest.skip(f"{MODEL_ID} at {REVISION} is not in the local Hugging Face cache", allow_module_level=True)

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "temp_14day_sensbee_data.json"
PATTERN = 10 + 5 * np.sin(np.arange(96) * 2 * np.pi / 96)


# n daily-periodic contexts of `days` days with a little noise, each with its own noise and end time.
def _contexts(n: int, days: int, noise: float = 0.3) -> list[pd.Series]:
    out = []
    for i in range(n):
        rng = np.random.default_rng(i)
        index = pd.date_range("2026-01-01", periods=days * 96, freq="15min") + pd.Timedelta(hours=6 * i)
        values = np.tile(np.roll(PATTERN, -24 * i), days) + rng.normal(0, noise, days * 96)
        out.append(pd.Series(values, index=index))
    return out


# The forecaster is built once: loading the weights takes seconds.
@pytest.fixture(scope="module")
def chronos2():
    return ZERO_SHOT["chronos2"].build()


def test_shape_batch_independence_and_determinism(chronos2):
    forecaster, params = chronos2
    contexts = _contexts(3, 7)

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


def test_equals_autogluon_chronos2_without_cross_learning(chronos2, tmp_path):
    pytest.importorskip("autogluon.timeseries")
    from autogluon.timeseries import TimeSeriesPredictor

    from benchmark.models.autogluon_models import _batch_frame
    from benchmark.snapshot import BUCKET_FREQ
    forecaster, _ = chronos2
    contexts = _contexts(3, 7)
    data = _batch_frame(contexts)

    predictor = TimeSeriesPredictor(prediction_length=HORIZON, freq=BUCKET_FREQ, quantile_levels=list(QUANTILE_LEVELS),
                                    path=str(tmp_path / "predictor"), verbosity=0, log_to_file=False)
    predictor.fit(data, hyperparameters={"Chronos2": {
        "model_path": MODEL_ID, "revision": REVISION, "device": "cpu", "cross_learning": False,
        "context_length": None, "batch_size": 256}}, enable_ensemble=False, skip_model_selection=True)
    predictions = predictor.predict(data)
    reference = np.stack([predictions.loc[f"{i:06d}", [str(q) for q in QUANTILE_LEVELS]].to_numpy(dtype=float)
                          for i in range(len(contexts))])
    ours, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)

    print(f"MAX |OURS - AUTOGLUON| = {np.abs(ours - reference).max():.3g}")
    np.testing.assert_allclose(ours, reference, rtol=1e-5, atol=1e-4)
