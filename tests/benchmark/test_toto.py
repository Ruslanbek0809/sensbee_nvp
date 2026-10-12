# Tests specific to the Toto adapter (benchmark/models/toto_models.py); the generic zero-shot checks are in
# test_zero_shot.py. Toto samples, and its random draws depend on the batch layout, so batch independence is checked
# on its deterministic mean path here. The cross-check runs AutoGluon's own TotoModel on the CPU (its GPU check is
# patched, test only) in eval mode with the same seed at the forecast call. The input-building test needs no weights
# and runs in the default bench suite; the others need the cached weights (never downloaded) and run with
#   HF_HUB_OFFLINE=1 venv-bench/bin/python -m pytest tests/benchmark/test_toto.py -m integration -q -s

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("autogluon.timeseries")

from huggingface_hub import try_to_load_from_cache  # noqa: E402

from benchmark.models import ZERO_SHOT  # noqa: E402
from benchmark.models.toto_models import CONTEXT_LENGTH, load_toto, masked_batch  # noqa: E402
from benchmark.tasks import HORIZON, QUANTILE_LEVELS  # noqa: E402

BUILT: dict = {}


# The cached weight file of the pinned Toto revision; skips the test when it isn't in the local cache.
def _weights() -> str:
    model_id, revision, filename = ZERO_SHOT["toto"].weights
    path = try_to_load_from_cache(model_id, filename, revision=revision)
    if not isinstance(path, str):
        pytest.skip(f"{model_id} at {revision} is not in the local Hugging Face cache")
    return path


# The harness forecaster and the bare TotoForecaster, built once per test session.
def _built() -> tuple:
    _weights()
    if not BUILT:
        model_id, revision, _ = ZERO_SHOT["toto"].weights
        BUILT["forecaster"] = ZERO_SHOT["toto"].build()
        BUILT["toto"] = load_toto(model_id, revision)
    return BUILT["forecaster"], BUILT["toto"]


# Toto's mean forecast (no sampling) of each context.
def _means(toto, contexts: list[pd.Series]) -> np.ndarray:
    with torch.inference_mode():
        return toto.forecast(masked_batch(contexts), HORIZON, num_samples=None).mean[:, 0].numpy()


@pytest.mark.integration
def test_mean_forecast_alone_equals_it_in_a_batch(make_contexts):
    _, toto = _built()
    contexts = make_contexts(3, 7)

    batch = _means(toto, contexts)
    alone = _means(toto, [contexts[1]])

    print(f"toto: MAX |ALONE - BATCH| (MEAN PATH) = {np.abs(alone[0] - batch[1]).max():.3g}")
    np.testing.assert_allclose(alone[0], batch[1], rtol=1e-5, atol=1e-4)


# Left padding changes Toto's forecast (a 2-day context next to a 7-day one: up to 5.4 off on a ~10 series, 2026-10-12),
# so the forecaster refuses contexts of different lengths; the runner then retries those origins one by one.
@pytest.mark.integration
def test_forecaster_refuses_contexts_of_different_lengths(make_contexts):
    (forecaster, _), _ = _built()
    long, short = make_contexts(1, 7)[0], make_contexts(2, 2)[1]

    with pytest.raises(ValueError, match="SAME LENGTH"):
        forecaster([long, short], HORIZON, QUANTILE_LEVELS, 0)


# AutoGluon's loader leaves the model in training mode (dropout 0.1 active); the adapter switches every module off.
@pytest.mark.integration
def test_model_is_in_eval_mode():
    forecaster, toto = _built()

    assert not any(module.training for module in toto.model.modules())
    assert forecaster[1]["eval_mode"] is True and forecaster[1]["num_samples"] == 256


# Gaps are forward- then backward-filled, contexts longer than 4,096 steps keep their last 4,096.
def test_masked_batch_fills_gaps_and_cuts_long_contexts():
    index = pd.date_range("2026-01-01", periods=CONTEXT_LENGTH + 10, freq="15min")
    values = np.arange(len(index), dtype=float)
    values[[-3, -2]] = np.nan
    gappy = pd.Series([np.nan, 5.0, np.nan, 7.0], index=index[:4])

    long_batch = masked_batch([pd.Series(values, index=index)])
    gappy_batch = masked_batch([gappy])

    assert long_batch.series.shape == (1, 1, CONTEXT_LENGTH) and bool(long_batch.padding_mask.all())
    assert long_batch.series[0, 0, 0].item() == 10.0 and long_batch.series[0, 0, -2].item() == values[-4]
    assert gappy_batch.series[0, 0].tolist() == [5.0, 5.0, 5.0, 7.0]


@pytest.mark.integration
def test_toto_equals_autogluon_toto(tmp_path, make_contexts):
    from autogluon.timeseries.models.toto._internal import TotoForecaster
    from autogluon.timeseries.models.toto.model import TotoModel

    from benchmark.models.autogluon_models import _batch_frame
    from benchmark.snapshot import BUCKET_FREQ
    (forecaster, _), _ = _built()
    contexts = make_contexts(3, 7)
    data = _batch_frame(contexts)
    model = TotoModel(path=str(tmp_path), freq=BUCKET_FREQ, prediction_length=HORIZON,
                      quantile_levels=list(QUANTILE_LEVELS),
                      hyperparameters={"model_path": os.path.dirname(_weights()), "device": "cpu",
                                       "batch_size": 24, "num_samples": 256, "context_length": CONTEXT_LENGTH,
                                       "compile_model": False})
    with patch.object(TotoModel, "_is_gpu_available", autospec=True, return_value=True):
        model.fit(train_data=data)
    model._forecaster.model.eval()
    forecast = TotoForecaster.forecast

    # AutoGluon's predict draws random numbers before it forecasts, so the seed is set right at Toto's forecast call.
    def seeded(self, *args, **kwargs):
        torch.manual_seed(0)
        return forecast(self, *args, **kwargs)

    with patch.object(TotoForecaster, "forecast", seeded):
        predictions = model.predict(data)
    reference = np.stack([predictions.loc[f"{i:06d}", [str(q) for q in QUANTILE_LEVELS]].to_numpy(dtype=float)
                          for i in range(len(contexts))])

    ours, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)

    print(f"toto: MAX |OURS - AUTOGLUON| = {np.abs(ours - reference).max():.3g}")
    np.testing.assert_allclose(ours, reference, rtol=1e-5, atol=1e-4)
