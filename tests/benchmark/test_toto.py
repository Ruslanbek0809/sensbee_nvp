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

from benchmark.models import ZERO_SHOT  # noqa: E402
from benchmark.models.toto_models import CONTEXT_LENGTH, load_toto, masked_batch  # noqa: E402
from benchmark.tasks import HORIZON, QUANTILE_LEVELS  # noqa: E402

BARE: dict = {}


# The bare TotoForecaster (for the mean path), loaded once; the zero_shot fixture skips the test when the weights
# aren't cached.
def _toto(zero_shot):
    zero_shot("toto")
    if not BARE:
        BARE["toto"] = load_toto(*ZERO_SHOT["toto"].weights[:2])
    return BARE["toto"]


# Toto's mean forecast (no sampling) of each context.
def _means(toto, contexts: list[pd.Series]) -> np.ndarray:
    with torch.inference_mode():
        return toto.forecast(masked_batch(contexts), HORIZON, num_samples=None).mean[:, 0].numpy()


@pytest.mark.integration
def test_mean_forecast_alone_equals_it_in_a_batch(make_contexts, zero_shot):
    toto = _toto(zero_shot)
    contexts = make_contexts(3, 7)

    batch = _means(toto, contexts)
    alone = _means(toto, [contexts[1]])

    print(f"toto: MAX |ALONE - BATCH| (MEAN PATH) = {np.abs(alone[0] - batch[1]).max():.3g}")
    np.testing.assert_allclose(alone[0], batch[1], rtol=1e-5, atol=1e-4)


# AutoGluon's loader leaves the model in training mode (dropout 0.1 active); load_toto switches every module off.
@pytest.mark.integration
def test_model_is_in_eval_mode(zero_shot):
    assert not any(module.training for module in _toto(zero_shot).model.modules())


# Contexts longer than 4,096 steps keep their last 4,096, every step observed.
def test_masked_batch_cuts_long_contexts():
    index = pd.date_range("2026-01-01", periods=CONTEXT_LENGTH + 10, freq="15min")
    values = np.arange(len(index), dtype=float)

    batch = masked_batch([pd.Series(values, index=index)] * 2)

    assert batch.series.shape == (2, 1, CONTEXT_LENGTH) and bool(batch.padding_mask.all())
    assert batch.series[0, 0, 0].item() == 10.0 and batch.series[1, 0, -1].item() == values[-1]
    assert batch.time_interval_seconds.tolist() == [[900], [900]]


@pytest.mark.integration
def test_toto_equals_autogluon_toto(tmp_path, make_contexts, autogluon_deciles, zero_shot):
    from autogluon.timeseries.models.toto._internal import TotoForecaster
    from autogluon.timeseries.models.toto.model import TotoModel
    from huggingface_hub import try_to_load_from_cache
    forecaster, params = zero_shot("toto")
    model_id, revision, filename = ZERO_SHOT["toto"].weights
    contexts = make_contexts(3, 7)
    forecast = TotoForecaster.forecast

    # AutoGluon leaves the model in training mode and draws random numbers before it forecasts, so eval mode and the
    # seed are set right at Toto's forecast call.
    def seeded(self, *args, **kwargs):
        self.model.eval()
        torch.manual_seed(0)
        return forecast(self, *args, **kwargs)

    with patch.object(TotoModel, "_is_gpu_available", autospec=True, return_value=True), \
            patch.object(TotoForecaster, "forecast", seeded):
        reference = autogluon_deciles("Toto", {
            "model_path": os.path.dirname(try_to_load_from_cache(model_id, filename, revision=revision)),
            "device": "cpu", "batch_size": params["batch_size"], "num_samples": params["num_samples"],
            "context_length": CONTEXT_LENGTH, "compile_model": False}, contexts, tmp_path / "predictor")
    ours, _ = forecaster(contexts, HORIZON, QUANTILE_LEVELS, 0)

    print(f"toto: MAX |OURS - AUTOGLUON| = {np.abs(ours - reference).max():.3g}")
    np.testing.assert_allclose(ours, reference, rtol=1e-5, atol=1e-4)
