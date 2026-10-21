# Toto-Open-Base-1.0 (Datadog) as a zero-shot harness forecaster (contract: benchmark/runner.py header).
# It runs with the copy of the Toto code that ships inside autogluon.timeseries 1.6.3 (models/toto/).
# That code is private API, pinned by the AutoGluon version.
# AutoGluon's own TotoModel refuses to run without CUDA, but the copied code is plain torch.
# So the adapter calls it directly on the CPU. The GPU check is the only part of AutoGluon's path that is bypassed.
#
# The model is loaded once per build at a pinned Hugging Face revision with strict weight loading.
# (The checkpoint and the model have exactly the same keys, 151,306,080 parameters; checked 2026-10-12.)
# It is put in eval mode. AutoGluon's loader leaves it in training mode, so its config's dropout of 0.1 would stay
# active while forecasting.
# Inputs are built the way AutoGluon's TotoDataLoader builds them for gap-free contexts of one length
# (the runner's contract; tasks.context_window fills gaps):
# - contexts cut to the last 4,096 steps (AutoGluon's default);
# - one variate per series, with its own id;
# - every step observed, a 15-min interval and zero timestamps.
# Left padding is not neutral for Toto. A 2-day context next to a 7-day one is forecast differently than alone
# (mean path, 2026-10-12). That is why a call never mixes lengths.
# Toto forecasts in 64-step patches and feeds its sampled patches back, so 96 steps take 2 patches.
# The 9 deciles are the empirical quantiles of 256 sample paths (drawn 32 at a time, as AutoGluon does).
# So the forecast depends on the seed, which is set at the start of every call.
# Series are independent (one variate each, no cross-series attention).
# But the random draws depend on the batch layout, so a forecast alone equals it in a batch only on the deterministic
# mean path.

import logging

import numpy as np
import pandas as pd

from benchmark.tasks import STEP

logger = logging.getLogger(__name__)

CONTEXT_LENGTH = 4096


# Loads the Toto backbone for one model id and revision on the CPU in eval mode and wraps it in AutoGluon's
# TotoForecaster.
def load_toto(model_id: str, revision: str, device: str = "cpu"):
    from autogluon.timeseries.models.toto._internal import TotoForecaster
    from autogluon.timeseries.models.toto.hf_pretrained_model import TotoConfig, TotoPretrainedModel
    pretrained = TotoPretrainedModel.from_pretrained(
        model_id=model_id, revision=revision, config=TotoConfig.from_pretrained(model_id, revision=revision),
        device_map=device, strict=True)
    pretrained.eval()
    return TotoForecaster(model=pretrained.model)


# One batch of gap-free contexts of one length as Toto's MaskedTimeseries [batch, 1 variate, steps].
def masked_batch(contexts: list[pd.Series], device: str = "cpu"):
    import torch
    from autogluon.timeseries.models.toto._internal import MaskedTimeseries
    series = torch.stack([torch.tensor(c.to_numpy(dtype=np.float32)[-CONTEXT_LENGTH:]) for c in contexts])
    series = series.unsqueeze(1).to(device)
    n, _, steps = series.shape
    return MaskedTimeseries(
        series, padding_mask=torch.ones_like(series, dtype=torch.bool),
        id_mask=torch.arange(n, dtype=torch.int, device=device)[:, None, None].repeat(1, 1, steps),
        timestamp_seconds=torch.zeros_like(series, dtype=torch.int),
        time_interval_seconds=torch.full((n, 1), int(STEP.total_seconds()), dtype=torch.int, device=device))


# Builds a Toto forecaster for one model id and revision, and returns it with the effective settings.
def toto_forecaster(model_id: str, revision: str, device: str = "cpu", batch_size: int = 24, num_samples: int = 256,
                    samples_per_batch: int = 32):
    import torch
    toto = load_toto(model_id, revision, device)

    def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...],
                 seed: int) -> tuple[np.ndarray, list[str]]:
        torch.manual_seed(seed)
        out = []
        with torch.inference_mode():
            for start in range(0, len(contexts), batch_size):
                result = toto.forecast(masked_batch(contexts[start:start + batch_size], device), horizon,
                                       num_samples=num_samples, samples_per_batch=samples_per_batch)
                out.append(np.stack([result.quantile(level)[:, 0].cpu().numpy() for level in levels], axis=-1))
        return np.concatenate(out), ["ok"] * len(contexts)

    return forecast, {"library": "autogluon.timeseries (copied Toto code)", "model_id": model_id,
                      "revision": revision, "device": device, "dtype": "float32", "batch_size": batch_size,
                      "num_samples": num_samples, "samples_per_batch": samples_per_batch,
                      "model_context_length": CONTEXT_LENGTH, "patch": toto.model.patch_embed.patch_size,
                      "licence": "Apache-2.0", "torch_threads": torch.get_num_threads(), "point_only": False}
