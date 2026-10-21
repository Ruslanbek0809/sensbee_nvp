# TiRex (tirex-ts 1.4.2, an xLSTM forecaster) as a zero-shot harness forecaster (contract: benchmark/runner.py header).
#
# The model is loaded once per build at a pinned Hugging Face revision.
# It uses the plain torch backend (the CUDA kernels need an NVIDIA GPU) and no torch.compile.
# It reads at most train_ctx_len = 2,048 steps (21.3 days at 15 min) and cuts longer contexts itself.
# So its "28-day" forecasts use the last 2,048 steps. Shorter contexts are left-padded with masked NaNs.
# Steps beyond the 32-step output patch are produced by feeding the future patches in as missing values, not by
# feeding predictions back.
# Frequency resampling (resample_strategy) stays off.
# The 9 trained quantiles are the deciles, read out by level.
# Series are forecast independently and the model is deterministic, so the seed is unused.
# Each call runs on one torch thread (restored afterwards).
# With several threads, the batched rollout after the first patch differs from a single-series call by up to 0.2%.
# That is float arithmetic, not information from other series (swapping them changes nothing).
# One thread gives identical forecasts alone and in a batch, at about the same speed.
# Weights: NXAI Community Licence (not Apache-2.0).

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Builds a TiRex forecaster for one model id and revision, and returns it with the effective settings.
def tirex_forecaster(model_id: str, revision: str, device: str = "cpu", batch_size: int = 64):
    import torch
    from tirex import load_model
    model = load_model(model_id, device=device, backend="torch", compile=False, hf_kwargs={"revision": revision})
    model.eval()
    trained = list(model.config.quantiles)

    def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...],
                 seed: int) -> tuple[np.ndarray, list[str]]:
        columns = [trained.index(level) for level in levels]  # a level the model wasn't trained on raises
        inputs = np.stack([c.to_numpy(dtype=np.float32) for c in contexts])
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            quantiles, _ = model.forecast(context=inputs, prediction_length=horizon, output_type="numpy",
                                          batch_size=batch_size, resample_strategy=None)
        finally:
            torch.set_num_threads(threads)
        return quantiles[:, :horizon, columns], ["ok"] * len(contexts)

    return forecast, {"library": "tirex-ts", "model_id": model_id, "revision": revision, "device": device,
                      "dtype": "float32", "backend": "torch", "batch_size": batch_size,
                      "model_context_length": model.config.train_ctx_len, "patch": model.config.output_patch_size,
                      "resample_strategy": None, "max_accelerated_rollout_steps": 1,
                      "licence": "NXAI Community Licence", "torch_threads": 1,
                      "point_only": False}
