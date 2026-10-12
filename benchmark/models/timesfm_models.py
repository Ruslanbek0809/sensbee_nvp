# TimesFM 2.5 (timesfm 3.0.2, TimesFM_2p5_200M_torch) as a zero-shot harness forecaster (contract:
# benchmark/runner.py header).
#
# The model is loaded once per build at a pinned Hugging Face revision, without torch.compile (CPU). Every call is
# compiled for its own context length rounded up to the 32-step patch, so the harness's equal-length contexts are
# never padded; context + the 128-step output block must stay within 16,384 steps, i.e. at most 16,256 context steps
# (169.3 days at 15 min). Settings are the model card's, except two:
#   - infer_is_positive=False: the default True clamps every output at 0 when the whole context is >= 0, which would
#     clip visitor-count forecasts (timesfm_2p5_torch.py, v3.0.2); clipping is never part of the main tables;
#   - fix_quantile_crossing=False: the runner counts and sorts crossing quantiles the same way for every model.
# The output holds the mean and then the 0.1..0.9 quantiles; the 9 deciles are read out directly. Series are
# forecast independently (no covariates, no cross-series terms), and the model is deterministic, so the seed is unused.

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DECILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# Builds a TimesFM 2.5 forecaster for one model id and revision, and returns it with the effective settings.
def timesfm25_forecaster(model_id: str, revision: str, batch_size: int = 64):
    import timesfm
    import torch
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id, revision=revision, torch_compile=False)
    patch, block, limit = model.model.p, model.model.o, model.model.config.context_limit
    settings = {"normalize_inputs": True, "use_continuous_quantile_head": True, "force_flip_invariance": True,
                "infer_is_positive": False, "fix_quantile_crossing": False}

    def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...],
                 seed: int) -> tuple[np.ndarray, list[str]]:
        if tuple(levels) != DECILES:
            raise ValueError(f"TIMESFM GIVES ONLY THE DECILES, NOT {levels}")
        steps = max(len(c) for c in contexts)
        model.compile(timesfm.ForecastConfig(max_context=math.ceil(steps / patch) * patch,
                                             max_horizon=math.ceil(horizon / block) * block,
                                             per_core_batch_size=batch_size, **settings))
        _, quantiles = model.forecast(horizon=horizon, inputs=[c.to_numpy(dtype=np.float32) for c in contexts])
        return quantiles[:, :horizon, 1:], ["ok"] * len(contexts)

    return forecast, {"library": "timesfm", "model_id": model_id, "revision": revision,
                      "device": str(model.model.device), "dtype": "float32", "batch_size": batch_size,
                      "torch_compile": False, **settings,
                      "model_context_length": limit - block, "patch": patch, "output_block": block,
                      "torch_threads": torch.get_num_threads(), "point_only": False}
