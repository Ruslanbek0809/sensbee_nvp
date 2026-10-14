# LLMTime (the research project's method, src/models/nvp_llms.py) with a local Hugging Face model as a zero-shot
# harness forecaster (contract: benchmark/runner.py header).
#
# Each context goes through the service pipeline (scale, serialize, prompt, complete with token control, parse and
# filter: llmtime_samples), and the forecast is the median of the valid samples mapped back to the original scale, as
# in nvp_llms_forecast. Two service steps are changed for the benchmark: an origin whose samples are all filtered out
# gets the last observed value with status "fallback" (the service does this silently), and visitor forecasts are not
# clipped at 0 (D-B3). A handful of samples doesn't make honest deciles, so the forecaster is point-only: every
# quantile is the median. Origins run one at a time, each after torch.manual_seed(seed), so a forecast doesn't depend
# on its position in the batch. An origin whose generation raises is reported "failed" and the others go on, so the
# runner never re-runs a whole batch of long generations. Completions, parse failures and samples removed per filter
# rule are counted and drained into run.json per sensor (drain_diagnostics).
# Local generation: the pinned revision in float16, one sample at a time (batch_size=1: five copies of a 14-day
# prompt at once wouldn't fit next to the weights on an 18 GB Mac), top_p 0.9, always on one device (multi_gpu=False:
# the multi-GPU path splits samples across devices by free memory and ignores batch_size). Building the forecaster
# fails without a GPU backend (CUDA or MPS), so a 7B model never crawls on the CPU; the backend goes into run.json.

import logging
from collections import Counter
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TOP_P = 0.9


# Builds an LLMTime forecaster for one model id and revision, and returns it with the effective settings. complete:
# the completion function (called like nvp_llms.call_llm_completion); by default local generation with the model.
def llmtime_forecaster(model_id: str, revision: str, num_forecasts: int = 5, temperature: float = 0.9,
                       use_normalization: bool = True, dtype: str = "float16",
                       complete: Optional[Callable] = None):
    diagnostics: Counter = Counter()
    backend = None
    if complete is None:
        from src.models.local_llm import get_device_backend
        backend = get_device_backend()
        if backend == "cpu":
            raise RuntimeError("LLMTIME NEEDS A GPU BACKEND (CUDA OR MPS); NONE IS AVAILABLE")

    # Local generation at the pinned revision, one sample at a time.
    def generate(prompt: str, settings, num_forecasts: int, temperature: float, steps: int, **_) -> list[str]:
        from src.models.local_llm import local_llm_completion
        return local_llm_completion(input_str=prompt, settings=settings, model_name=model_id, steps=steps,
                                    num_forecasts=num_forecasts, batch_size=1, temperature=temperature, top_p=TOP_P,
                                    dtype=dtype, revision=revision, multi_gpu=False)

    def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...],
                 seed: int) -> tuple[np.ndarray, list[str]]:
        import torch
        from src.models.nvp_llms import aggregate_samples, llmtime_samples  # loads .env on import: only when used
        out = np.full((len(contexts), horizon, len(levels)), np.nan)
        status = []
        for i, context in enumerate(contexts):
            torch.manual_seed(seed)
            try:
                result = llmtime_samples(context, horizon, provider="local", model=model_id,
                                         num_forecasts=num_forecasts, temperature=temperature,
                                         use_normalization=use_normalization, complete=complete or generate)
            except Exception as exc:
                logger.warning(f"LLMTIME ORIGIN {i} FAILED ({type(exc).__name__}: {exc})")
                diagnostics["failed"] += 1
                status.append("failed")
                continue
            diagnostics.update(result.counts)
            if result.samples:
                point = aggregate_samples(result.samples, method="median")
                point = result.scaler.inverse_transform(point) if use_normalization else point
                status.append("ok")
            else:
                point = np.full(horizon, result.last_value)
                status.append("fallback")
            out[i] = np.repeat(np.asarray(point, dtype=float)[:, None], len(levels), axis=1)
        return out, status

    # Returns the counts since the last call and resets them.
    def drain_diagnostics() -> dict:
        counts = dict(diagnostics)
        diagnostics.clear()
        return counts

    forecast.drain_diagnostics = drain_diagnostics
    return forecast, {"library": "transformers", "method": "LLMTime (src/models/nvp_llms.py)", "model_id": model_id,
                      "revision": revision, "dtype": dtype, "device_backend": backend, "multi_gpu": False,
                      "num_forecasts": num_forecasts,
                      "temperature": temperature, "top_p": TOP_P, "use_normalization": use_normalization,
                      "alpha": 0.95, "beta": 0.3, "batch_size": 1, "aggregation": "median", "clipping": None,
                      "fallback": "last value, status fallback", "point_only": True}
