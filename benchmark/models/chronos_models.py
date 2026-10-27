# Chronos-2 (chronos-forecasting's Chronos2Pipeline) and Chronos-Bolt (ChronosBoltPipeline) as zero-shot harness
# forecasters (contract: benchmark/runner.py header).
#
# The model is loaded once per build, at a pinned Hugging Face revision, in float32.
# Every context is forecast on its own. cross_learning=False gives each series its own group id, so series in a batch
# never attend to each other (chronos/chronos2/dataset.py, v2.3.2).
# With cross-learning on (AutoGluon's Chronos2 default), one harness call holds all origins of a sensor.
# The context of a later origin contains the target of an earlier one, so a forecast could see its own future.
# context_length is the model maximum (8192). The harness cuts the context, so the model never truncates it.
# The 9 deciles are among the model's trained quantiles and are read out without interpolation.
# Chronos-2 has no sampling, so the seed is unused (the runner records it).
# Covariates (Phase 4): with covariate windows from the runner, each context goes in as a dict.
# - "past_covariates": every covariate column over the context.
# - "future_covariates": the window's future columns (known and oracle; attrs["future_columns"]) over the horizon.
# Past-only columns get no future key, so one call has one schema.
# cross_learning=False still gives each dict (its target and covariates) its own group, so batch items stay independent.
# Checked 2026-10-14 (benchmark_data/spikes/*_phase4_4.1_gates/):
# - passing the other stations as past covariates gives the target row exactly the forecast of the joint 3-variate
#   target;
# - a past-only column passed as an all-NaN future covariate differs by up to 1.1e-5.
# That is why the schema follows the window's kinds and not its NaNs.
#
# Chronos-Bolt is univariate, so series in a batch are independent by construction.
# It reads at most 2,048 steps (21.3 days at 15 min) and cuts longer contexts itself.
# So its "28-day" forecasts use the last 2,048 steps.
# It forecasts 64 steps in one pass. Steps 65-96 come from the library's heuristic (chronos_bolt.py, v2.3.2).
# The heuristic feeds each of the 9 quantile paths back as context and takes the quantiles of the 81 resulting values.
# Its 9 trained quantiles are the deciles, read out without interpolation.
# It is deterministic, so the seed is unused too.

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# One Chronos-2 dict input: the context as target, every covariate column over the context as past covariates, and
# the window's future columns over the horizon as future covariates.
def _covariate_input(context: pd.Series, window: pd.DataFrame, horizon: int) -> dict:
    n = len(context)
    if len(window) != n + horizon or not window.index[:n].equals(context.index):
        raise ValueError("COVARIATE WINDOW DOES NOT MATCH ITS CONTEXT")
    return {"target": context.to_numpy(dtype=np.float32),
            "past_covariates": {c: window[c].iloc[:n].to_numpy(dtype=np.float32) for c in window.columns},
            "future_covariates": {c: window[c].iloc[n:].to_numpy(dtype=np.float32)
                                  for c in window.attrs["future_columns"]}}


# The forecaster of a loaded Chronos-2 pipeline, zero-shot or fine-tuned (benchmark/models/finetune_models.py).
def chronos2_pipeline_forecaster(pipeline, batch_size: int = 256):
    context_length = pipeline.model_context_length

    def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...], seed: int,
                 covariates: Optional[list[pd.DataFrame]] = None) -> tuple[np.ndarray, list[str]]:
        if covariates is None:
            inputs = [c.to_numpy(dtype=np.float32) for c in contexts]
        else:
            inputs = [_covariate_input(c, w, horizon) for c, w in zip(contexts, covariates)]
        quantiles, _ = pipeline.predict_quantiles(inputs, prediction_length=horizon, quantile_levels=list(levels),
                                                  batch_size=batch_size, context_length=context_length,
                                                  cross_learning=False, limit_prediction_length=True)
        return np.stack([q[0].numpy() for q in quantiles]), ["ok"] * len(contexts)

    return forecast


# Builds a Chronos-2 forecaster for one model id and revision, and returns it with the effective settings.
def chronos2_forecaster(model_id: str, revision: str, device: str = "cpu", batch_size: int = 256):
    import torch
    from chronos.chronos2.pipeline import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(model_id, revision=revision, device_map=device, dtype=torch.float32)
    context_length = pipeline.model_context_length
    forecast = chronos2_pipeline_forecaster(pipeline, batch_size)

    return forecast, {"library": "chronos-forecasting", "model_id": model_id, "revision": revision, "device": device,
                      "dtype": "float32", "batch_size": batch_size, "context_length": context_length,
                      "cross_learning": False, "torch_threads": torch.get_num_threads(), "point_only": False}


# Builds a Chronos-Bolt forecaster for one model id and revision, and returns it with the effective settings.
# The pipeline has no batching of its own, and steps 65-96 multiply the batch by 9.
# So contexts go in chunks of batch_size.
def chronos_bolt_forecaster(model_id: str, revision: str, device: str = "cpu", batch_size: int = 64):
    import torch
    from chronos.chronos_bolt import ChronosBoltPipeline
    pipeline = ChronosBoltPipeline.from_pretrained(model_id, revision=revision, device_map=device, dtype=torch.float32)

    def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...],
                 seed: int) -> tuple[np.ndarray, list[str]]:
        out = []
        for start in range(0, len(contexts), batch_size):
            chunk = [torch.tensor(c.to_numpy(dtype=np.float32)) for c in contexts[start:start + batch_size]]
            quantiles, _ = pipeline.predict_quantiles(chunk, prediction_length=horizon, quantile_levels=list(levels),
                                                      limit_prediction_length=False)
            out.append(quantiles.numpy())
        return np.concatenate(out), ["ok"] * len(contexts)

    return forecast, {"library": "chronos-forecasting", "model_id": model_id, "revision": revision, "device": device,
                      "dtype": "float32", "batch_size": batch_size,
                      "model_context_length": pipeline.model_context_length,
                      "native_horizon": pipeline.model_prediction_length,
                      "beyond_native_horizon": "library quantile-expansion heuristic",
                      "torch_threads": torch.get_num_threads(), "point_only": False}
