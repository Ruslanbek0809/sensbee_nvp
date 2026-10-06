# Chronos-2 (chronos-forecasting's Chronos2Pipeline) as a zero-shot harness forecaster (contract: benchmark/runner.py
# header).
#
# The model is loaded once per build, at a pinned Hugging Face revision, in float32. Every context is forecast on its
# own: cross_learning=False gives each series its own group id, so series in a batch never attend to each other
# (chronos/chronos2/dataset.py, v2.3.2). With cross-learning on (AutoGluon's Chronos2 default), one harness call holds
# all origins of a sensor, and the context of a later origin contains the target of an earlier one, so a forecast
# could see its own future. context_length is the model maximum (8192): the harness cuts the context, so the model
# never truncates it. The 9 deciles are among the model's trained quantiles and are read out without interpolation.
# Chronos-2 has no sampling, so the seed is unused (the runner records it).

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Builds a Chronos-2 forecaster for one model id and revision, and returns it with the effective settings.
def chronos2_forecaster(model_id: str, revision: str, device: str = "cpu", batch_size: int = 256):
    import torch
    from chronos.chronos2.pipeline import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(model_id, revision=revision, device_map=device, dtype=torch.float32)
    context_length = pipeline.model_context_length

    def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...],
                 seed: int) -> tuple[np.ndarray, list[str]]:
        inputs = [c.to_numpy(dtype=np.float32) for c in contexts]
        quantiles, _ = pipeline.predict_quantiles(inputs, prediction_length=horizon, quantile_levels=list(levels),
                                                  batch_size=batch_size, context_length=context_length,
                                                  cross_learning=False, limit_prediction_length=True)
        return np.stack([q[0].numpy() for q in quantiles]), ["ok"] * len(contexts)

    return forecast, {"library": "chronos-forecasting", "model_id": model_id, "revision": revision, "device": device,
                      "dtype": "float32", "batch_size": batch_size, "context_length": context_length,
                      "cross_learning": False, "torch_threads": torch.get_num_threads(), "point_only": False}
