# "0 when closed" baselines for the halls (Phase 4, RQ2): a values-only base forecaster whose quantiles are all set to
# 0 at every future step where the hall's learned opening profile (covariate "open_share", benchmark/covariates.py:
# the share of days with visitors in that Berlin weekday and 15-min slot before the cutoff) is below a threshold. The
# profile is known in advance, so the rule uses no future observation. These are separate named models, never a
# post-processing of another model's results (experiments rule, D-B3); the threshold is chosen per hall on the
# validation period (plan section D). The base sees only its context, as in its own runs.

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Wraps a base builder (→ forecaster, parameters) into a masked forecaster that needs the covariate windows of a set
# with "open_share", and returns it with its parameters (the base's, its point_only, the threshold, and base_model: the
# registry name of the base, so the report can test the masked run against the base's own run).
def mask_forecaster(base_build: Callable, threshold: float, base_model: str):
    base, base_params = base_build()

    def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...], seed: int,
                 covariates: list[pd.DataFrame]) -> tuple[np.ndarray, list[str]]:
        quantiles, status = base(contexts, horizon, levels, seed)
        quantiles = np.array(quantiles, dtype=float)
        for i, (context, window) in enumerate(zip(contexts, covariates)):
            closed = window["open_share"].iloc[len(context):].to_numpy() < threshold
            quantiles[i, closed, :] = 0.0
        return quantiles, list(status)

    return forecast, {"base": base_params, "rule": "every quantile = 0 where open_share < threshold",
                      "threshold": threshold, "point_only": bool(base_params.get("point_only", False)),
                      "base_model": base_model}
