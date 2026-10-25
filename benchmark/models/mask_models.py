# "0 when closed" baselines for the halls (Phase 4, RQ2).
# A values-only base forecaster has all its quantiles set to 0 at every future step where the hall's learned opening
# profile is below a threshold.
# The profile is the covariate "open_share" (benchmark/covariates.py): the share of days with visitors in that Berlin
# weekday and 15-min slot before the cutoff.
# The profile is known in advance, so the rule uses no future observation.
# These are separate named models, never a post-processing of another model's results (experiments rule, D-B3).
# The threshold is chosen per hall on the validation period (plan section D).
# The base sees only its context, as in its own runs.
# Phase 5 (D5-2): recent_mask_forecaster() takes the profile from each context instead
# (the last 28 days before the origin, refitted at every origin).
# This is the rule the service applies, because it doesn't know when a season starts. It needs no covariate set.

import logging
from typing import Callable

import numpy as np
import pandas as pd

from benchmark.covariates import open_share, series_open_profile
from benchmark.tasks import STEP

logger = logging.getLogger(__name__)


# Wraps a base builder (→ forecaster, parameters) into a masked forecaster that needs the covariate windows of a set
# with "open_share". Returns it with its parameters:
# - the base's parameters and its point_only;
# - the threshold;
# - base_model, the registry name of the base, so the report can test the masked run against the base's own run.
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


# The horizon steps that the recent rule sets to 0: where the context's own opening profile (series_open_profile),
# looked up at the horizon's timestamps (which follow the context's last step), is below the threshold.
def closed_steps(context: pd.Series, horizon: int, threshold: float) -> np.ndarray:
    future = pd.date_range(context.index[-1] + STEP, periods=horizon, freq=STEP)
    return open_share(future, series_open_profile(context)) < threshold


# Wraps a base builder (→ forecaster, parameters) into a masked forecaster whose opening profile is computed from each
# context (closed_steps).
# Returns it with its parameters (as mask_forecaster, plus where the profile comes from).
def recent_mask_forecaster(base_build: Callable, threshold: float, base_model: str):
    base, base_params = base_build()

    def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...],
                 seed: int) -> tuple[np.ndarray, list[str]]:
        quantiles, status = base(contexts, horizon, levels, seed)
        quantiles = np.array(quantiles, dtype=float)
        for i, context in enumerate(contexts):
            quantiles[i, closed_steps(context, horizon, threshold), :] = 0.0
        return quantiles, list(status)

    return forecast, {"base": base_params, "rule": "every quantile = 0 where open_share < threshold",
                      "profile": "share of the context's days with visitors > 0 per Berlin weekday and slot",
                      "threshold": threshold, "point_only": bool(base_params.get("point_only", False)),
                      "base_model": base_model}
