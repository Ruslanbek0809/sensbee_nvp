# statsforecast's MSTL as a harness forecaster (contract: benchmark/runner.py header): the seasonal baselines that
# handle a daily season of 96 steps, which AutoETS and seasonal AutoARIMA can't do in our budget (task 4 spike;
# baselines step). MSTL removes the daily (and weekly) pattern with STL, forecasts the rest with a non-seasonal model
# (ETS "ZZN", or AutoARIMA: the classic STL + ARIMA), and adds the pattern back. Its intervals come from that model
# (normal), shifted by the seasonal forecast. Each context is fit on its own, in parallel.

import logging

import numpy as np
import pandas as pd

from benchmark.models.autogluon_models import physical_cores

logger = logging.getLogger(__name__)


# Forecast of one context: the mean for the 0.5 level and the statsforecast interval bounds for the others, mapped the
# way AutoGluon maps quantiles (q < 0.5 → "lo-{|q − 0.5|·200}", q > 0.5 → "hi-…").
def _mstl_one(y: np.ndarray, season_length: list[int], trend: str, horizon: int,
              levels: tuple[float, ...]) -> np.ndarray:
    from statsforecast.models import MSTL, AutoARIMA, AutoETS
    widths = sorted({round(abs(q - 0.5) * 200, 1) for q in levels if q != 0.5})
    trend_forecaster = AutoETS(model="ZZN") if trend == "ets" else AutoARIMA(season_length=1)
    result = MSTL(season_length=season_length, trend_forecaster=trend_forecaster).forecast(
        y=y, h=horizon, level=widths)
    columns = []
    for q in levels:
        key = "mean" if q == 0.5 else f"{'lo' if q < 0.5 else 'hi'}-{round(abs(q - 0.5) * 200, 1)}"
        columns.append(np.asarray(result[key], dtype=float))
    return np.stack(columns, axis=1)


# Builds an MSTL forecaster with the given seasons (steps) and the model for the seasonally adjusted series ("ets":
# AutoETS(model="ZZN"); "arima": non-seasonal AutoARIMA), and returns it with its parameters.
def mstl_forecaster(season_length: list[int], trend: str = "ets"):
    if trend not in ("ets", "arima"):
        raise ValueError(f"UNKNOWN TREND MODEL {trend}")
    n_jobs = physical_cores()

    def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...],
                 seed: int) -> tuple[np.ndarray, list[str]]:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs)(delayed(_mstl_one)(c.to_numpy(dtype=float), season_length, trend, horizon,
                                                         levels) for c in contexts)
        return np.stack(out), ["ok"] * len(contexts)

    return forecast, {"library": "statsforecast", "model": "MSTL", "season_length": season_length,
                      "trend_forecaster": "AutoETS(model='ZZN')" if trend == "ets" else "AutoARIMA(season_length=1)",
                      "n_jobs": n_jobs, "point_only": False}
