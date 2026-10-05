# AutoGluon's local models (Naive, SeasonalNaive, AutoETS, AutoARIMA, Theta) as harness forecasters
# (contract: benchmark/runner.py header).
#
# Local models fit inside predict(), one fit per series, so every context of a batch is forecast on its own: fit() only
# stores the settings plus a marginal "dummy" forecast that AutoGluon uses for all-NaN series, which valid contexts
# never are (autogluon/timeseries/models/local/abstract_local_model.py, v1.6.3). Every default that matters is set
# explicitly: max_ts_length=None (the harness cuts the context), use_fallback_model=False
# (an error surfaces and the runner marks the origin failed instead of AutoGluon silently swapping in SeasonalNaive),
# and n_jobs = the physical cores.

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.snapshot import BUCKET_FREQ

logger = logging.getLogger(__name__)

AG_COMMON = {"max_ts_length": None, "use_fallback_model": False}


# Number of physical CPU cores (AutoGluon's own default for n_jobs, made explicit and recorded).
def physical_cores() -> int:
    import joblib
    return joblib.cpu_count(only_physical_cores=True)


# One TimeSeriesDataFrame with one item per context; item ids keep the batch order.
def _batch_frame(contexts: list[pd.Series]):
    from autogluon.timeseries import TimeSeriesDataFrame
    frames = [pd.DataFrame({"item_id": f"{i:06d}", "timestamp": c.index, "target": c.to_numpy(dtype=float)})
              for i, c in enumerate(contexts)]
    return TimeSeriesDataFrame.from_data_frame(pd.concat(frames, ignore_index=True))


# Builds a forecaster for one AutoGluon local model (its hyperparameter key, e.g. "AutoETS") and returns it with the
# effective hyperparameters. point_only: every quantile is the model's own forecast ("mean"), for models whose
# quantiles are not their forecast distribution (Theta: 200 simulated paths, decision D-B4).
def autogluon_forecaster(model: str, params: dict, point_only: bool = False):
    hyperparameters = {**AG_COMMON, **params, "n_jobs": physical_cores()}

    def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...],
                 seed: int) -> tuple[np.ndarray, list[str]]:
        from autogluon.timeseries import TimeSeriesPredictor
        data = _batch_frame(contexts)
        with tempfile.TemporaryDirectory() as tmp:
            predictor = TimeSeriesPredictor(prediction_length=horizon, freq=BUCKET_FREQ, quantile_levels=list(levels),
                                            path=str(Path(tmp) / "predictor"), verbosity=0, log_to_file=False)
            predictor.fit(data, hyperparameters={model: hyperparameters}, enable_ensemble=False,
                          skip_model_selection=True, random_seed=seed)
            predictions = predictor.predict(data, random_seed=seed)
        columns = ["mean"] * len(levels) if point_only else [str(level) for level in levels]
        quantiles = np.stack([predictions.loc[f"{i:06d}", columns].to_numpy(dtype=float)
                              for i in range(len(contexts))])
        return quantiles, ["ok"] * len(contexts)

    return forecast, {"library": "autogluon.timeseries", "model": model, "hyperparameters": hyperparameters,
                      "point_only": point_only}
