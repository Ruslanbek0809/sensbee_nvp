# Trained forecasters for the benchmark harness (Phase 3): AutoGluon's global models behind the runner's trainer
# contract (benchmark/runner.py header). A trainer fits one model per sensor on the training items before the period
# and returns a forecaster that predicts every harness context as a new series with that frozen model.
#
# Values only: no date or time features, no covariates, no static features, so a trained model sees what the other
# models see. Every default that matters is set explicitly and recorded (AutoGluon 1.6.3 source, checked 2026-10-14):
# - lags: AutoGluon's own 15-min set capped at 672 (one week), so a model's input window fits the 14-day context;
# - date_features=[] (the default adds minute, hour, weekday, day of month and day of year from UTC timestamps);
# - eval_metric="SQL" makes DirectTabular train a quantile LightGBM (only quantile metrics do); nothing is selected;
# - LightGBM: AutoGluon's defaults written out (10,000 rounds, learning rate 0.05, adaptive early stopping on the last
#   96 rows of each training item, which the tabular models hold out themselves) plus the harness seed;
# - max_num_items / max_num_samples = None: no random subsampling of items or rows.
# The tabular target scaler is fitted per series on whatever it is given: the whole training item when fitting, the
# 14- or 28-day context when forecasting. Training items get ids ("train000000", ...) that no context id ("000000",
# ...) can match, so every context is a series unseen in fit. RecursiveTabular's quantiles are then Gaussian around its
# forecast with a spread taken from the context's seasonal error, not from the model, so it is scored as a point
# forecaster (decision D3-5).

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.models.autogluon_models import _batch_frame
from benchmark.snapshot import BUCKET_FREQ
from benchmark.tasks import HORIZON, QUANTILE_LEVELS, SEASON

logger = logging.getLogger(__name__)

# AutoGluon's get_lags_for_frequency("15min") without its largest lag, 673 (checked in the bench venv, 2026-10-14).
LAGS = (*range(1, 15), *(lag for day in range(1, 8) for lag in (96 * day - 1, 96 * day, 96 * day + 1)))[:-1]
LIGHTGBM = {"num_boost_round": 10000, "learning_rate": 0.05, "ag.early_stop": "adaptive"}
TABULAR_COMMON = {"lags": list(LAGS), "date_features": [], "model_name": "GBM", "max_num_items": None,
                  "max_num_samples": None}
EVAL_METRIC = "SQL"


# Steps of history a tabular model reads: its largest lag plus its differencing.
def tabular_input_steps(params: dict) -> int:
    return max(params["lags"]) + sum(params["differences"])


# Builds a trainer for one AutoGluon global model (its hyperparameter key, e.g. "DirectTabular") and returns it with
# the hyperparameters recorded in run.json. point_only: every quantile is the model's own forecast ("mean").
def autogluon_trainer(model: str, params: dict, point_only: bool = False):
    hyperparameters = {**params, "model_hyperparameters": dict(LIGHTGBM)}

    def train(sensor: str, items: list[pd.Series], seed: int):
        from autogluon.timeseries import TimeSeriesPredictor
        if not items:
            raise ValueError(f"NO TRAINING ITEMS FOR {sensor}")
        workdir = tempfile.TemporaryDirectory(prefix="trained_")  # kept alive by the forecaster below
        predictor = TimeSeriesPredictor(prediction_length=HORIZON, freq=BUCKET_FREQ,
                                        quantile_levels=list(QUANTILE_LEVELS), eval_metric=EVAL_METRIC,
                                        path=str(Path(workdir.name) / "predictor"), verbosity=0, log_to_file=False)
        fitted = {**hyperparameters, "model_hyperparameters": {**hyperparameters["model_hyperparameters"],
                                                               "seed": seed}}
        predictor.fit(_batch_frame(items, prefix="train"), hyperparameters={model: fitted}, enable_ensemble=False,
                      skip_model_selection=True, random_seed=seed)
        if not predictor.model_names():  # AutoGluon logs a failed fit and skips the model instead of raising
            raise RuntimeError(f"AUTOGLUON TRAINED NO {model} MODEL FOR {sensor}")
        columns = ["mean"] * len(QUANTILE_LEVELS) if point_only else [str(level) for level in QUANTILE_LEVELS]

        def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...],
                     seed: int) -> tuple[np.ndarray, list[str]]:
            if horizon != HORIZON or tuple(levels) != QUANTILE_LEVELS:
                raise ValueError(f"TRAINED FOR HORIZON {HORIZON} AND {QUANTILE_LEVELS}")
            predictions = predictor.predict(_batch_frame(contexts), random_seed=seed)
            quantiles = np.stack([predictions.loc[f"{i:06d}", columns].to_numpy(dtype=float)
                                  for i in range(len(contexts))])
            return quantiles, ["ok"] * len(contexts)

        forecast.workdir, forecast.predictor = workdir, predictor  # the folder lives as long as the forecaster
        return forecast, {"models": predictor.model_names(), "training_rows": int(sum(len(i) for i in items))}

    return train, {"library": "autogluon.timeseries", "model": model, "hyperparameters": hyperparameters,
                   "eval_metric": EVAL_METRIC, "point_only": point_only, "input_steps": tabular_input_steps(params)}


# The two tabular models (AutoGluon's own differences and scalers, written out).
DIRECT_TABULAR = {**TABULAR_COMMON, "differences": [], "target_scaler": "mean_abs"}
RECURSIVE_TABULAR = {**TABULAR_COMMON, "differences": [SEASON], "target_scaler": "standard"}
