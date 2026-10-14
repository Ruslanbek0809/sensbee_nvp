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
# DeepAR (GluonTS): time_features=[], no static or known covariates, AutoGluon's defaults for the network and training
# written out, and context_length = 2 days unrolled plus the same lags (one week back), an input window of 863 steps.
# PatchTST (GluonTS): no covariates, AutoGluon's and GluonTS's defaults written out, and a context_length of one week
# (672 steps, 84 patches of 16 with stride 8); it has no lags and no time features.
# Without validation data AutoGluon gives GluonTS models no early stopping, so DeepAR and PatchTST stop on the final
# TUNING_DAYS before the cutoff: those days are held out of training and validated one day at a time, and GluonTS keeps
# the epoch with the lowest validation loss. DeepAR's validation loss also scores the context_length - 1 steps before
# each validated day (DeepARLightningModule.validation_step has no future_only), so its training ends that much earlier
# as well (DEEPAR_HOLDOUT) and no scored step was trained on; PatchTST's loss scores only the validated day. The epochs
# run, the best epoch and its loss come from the Lightning logs (keep_lightning_logs) into the fit record. GluonTS
# drops unknown hyperparameter keys without a word, so the tests check the fitted network itself. DeepAR's built-in
# log "age" feature (position in the given series) has no switch. GluonTS's default training sampler lets a training
# window end anywhere after an item's first step and fills the missing past with zeros, which the network can't tell
# from real zeros (a closed hall, 0 °C) and which no harness context contains (3-9% of the windows on our training
# items), so training windows start only once an item holds a full input window (min_past = input_steps).
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

from benchmark.models.autogluon_models import _batch_frame, _quantiles
from benchmark.snapshot import BUCKET_FREQ, BUCKETS_PER_WEEK
from benchmark.tasks import BUCKETS_PER_DAY, HORIZON, QUANTILE_LEVELS, SEASON

logger = logging.getLogger(__name__)

# AutoGluon's get_lags_for_frequency("15min") without its largest lag, 673 (checked in the bench venv, 2026-10-14).
LAGS = (*range(1, 15), *(lag for day in range(1, 8) for lag in (96 * day - 1, 96 * day, 96 * day + 1)))[:-1]
LIGHTGBM = {"num_boost_round": 10000, "learning_rate": 0.05, "ag.early_stop": "adaptive"}
TABULAR_COMMON = {"lags": list(LAGS), "date_features": [], "model_name": "GBM", "max_num_items": None,
                  "max_num_samples": None}
EVAL_METRIC = "SQL"
TUNING_DAYS = 7
# AutoGluon's training defaults for GluonTS models, written out, plus no covariates and the kept Lightning logs.
GLUONTS_COMMON = {"disable_static_features": True, "disable_known_covariates": True, "max_epochs": 100,
                  "num_batches_per_epoch": 50, "batch_size": 64, "lr": 1e-3, "early_stopping_patience": 20,
                  "predict_batch_size": 500, "keep_lightning_logs": True}


# Steps of history a model reads: a tabular model's largest lag plus its differencing, DeepAR's unrolled context plus
# its largest lag minus one (GluonTS stores lag l as index l - 1), or PatchTST's context.
def input_steps(params: dict) -> int:
    if "lags_seq" in params:
        return params["context_length"] + max(params["lags_seq"]) - 1
    if "lags" in params:
        return max(params["lags"]) + sum(params["differences"])
    return params["context_length"]


# Splits training items for early stopping on the last `days` days: training then ends `days` days plus `holdout`
# steps before the end of the last item, and the tuning data holds `days` copies of that item, cut 0, 1, ... days
# before its end, so the model validates on each of the final days (the last HORIZON steps of each tuning item, plus
# the `holdout` steps before them that a GluonTS validation loss also scores). Raises when the shortened item can't hold
# one full input window plus a horizon.
def tuning_split(items: list[pd.Series], days: int, steps: int,
                 holdout: int = 0) -> tuple[list[pd.Series], list[pd.Series]]:
    last = items[-1]
    keep = len(last) - days * BUCKETS_PER_DAY - holdout
    if keep < steps + HORIZON:
        raise ValueError(f"LAST TRAINING ITEM OF {len(last)} STEPS IS TOO SHORT FOR {days} TUNING DAYS")
    return [*items[:-1], last.iloc[:keep]], [last.iloc[: len(last) - k * BUCKETS_PER_DAY] for k in range(days)]


# Epochs run, the best epoch (1-based, from the best checkpoint's file name) and its validation loss, read from the
# Lightning logs a GluonTS model keeps with keep_lightning_logs. Raises when no validation loss was logged.
def _lightning_epochs(folder: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    (version,) = folder.glob("**/lightning_logs/version_*")
    events = EventAccumulator(str(version))
    events.Reload()
    if "val_loss" not in events.Tags()["scalars"]:
        raise RuntimeError("NO VALIDATION LOSS LOGGED: EARLY STOPPING DID NOT RUN")
    losses = [event.value for event in events.Scalars("val_loss")]
    (checkpoint,) = version.glob("checkpoints/epoch=*.ckpt")
    best = int(checkpoint.name.split("-")[0].removeprefix("epoch=")) + 1
    return {"epochs_run": len(losses), "best_epoch": best, "best_val_loss": float(losses[best - 1])}


# Builds a trainer for one AutoGluon global model (its hyperparameter key, e.g. "DirectTabular") and returns it with
# the hyperparameters recorded in run.json. point_only: every quantile is the model's own forecast ("mean");
# tuning_days > 0: early stopping on the final days before the cutoff (tuning_split), with `holdout` steps before them
# that the model's validation loss also scores. A tabular model gets the harness seed in its LightGBM
# hyperparameters; every model gets it as AutoGluon's fit and predict seed. A GluonTS model (one with a
# context_length) samples training windows only where a full input window exists (min_past). The fitted model is
# not persisted in memory: AutoGluon reloads it per prediction, and persisting it changes DeepAR's sampled forecasts.
def autogluon_trainer(model: str, params: dict, point_only: bool = False, tuning_days: int = 0, holdout: int = 0):
    hyperparameters = dict(params)
    steps = input_steps(params)
    min_past = steps if "context_length" in params else 0

    def train(sensor: str, items: list[pd.Series], seed: int):
        from autogluon.timeseries import TimeSeriesPredictor
        if not items:
            raise ValueError(f"NO TRAINING ITEMS FOR {sensor}")
        workdir = tempfile.TemporaryDirectory(prefix="trained_")  # kept alive by the forecaster below
        predictor = TimeSeriesPredictor(prediction_length=HORIZON, freq=BUCKET_FREQ,
                                        quantile_levels=list(QUANTILE_LEVELS), eval_metric=EVAL_METRIC,
                                        path=str(Path(workdir.name) / "predictor"), verbosity=0, log_to_file=False)
        fitted = dict(hyperparameters)
        if "model_hyperparameters" in fitted:
            fitted["model_hyperparameters"] = {**fitted["model_hyperparameters"], "seed": seed}
        if min_past:
            from gluonts.transform import ExpectedNumInstanceSampler
            fitted["train_sampler"] = ExpectedNumInstanceSampler(num_instances=1.0, min_past=min_past,
                                                                 min_future=HORIZON)
        tuning, info = None, {}
        if tuning_days:
            items, tune = tuning_split(items, tuning_days, steps, holdout)
            tuning = _batch_frame(tune, prefix="tune")
            info = {"tuning_days": tuning_days, "holdout_steps": holdout, "tuning_end": str(tune[0].index[-1])}
        info = {"training_rows": int(sum(len(i) for i in items)), **info}
        predictor.fit(_batch_frame(items, prefix="train"), tuning_data=tuning, hyperparameters={model: fitted},
                      enable_ensemble=False, skip_model_selection=True, random_seed=seed)
        if not predictor.model_names():  # AutoGluon logs a failed fit and skips the model instead of raising
            raise RuntimeError(f"AUTOGLUON TRAINED NO {model} MODEL FOR {sensor}")
        if params.get("keep_lightning_logs"):
            info.update(_lightning_epochs(Path(workdir.name)))

        def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...],
                     seed: int) -> tuple[np.ndarray, list[str]]:
            if horizon != HORIZON or tuple(levels) != QUANTILE_LEVELS:
                raise ValueError(f"TRAINED FOR HORIZON {HORIZON} AND {QUANTILE_LEVELS}")
            predictions = predictor.predict(_batch_frame(contexts), random_seed=seed)
            return _quantiles(predictions, len(contexts), QUANTILE_LEVELS, point_only), ["ok"] * len(contexts)

        forecast.workdir, forecast.predictor = workdir, predictor  # the folder lives as long as the forecaster
        return forecast, {"models": predictor.model_names(), **info}

    return train, {"library": "autogluon.timeseries", "model": model, "hyperparameters": hyperparameters,
                   "eval_metric": EVAL_METRIC, "point_only": point_only, "input_steps": steps,
                   "tuning_days": tuning_days, "holdout_steps": holdout, "train_min_past": min_past}


# The two tabular models (AutoGluon's own differences and scalers, written out).
DIRECT_TABULAR = {**TABULAR_COMMON, "differences": [], "target_scaler": "mean_abs",
                  "model_hyperparameters": dict(LIGHTGBM)}
RECURSIVE_TABULAR = {**TABULAR_COMMON, "differences": [SEASON], "target_scaler": "standard",
                     "model_hyperparameters": dict(LIGHTGBM)}
# DeepAR: AutoGluon's network and training defaults written out (2 LSTM layers of 40, dropout 0.1, Student-t output,
# mean scaling per window, 100 epochs of 50 batches of 64, learning rate 1e-3, patience 20, 250 sample paths).
DEEPAR = {"context_length": 2 * SEASON, "lags_seq": list(LAGS), "time_features": [], "num_layers": 2, "hidden_size": 40,
          "dropout_rate": 0.1, "scaling": True, **GLUONTS_COMMON}
DEEPAR_HOLDOUT = DEEPAR["context_length"] - 1  # context steps its validation loss scores before each validated day
# PatchTST: AutoGluon's training defaults and GluonTS's network defaults written out (patches of 16 with stride 8,
# padded at the end, 2 encoder layers of width 32 with 4 heads and feedforward 128, dropout 0.1, Student-t output,
# mean scaling per window, weight decay 1e-8), with a one-week context instead of AutoGluon's 96 steps.
PATCHTST = {"context_length": BUCKETS_PER_WEEK, "patch_len": 16, "stride": 8, "padding_patch": "end", "d_model": 32,
            "nhead": 4, "dim_feedforward": 128, "num_encoder_layers": 2, "dropout": 0.1, "activation": "relu",
            "norm_first": False, "scaling": "mean", "weight_decay": 1e-8, **GLUONTS_COMMON}
