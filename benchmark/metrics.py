# Forecast metrics for one origin's window, and the relative skill of models across sensors.
#
# Definitions follow the code of fev 0.10.0 and AutoGluon 1.6.3 (checked 2026-10-01): the quantile loss is
# 2·ρ_τ(y − q̂_τ) averaged over the levels (≈ CRPS in the target's units), SQL divides it by the seasonal scale, and
# the skill score is 1 − gmean(clip(error / reference error, 0.01, 100)) over sensors. The point forecast is the
# median. One deviation: the scale is the mean |y_t − y_{t−m}| over a fixed window before the origin
# (tasks.SCALE_DAYS) rather than the whole history, which for visitor counts would mix in off-season zeros.

import math

import numpy as np
import pandas as pd

METRICS = ("MAE", "RMSE", "MASE", "QL", "SQL", "COV80")
MIN_RELATIVE_ERROR = 1e-2
MAX_RELATIVE_ERROR = 100.0


# Mean absolute seasonal difference |y_t − y_{t−m}| over the pairs where both values are known. NaN when there is no
# such pair or the mean is 0, so a scale that can't divide never produces inf.
def seasonal_scale(history: np.ndarray, m: int) -> float:
    values = np.asarray(history, dtype=float)
    if len(values) <= m:
        return math.nan
    diffs = np.abs(values[m:] - values[:-m])
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0 or diffs.mean() == 0:
        return math.nan
    return float(diffs.mean())


# Quantile loss per step: the mean over the levels of 2·ρ_τ(y − q̂_τ), with ρ_τ(u) = max(τ·u, (τ − 1)·u).
# y has shape [steps], quantiles [steps, levels].
def quantile_loss(y: np.ndarray, quantiles: np.ndarray, levels: tuple[float, ...]) -> np.ndarray:
    errors = np.asarray(y, dtype=float)[:, None] - np.asarray(quantiles, dtype=float)
    tau = np.asarray(levels, dtype=float)[None, :]
    return (2 * np.maximum(tau * errors, (tau - 1) * errors)).mean(axis=1)


# Metrics of one origin on the first h steps, for each h in `steps`: MAE, RMSE, MASE and the quantile loss QL of the
# median forecast, SQL = QL / scale, and COV80, the share of steps with q̂_0.1 ≤ y ≤ q̂_0.9. A NaN scale gives NaN
# MASE and SQL.
def window_metrics(y: np.ndarray, quantiles: np.ndarray, levels: tuple[float, ...], scale: float,
                   steps: tuple[int, ...]) -> list[dict]:
    levels = tuple(levels)
    y = np.asarray(y, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    median = quantiles[:, levels.index(0.5)]
    low, high = quantiles[:, levels.index(0.1)], quantiles[:, levels.index(0.9)]
    loss = quantile_loss(y, quantiles, levels)
    rows = []
    for h in steps:
        errors = y[:h] - median[:h]
        mae = float(np.mean(np.abs(errors)))
        ql = float(np.mean(loss[:h]))
        rows.append({
            "horizon_steps": h,
            "MAE": mae,
            "RMSE": float(np.sqrt(np.mean(errors ** 2))),
            "MASE": mae / scale,
            "QL": ql,
            "SQL": ql / scale,
            "COV80": float(np.mean((low[:h] <= y[:h]) & (y[:h] <= high[:h]))),
        })
    return rows


# Relative error of every model against a reference model on one metric, per sensor: the model's mean over origins
# divided by the reference's mean over the same origins. per_origin has one row per (model, sensor, origin) with the
# metric column, for one context length and horizon. A model must have finite values on exactly the origins where the
# reference has them; otherwise this raises, or with missing="impute" sets that sensor's relative error to 1.0
# (fev's convention for a missing result).
def relative_errors(per_origin: pd.DataFrame, metric: str, reference: str, missing: str = "error") -> pd.DataFrame:
    if per_origin.duplicated(["model", "sensor", "origin"]).any():
        raise ValueError("EXPECTED ONE ROW PER (model, sensor, origin); filter to one context and horizon first")
    finite = per_origin[np.isfinite(per_origin[metric].astype(float))]
    rows = []
    for sensor in sorted(per_origin.loc[per_origin["model"] == reference, "sensor"].unique()):
        own_sensor = finite[finite["sensor"] == sensor]
        ref = own_sensor[own_sensor["model"] == reference].set_index("origin")[metric]
        if len(ref) == 0 or not ref.mean() > 0:
            raise ValueError(f"REFERENCE {reference} HAS NO POSITIVE MEAN {metric} on {sensor}")
        for model in sorted(per_origin["model"].unique()):
            own = own_sensor[own_sensor["model"] == model].set_index("origin")[metric]
            paired = set(own.index) == set(ref.index)
            if not paired and missing != "impute":
                raise ValueError(f"UNPAIRED ORIGINS for {model} on {sensor}: {len(own)} finite vs {len(ref)} "
                                 f"for {reference}")
            rows.append({
                "model": model,
                "sensor": sensor,
                "n_origins": len(own),
                "error": float(own.mean()) if paired else math.nan,
                "reference_error": float(ref.mean()),
                "relative_error": float(own.mean() / ref.mean()) if paired else 1.0,
                "imputed": not paired,
            })
    return pd.DataFrame(rows)


# Per model: the geometric mean over sensors of the relative errors clipped to [0.01, 100], and the skill score
# 1 − that mean (fev-bench). Pass only the sensors that belong in the aggregate.
def relative_skill(relative: pd.DataFrame) -> pd.DataFrame:
    logs = np.log(relative["relative_error"].clip(MIN_RELATIVE_ERROR, MAX_RELATIVE_ERROR))
    grouped = logs.groupby(relative["model"])
    gmean = np.exp(grouped.mean())
    return pd.DataFrame({
        "model": gmean.index,
        "gmean_relative_error": gmean.to_numpy(),
        "skill": 1 - gmean.to_numpy(),
        "n_sensors": grouped.size().to_numpy(),
    })
