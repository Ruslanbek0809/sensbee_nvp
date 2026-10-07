# Diagnostics for the context-length ablation, built only from harness run files: does a forecast keep the daily cycle
# (amplitude and shape of its median compared with the reference's median), and how does the error depend on the hour
# of the origin.
#
# The reference (Seasonal Naive m = 96, D-B6) forecasts the last observed day again, so its median is the daily cycle
# the context ends with. amplitude_ratio = std of the model's median over the horizon / std of the reference's median
# (≈ 0: a flat forecast, ≈ 1: the cycle's full swing); shape_correlation = their correlation (≈ 1: the same daily
# shape). reference_last_day_difference() checks the "last observed day" property on the snapshot.

import numpy as np
import pandas as pd

from benchmark.tasks import BUCKETS_PER_DAY, context_window

KEYS = ["model", "sensor", "context", "origin"]


# Medians of complete forecast windows as a matrix: one row per (model, sensor, context, origin), one column per step.
# Rows of failed origins are dropped. Returns (keys of the rows, medians).
def _median_windows(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    ok = forecasts[forecasts["status"] != "failed"].sort_values(KEYS + ["step"])
    steps = int(ok["step"].max())
    medians = ok["q0.5"].to_numpy(dtype=float).reshape(-1, steps)
    keys = ok[KEYS].iloc[::steps].reset_index(drop=True)
    return keys, medians


# Amplitude ratio and shape correlation of every model forecast against the reference's forecast for the same sensor
# and origin. forecasts: rows of forecasts.csv.gz with a "model" column; reference: the reference run's rows at its
# context (no "model" column needed). Both are NaN when the reference median is constant, the correlation also when
# the model's median is.
def shape_table(forecasts: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    keys, medians = _median_windows(forecasts)
    ref_keys, ref_medians = _median_windows(reference.assign(model="reference"))
    ref_index = {(s, o): i for i, (s, o) in enumerate(zip(ref_keys["sensor"], ref_keys["origin"]))}
    rows = [ref_index[(s, o)] for s, o in zip(keys["sensor"], keys["origin"])]
    ref = ref_medians[rows]
    model_std, ref_std = medians.std(axis=1), ref.std(axis=1)
    centred = (medians - medians.mean(axis=1, keepdims=True)) * (ref - ref.mean(axis=1, keepdims=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        amplitude = np.where(ref_std > 0, model_std / ref_std, np.nan)
        correlation = np.where((ref_std > 0) & (model_std > 0), centred.mean(axis=1) / (model_std * ref_std), np.nan)
    return keys.assign(amplitude_ratio=amplitude, shape_correlation=correlation)


# Mean model error / mean reference error per model, sensor, context, horizon and UTC hour of the origin, over the
# origins both have a finite value for. metrics: metrics.csv rows with a "model" column; reference: the reference
# run's rows at its context.
def hour_table(metrics: pd.DataFrame, reference: pd.DataFrame, metric: str = "MASE") -> pd.DataFrame:
    ref = reference[["sensor", "origin", "horizon_steps", metric]].rename(columns={metric: "reference_error"})
    paired = metrics[["model", "sensor", "context", "origin", "horizon_steps", metric]].merge(
        ref, on=["sensor", "origin", "horizon_steps"])
    paired = paired[np.isfinite(paired[metric]) & np.isfinite(paired["reference_error"])]
    paired = paired.assign(hour=pd.to_datetime(paired["origin"]).dt.hour)
    out = paired.groupby(["model", "sensor", "context", "horizon_steps", "hour"]).agg(
        n_origins=(metric, "size"), error=(metric, "mean"), reference_error=("reference_error", "mean")).reset_index()
    return out.assign(relative_error=out["error"] / out["reference_error"])


# Largest |reference median − the last day of the context| over all origins of the reference rows, per sensor: 0 when
# the reference forecast repeats the last observed day exactly. series: the target series of each sensor.
def reference_last_day_difference(reference: pd.DataFrame, series: dict[str, pd.Series]) -> dict[str, float]:
    out = {}
    for sensor, rows in reference[reference["status"] != "failed"].groupby("sensor"):
        worst = 0.0
        for origin, window in rows.groupby("origin"):
            last_day = context_window(series[sensor], pd.Timestamp(origin), BUCKETS_PER_DAY).to_numpy(dtype=float)
            median = window.sort_values("step")["q0.5"].to_numpy(dtype=float)
            worst = max(worst, float(np.abs(median - np.resize(last_day, len(median))).max()))
        out[sensor] = worst
    return out
