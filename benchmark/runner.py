# Runs one forecaster over the benchmark tasks of a frozen snapshot and writes the raw results to a new folder that is
# never overwritten: forecasts.csv.gz (every valid origin, step and quantile with the true value), metrics.csv (per
# origin and horizon), origins.csv (every candidate origin with its validity) and run.json (snapshot, tasks, model,
# seed, code commit, versions, counts).
#
# Forecaster contract: f(contexts, horizon, levels, seed) -> (quantiles [n, horizon, levels], status per origin), with
# status "ok" or "fallback" (the forecaster reports its own fallbacks). Batch items are independent: a forecast may use
# only its own context, so no cross-series learning across origins, whose contexts hold earlier origins' targets.
# Fitted models get training data only through tasks.training_series(). An origin is "failed" when the forecaster
# raises (the runner then retries the origins one by one), returns the wrong shape, or returns NaN or inf. Crossing
# quantiles are counted, then sorted for every model, so scores don't depend on which adapter sorts.

import dataclasses
import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from benchmark.metrics import METRICS, seasonal_scale, window_metrics
from benchmark.snapshot import (
    BUCKET_FREQ, BUCKETS_PER_DAY, MANIFEST_NAME, TIME_FORMAT, git_state, load_snapshot, sha256_file, write_raw_csv,
)
from benchmark.tasks import (
    CLOSURE_GAP_BUCKETS, CONTEXT_DAYS, GUARD, HORIZON, MIN_CONTEXT_COVERAGE, ORIGIN_FREQ, QUANTILE_LEVELS,
    REPORT_STEPS, SCALE_DAYS, SEASON, TASKS, VALIDATION_DAYS, SensorTask, context_window, origin_table, scale_history,
    target_series, target_window,
)

logger = logging.getLogger(__name__)

STATUSES = ("ok", "fallback")

Forecaster = Callable[[list[pd.Series], int, tuple[float, ...], int], tuple[np.ndarray, list[str]]]


# Naive check forecaster: every step and every quantile is the last context value.
def naive_forecaster(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...],
                     seed: int) -> tuple[np.ndarray, list[str]]:
    quantiles = np.stack([np.full((horizon, len(levels)), float(c.iloc[-1])) for c in contexts])
    return quantiles, ["ok"] * len(contexts)


# Seasonal-naive check forecaster with season m: the forecast repeats the last m context values, the same for every
# quantile. Raises when a context is shorter than m (the runner then marks that origin failed).
def seasonal_naive_forecaster(m: int) -> Forecaster:
    def forecast(contexts: list[pd.Series], horizon: int, levels: tuple[float, ...],
                 seed: int) -> tuple[np.ndarray, list[str]]:
        out = []
        for context in contexts:
            if len(context) < m:
                raise ValueError(f"CONTEXT OF {len(context)} STEPS IS SHORTER THAN THE SEASON {m}")
            point = np.resize(context.to_numpy(dtype=float)[-m:], horizon)
            out.append(np.repeat(point[:, None], len(levels), axis=1))
        return np.stack(out), ["ok"] * len(contexts)
    return forecast


# Check forecasters for harness runs (point forecasts, all quantiles equal), not thesis baselines: name → (forecaster,
# parameters recorded in run.json).
CHECK_FORECASTERS: dict[str, tuple[Forecaster, dict]] = {
    "naive": (naive_forecaster, {}),
    "seasonal_naive_96": (seasonal_naive_forecaster(96), {"season": 96}),
    "seasonal_naive_672": (seasonal_naive_forecaster(672), {"season": 672}),
}


# Marks origins with non-finite quantiles or an unknown status as failed (their quantiles become NaN).
def _check(quantiles: np.ndarray, status: list[str]) -> tuple[np.ndarray, list[str]]:
    for i in range(len(status)):
        if status[i] not in STATUSES or not np.isfinite(quantiles[i]).all():
            quantiles[i] = np.nan
            status[i] = "failed"
    return quantiles, status


# Calls the forecaster on one batch and checks its output. On an exception or a malformed batch result it retries the
# origins one by one, so one bad origin can't fail the whole batch. Returns quantiles [n, HORIZON, levels], NaN for
# failed origins, and the status per origin ("ok", "fallback" or "failed").
def _predict(forecaster: Forecaster, contexts: list[pd.Series], levels: tuple[float, ...],
             seed: int) -> tuple[np.ndarray, list[str]]:
    n = len(contexts)
    try:
        quantiles, status = forecaster(contexts, HORIZON, levels, seed)
        quantiles = np.array(quantiles, dtype=float)
        if quantiles.shape == (n, HORIZON, len(levels)) and len(status) == n:
            return _check(quantiles, list(status))
        logger.warning(f"MALFORMED BATCH RESULT {quantiles.shape} with {len(status)} statuses; RETRYING {n} ORIGINS")
    except Exception as exc:
        logger.warning(f"BATCH FAILED ({type(exc).__name__}: {exc}); RETRYING {n} ORIGINS ONE BY ONE")
    out = np.full((n, HORIZON, len(levels)), np.nan)
    statuses = ["failed"] * n
    for i, context in enumerate(contexts):
        try:
            quantiles, status = forecaster([context], HORIZON, levels, seed)
            quantiles = np.array(quantiles, dtype=float)
            if quantiles.shape == (1, HORIZON, len(levels)) and len(status) == 1:
                quantiles, status = _check(quantiles, list(status))
                out[i], statuses[i] = quantiles[0], status[0]
        except Exception as exc:
            logger.warning(f"ORIGIN {context.index[-1] + pd.Timedelta(BUCKET_FREQ)} FAILED "
                           f"({type(exc).__name__}: {exc})")
    return out, statuses


# Long table of one batch: one row per (origin, step) with the true value and the quantiles.
def _forecast_frame(sensor: str, context: str, origins: list[pd.Timestamp], status: list[str], targets: np.ndarray,
                    quantiles: np.ndarray, levels: tuple[float, ...]) -> pd.DataFrame:
    origin_index = pd.DatetimeIndex(origins).repeat(HORIZON)
    steps = np.tile(np.arange(1, HORIZON + 1), len(origins))
    return pd.DataFrame({
        "sensor": sensor,
        "context": context,
        "origin": origin_index,
        "step": steps,
        "timestamp": origin_index + (steps - 1) * pd.Timedelta(BUCKET_FREQ),
        "status": np.repeat(status, HORIZON),
        "y": targets.reshape(-1),
        **{f"q{level:g}": quantiles[:, :, j].reshape(-1) for j, level in enumerate(levels)},
    })


# Runs a forecaster over the given tasks (default: all TASKS) for one period and context lengths (days; None = all
# history before the origin), and writes the results to out_root/<run_id>. run_id defaults to
# "<UTC start time>_<model>". Raises FileExistsError if that folder exists. Returns the run folder.
def run(forecaster: Forecaster, model: str, snapshot_dir: Path, out_root: Path,
        tasks: Optional[dict[str, SensorTask]] = None, period: str = "test",
        context_days: tuple[Optional[int], ...] = CONTEXT_DAYS, seed: int = 0, run_id: Optional[str] = None,
        model_params: Optional[dict] = None, model_info: Optional[dict] = None) -> Path:
    tasks = TASKS if tasks is None else tasks
    started = datetime.now(timezone.utc)
    code = git_state()  # at the start: the code that runs, even if a commit happens during a long run
    run_dir = Path(out_root) / (run_id or f"{started:%Y-%m-%dT%H%M%SZ}_{model}")
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} EXISTS; runs are never overwritten")

    snapshot_dir = Path(snapshot_dir)
    frames = load_snapshot(snapshot_dir)
    manifest = json.loads((snapshot_dir / MANIFEST_NAME).read_text())
    levels = QUANTILE_LEVELS
    forecast_frames, metric_rows, origin_frames, counts = [], [], [], {}
    for sensor, task in tasks.items():
        series = target_series(frames[sensor], task)
        table = origin_table(series, task, period)
        origin_frames.append(table.assign(sensor=sensor))
        origins = list(table.loc[table["valid"], "origin"])
        scales = [seasonal_scale(scale_history(series, origin, task), SEASON) for origin in origins]
        targets = np.array([target_window(series, origin).to_numpy(dtype=float) for origin in origins]).reshape(
            len(origins), HORIZON)
        counts[sensor] = {"candidates": len(table), "valid": len(origins),
                          "reasons": {reason: int(n) for reason, n in table["reason"].value_counts().items()},
                          "undefined_scale": int(np.isnan(scales).sum()), "contexts": {}}
        for days in context_days:
            label = "all" if days is None else f"{days}d"
            contexts = [context_window(series, origin, None if days is None else days * BUCKETS_PER_DAY)
                        for origin in origins]
            quantiles, status = _predict(forecaster, contexts, levels, seed) if origins else (
                np.empty((0, HORIZON, len(levels))), [])
            crossed = int(np.any(np.diff(quantiles, axis=-1) < 0, axis=-1).sum())
            quantiles = np.sort(quantiles, axis=-1)
            counts[sensor]["contexts"][label] = {**{s: status.count(s) for s in (*STATUSES, "failed")},
                                                 "crossed_steps": crossed}
            for i, origin in enumerate(origins):
                if status[i] == "failed":
                    rows = [{"horizon_steps": h, **{name: np.nan for name in METRICS}} for h in REPORT_STEPS]
                else:
                    rows = window_metrics(targets[i], quantiles[i], levels, scales[i], REPORT_STEPS)
                for row in rows:
                    metric_rows.append({"sensor": sensor, "context": label, "origin": origin, "status": status[i],
                                        "context_steps": len(contexts[i]), "scale": scales[i], **row})
            forecast_frames.append(_forecast_frame(sensor, label, origins, status, targets, quantiles, levels))
        logger.info(f"RAN {model} on {sensor}: {len(origins)} of {len(table)} origins valid")

    run_dir.mkdir(parents=True)
    forecasts_sha256 = write_raw_csv(pd.concat(forecast_frames, ignore_index=True), run_dir / "forecasts.csv.gz")
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False, date_format=TIME_FORMAT)
    origins_table = pd.concat(origin_frames, ignore_index=True)
    origins_table = origins_table[["sensor", *[c for c in origins_table.columns if c != "sensor"]]]
    origins_table.to_csv(run_dir / "origins.csv", index=False, date_format=TIME_FORMAT)
    finished = datetime.now(timezone.utc)
    record = {
        "run_id": run_dir.name,
        "model": model,
        "model_params": model_params or {},
        "model_info": model_info or {},
        "seed": seed,
        "snapshot": {"id": manifest["snapshot_id"], "manifest_sha256": sha256_file(snapshot_dir / MANIFEST_NAME)},
        "period": period,
        "context_days": [days if days is not None else "all" for days in context_days],
        "settings": {
            "horizon": HORIZON, "report_steps": list(REPORT_STEPS), "quantile_levels": list(levels),
            "origin_freq": ORIGIN_FREQ, "guard_hours": GUARD / pd.Timedelta("1h"),
            "min_context_coverage": MIN_CONTEXT_COVERAGE, "season": SEASON, "scale_days": SCALE_DAYS,
            "validation_days": VALIDATION_DAYS, "closure_gap_buckets": CLOSURE_GAP_BUCKETS,
            "point_forecast": "median", "crossing_quantiles": "counted, then sorted",
        },
        "tasks": {sensor: dataclasses.asdict(task) for sensor, task in tasks.items()},
        "counts": counts,
        "forecasts_sha256": forecasts_sha256,
        "code": code,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "platform": {"system": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count()},
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished_utc": finished.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_s": round((finished - started).total_seconds(), 3),
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2))
    logger.info(f"RUN WRITTEN to {run_dir}")
    return run_dir
