#!/usr/bin/env python3
# AutoGluon spike (Phase 1 task 4): times one fit of AutoGluon's local models (AutoETS, AutoARIMA, Theta, Naive,
# SeasonalNaive; season 96) per context length on real snapshot contexts, checks the defaults that matter for the
# benchmark (fallback model, seasonality drop, interval shape, hyperparameter keys), and projects the cost of a full
# test run. Writes timings.csv, projection.csv and checks.json to a new folder under --out. No accuracy numbers: a few
# origins are not an evaluation. Needs the bench venv (requirements-bench.txt); no network.
#
# Usage (from sensbee_nvp/):
#   venv-bench/bin/python scripts/spike_autogluon.py --snapshot ../../benchmark_data/snapshots/sensbee-2026-10-01 \
#       --out ../../benchmark_data/spikes

import argparse
import inspect
import json
import logging
import platform
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.snapshot import BUCKET_FREQ, BUCKETS_PER_DAY, MANIFEST_NAME, git_state, load_snapshot, sha256_file  # noqa: E402
from benchmark.tasks import (  # noqa: E402
    CONTEXT_DAYS, HORIZON, QUANTILE_LEVELS, SEASON, TASKS, context_window, origin_table, target_series,
    training_series,
)

logger = logging.getLogger(__name__)

# Every default we rely on, set explicitly. max_ts_length=None: the harness decides the context length (the default
# 2500 would cut a 28-day context). use_fallback_model=False: AutoGluon only logs how many series fell back, never
# which. n_jobs=1: one timing is one fit on one core.
COMMON = {"seasonal_period": SEASON, "max_ts_length": None, "use_fallback_model": False, "n_jobs": 1}
MODELS = {
    "AutoETS": {**COMMON, "model": "ZZZ", "damped": False},
    "AutoARIMA": {**COMMON, "approximation": True, "allowmean": True, "allowdrift": True},  # allowdrift: effective default
    "Theta": {**COMMON, "decomposition_type": "multiplicative"},  # statsforecast's default, made explicit
    "Naive": dict(COMMON),
    "SeasonalNaive": dict(COMMON),
}
PACKAGES = ("autogluon.timeseries", "statsforecast", "coreforecast", "numpy", "pandas", "scipy", "torch")
DEFAULT_SENSORS = ["MANEBACH_WEATHER_STATION", "EISHALLE"]
FALLBACK_STEPS = 5  # ETS raises "tiny datasets" when n <= parameters + 4


# Raised by the alarm handler. A BaseException, so AutoGluon's `except Exception` can't swallow it.
class FitTimeout(BaseException):
    pass


# SIGALRM handler: interrupts a fit that runs longer than the timeout.
def _on_alarm(signum: int, frame) -> None:
    raise FitTimeout()


# Runs fn() under a timeout. Returns (result or None, seconds, status "ok" / "error" / "timeout", error text).
def _timed(fn: Callable, timeout: int) -> tuple[Optional[object], float, str, str]:
    signal.alarm(timeout)
    start = time.perf_counter()
    try:
        result = fn()
        return result, time.perf_counter() - start, "ok", ""
    except FitTimeout:
        return None, time.perf_counter() - start, "timeout", ""
    except Exception as exc:
        return None, time.perf_counter() - start, "error", f"{type(exc).__name__}: {exc}"[:200]
    finally:
        signal.alarm(0)


# One TimeSeriesDataFrame with one item per series.
def _tsdf(series_by_item: dict[str, pd.Series]):
    from autogluon.timeseries import TimeSeriesDataFrame
    frames = [pd.DataFrame({"item_id": item, "timestamp": s.index, "target": s.to_numpy(dtype=float)})
              for item, s in series_by_item.items()]
    return TimeSeriesDataFrame.from_data_frame(pd.concat(frames, ignore_index=True))


# A predictor with exactly one model and no ensemble or model selection. Local models only store their settings in
# fit(); the fit per series happens inside predict().
def _fit_predictor(name: str, params: dict, train, path: Path):
    from autogluon.timeseries import TimeSeriesPredictor
    predictor = TimeSeriesPredictor(prediction_length=HORIZON, freq=BUCKET_FREQ, quantile_levels=list(QUANTILE_LEVELS),
                                    path=str(path), verbosity=0, log_to_file=False)
    return predictor.fit(train, hyperparameters={name: params}, enable_ensemble=False, skip_model_selection=True,
                         random_seed=0)


# Shape checks of one forecast: finite values, crossing quantiles, mean vs median, symmetry of the 80% interval
# around the median (≈ 0 for Gaussian intervals), and the lowest 0.1 quantile (below 0 matters for counts).
def _quantile_checks(preds: pd.DataFrame) -> dict:
    q = preds[[str(level) for level in QUANTILE_LEVELS]].to_numpy(dtype=float)
    mean = preds["mean"].to_numpy(dtype=float)
    i10, i50, i90 = (QUANTILE_LEVELS.index(level) for level in (0.1, 0.5, 0.9))
    return {
        "finite": bool(np.isfinite(q).all() and np.isfinite(mean).all()),
        "crossed_steps": int(np.any(np.diff(q, axis=1) < 0, axis=1).sum()),
        "max_abs_mean_minus_median": float(np.max(np.abs(mean - q[:, i50]))),
        "max_asymmetry": float(np.max(np.abs((q[:, i90] - q[:, i50]) - (q[:, i50] - q[:, i10])))),
        "min_q10": float(q[:, i10].min()),
    }


# Whether AutoGluon accepts every hyperparameter key we pass (it only warns about unknown ones).
def _check_keys() -> dict:
    from autogluon.timeseries.models import AutoARIMAModel, AutoETSModel, NaiveModel, SeasonalNaiveModel, ThetaModel
    classes = {"AutoETS": AutoETSModel, "AutoARIMA": AutoARIMAModel, "Theta": ThetaModel, "Naive": NaiveModel,
               "SeasonalNaive": SeasonalNaiveModel}
    out = {}
    for name, params in MODELS.items():
        model = classes[name](freq=BUCKET_FREQ, prediction_length=HORIZON, hyperparameters=params)
        unknown = sorted(set(params) - set(model.allowed_hyperparameters))
        out[name] = {"all_accepted": not unknown, "unknown": unknown}
    return out


# AutoETS on the last FALLBACK_STEPS values of a context: without the fallback model the error must surface; with it,
# AutoGluon must silently return its fallback (here the naive branch: the last value at every step).
def _fallback_check(train, context: pd.Series, tmp: Path, timeout: int) -> dict:
    tiny = context.iloc[-FALLBACK_STEPS:]
    out = {}
    for flag in (False, True):
        predictor = _fit_predictor("AutoETS", {**MODELS["AutoETS"], "use_fallback_model": flag}, train,
                                   tmp / f"fallback_{flag}")
        preds, seconds, status, error = _timed(lambda: predictor.predict(_tsdf({"tiny": tiny}), random_seed=0), timeout)
        result = {"status": status, "error": error}
        if preds is not None:
            result["mean_equals_last_value"] = bool(np.allclose(preds["mean"].to_numpy(dtype=float), tiny.iloc[-1]))
        out["use_fallback_model_" + str(flag)] = result
    return out


# ETS checks with statsforecast directly: the components AutoETS chooses per context length, whether AutoGluon's
# 1-day AutoETS equals a non-seasonal "ZZN" fit (AutoGluon drops the season below 2 seasons), and the AICc and
# residual variance of fixed additive models with and without the season on the longest context (why the season
# isn't chosen).
def _ets_checks(contexts: dict[int, pd.Series], ag_one_day_mean: Optional[np.ndarray], timeout: int) -> dict:
    from statsforecast.ets import ets_f
    from statsforecast.models import AutoETS
    longest = contexts[max(contexts)].to_numpy(dtype=float)
    fixed = {}
    for model in ("ANN", "AAN", "ANA", "AAA"):
        fit, seconds, status, error = _timed(lambda: ets_f(longest, m=SEASON, model=model, damped=False), timeout)
        fixed[model] = ({"aicc": float(fit["aicc"]), "sigma2": float(fit["sigma2"]), "seconds": round(seconds, 3)}
                        if fit is not None else f"{status}: {error}")
    components = {}
    for days, context in contexts.items():
        y = context.to_numpy(dtype=float)
        model, seconds, status, error = _timed(
            lambda: AutoETS(season_length=SEASON, model="ZZZ", damped=False).fit(y), timeout)
        components[f"{days}d"] = model.model_["components"] if model is not None else f"{status}: {error}"
    y_one_day = contexts[1].to_numpy(dtype=float)
    zzn = AutoETS(season_length=SEASON, model="ZZN", damped=False).forecast(h=HORIZON, y=y_one_day)["mean"]
    diff = None if ag_one_day_mean is None else float(np.max(np.abs(ag_one_day_mean - zzn)))
    return {"components_zzz": components, "one_day_autogluon_vs_zzn_max_abs_diff": diff,
            f"fixed_models_{max(contexts)}d": fixed}


# Per model and context length: fit seconds over the ok fits, and the projected cost of a full test run (one fit per
# valid origin and context). A cell with a timeout or a skip has no mean: its fits take at least the timeout.
def _projection(timings: pd.DataFrame, n_origins: int, cores: int) -> pd.DataFrame:
    rows = []
    for (model, days), cell in timings.groupby(["model", "context_days"], sort=False):
        ok = cell[cell["status"] == "ok"]
        counts = {s: int((cell["status"] == s).sum()) for s in ("ok", "error", "timeout", "skipped")}
        slow = counts["timeout"] + counts["skipped"] > 0
        mean_s = ok["seconds"].mean() if len(ok) and not slow else np.nan
        note = "≥ timeout per fit" if slow else ("all fits failed" if not len(ok) else
                                                 (f"{counts['error']} errors" if counts["error"] else ""))
        cpu_hours = mean_s * n_origins / 3600
        rows.append({"model": model, "context_days": days, "mean_s": mean_s,
                     "max_s": ok["seconds"].max() if len(ok) else np.nan, **{f"n_{s}": n for s, n in counts.items()},
                     "n_fits_full_test": n_origins, "cpu_hours": cpu_hours,
                     "mac_wall_hours_lower_bound": cpu_hours / cores, "note": note})
    return pd.DataFrame(rows)


# Defaults of a statsforecast model's constructor (what we didn't set), as text so they fit in JSON.
def _defaults(cls) -> dict:
    return {k: repr(p.default) for k, p in inspect.signature(cls.__init__).parameters.items() if k != "self"}


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoGluon local-model timing spike (no network)")
    parser.add_argument("--snapshot", type=Path, required=True, help="snapshot folder (with manifest.json)")
    parser.add_argument("--out", type=Path, required=True, help="parent folder; a new <UTC>_autogluon_spike folder")
    parser.add_argument("--sensors", nargs="+", choices=list(TASKS), default=DEFAULT_SENSORS)
    parser.add_argument("--origins", type=int, default=2, help="the first N valid test origins per sensor")
    parser.add_argument("--timeout", type=int, default=180, help="seconds per fit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import joblib
    from statsforecast.models import AutoARIMA, AutoETS, Theta

    started = datetime.now(timezone.utc)
    run_dir = args.out / f"{started:%Y-%m-%dT%H%M%SZ}_autogluon_spike"
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} EXISTS; runs are never overwritten")
    signal.signal(signal.SIGALRM, _on_alarm)
    code = git_state()
    frames = load_snapshot(args.snapshot)
    manifest = json.loads((args.snapshot / MANIFEST_NAME).read_text())

    # Valid test origins over all tasks: the number of fits per context length in a full test run.
    series = {name: target_series(frames[name], task) for name, task in TASKS.items()}
    tables = {name: origin_table(series[name], task, "test") for name, task in TASKS.items()}
    n_origins = int(sum(table["valid"].sum() for table in tables.values()))
    cores = joblib.cpu_count(only_physical_cores=True)

    checks: dict = {"hyperparameter_keys": _check_keys(), "warmup_s": {}, "ets": {}, "origins": {}}
    rows, one_day_ets_mean = [], {}
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        for sensor in args.sensors:
            task = TASKS[sensor]
            origins = list(tables[sensor].loc[tables[sensor]["valid"], "origin"])[: args.origins]
            checks["origins"][sensor] = [o.isoformat() for o in origins]
            train = _tsdf({sensor: training_series(series[sensor], pd.Timestamp(task.test_start)).ffill().dropna()})
            for name, params in MODELS.items():
                predictor = _fit_predictor(name, params, train, tmp / f"{name}_{sensor}")
                warmup = context_window(series[sensor], origins[0], BUCKETS_PER_DAY)
                _, seconds, status, _ = _timed(lambda: predictor.predict(_tsdf({"warmup": warmup}), random_seed=0),
                                               args.timeout)
                checks["warmup_s"].setdefault(name, {})[sensor] = round(seconds, 3)
                too_slow = False
                for days in CONTEXT_DAYS:
                    for origin in origins:
                        context = context_window(series[sensor], origin, days * BUCKETS_PER_DAY)
                        row = {"model": name, "sensor": sensor, "origin": origin.isoformat(), "context_days": days,
                               "context_steps": len(context)}
                        if too_slow:
                            rows.append({**row, "seconds": np.nan, "status": "skipped", "error": ""})
                            continue
                        preds, seconds, status, error = _timed(
                            lambda: predictor.predict(_tsdf({origin.isoformat(): context}), random_seed=0),
                            args.timeout)
                        row.update({"seconds": round(seconds, 3), "status": status, "error": error})
                        if preds is not None:
                            row.update(_quantile_checks(preds))
                            if name == "AutoETS" and days == 1 and origin == origins[0]:
                                one_day_ets_mean[sensor] = preds["mean"].to_numpy(dtype=float)
                        rows.append(row)
                        logger.info(f"FIT {name} {sensor} {origin} {days}d: {status} in {seconds:.2f} s")
                    too_slow = too_slow or any(r["status"] == "timeout" for r in rows[-len(origins):])
            contexts = {days: context_window(series[sensor], origins[0], days * BUCKETS_PER_DAY)
                        for days in CONTEXT_DAYS}
            checks["ets"][sensor] = _ets_checks(contexts, one_day_ets_mean.get(sensor), args.timeout)
        first = args.sensors[0]
        first_train = _tsdf({first: training_series(series[first], pd.Timestamp(TASKS[first].test_start))
                             .ffill().dropna()})
        checks["fallback"] = _fallback_check(
            first_train, context_window(series[first], pd.Timestamp(checks["origins"][first][0]), BUCKETS_PER_DAY),
            tmp, args.timeout)

    timings = pd.DataFrame(rows)
    projection = _projection(timings, n_origins, cores)
    finished = datetime.now(timezone.utc)
    checks.update({
        "models": MODELS,
        "timeout_s": args.timeout,
        "statsforecast_defaults": {"AutoETS": _defaults(AutoETS), "AutoARIMA": _defaults(AutoARIMA),
                                   "Theta": _defaults(Theta)},
        "n_valid_test_origins_all_tasks": n_origins,
        "snapshot": {"id": manifest["snapshot_id"], "manifest_sha256": sha256_file(args.snapshot / MANIFEST_NAME)},
        "code": code,
        "versions": {"python": platform.python_version(), **{p: version(p) for p in PACKAGES}},
        "platform": {"system": platform.platform(), "machine": platform.machine(), "physical_cores": cores,
                     "logical_cores": joblib.cpu_count()},
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished_utc": finished.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_s": round((finished - started).total_seconds(), 1),
    })
    run_dir.mkdir(parents=True)
    timings.to_csv(run_dir / "timings.csv", index=False)
    projection.to_csv(run_dir / "projection.csv", index=False)
    (run_dir / "checks.json").write_text(json.dumps(checks, indent=2))
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(projection.round(3).to_string(index=False))
    logger.info(f"SPIKE WRITTEN to {run_dir}")


if __name__ == "__main__":
    main()
