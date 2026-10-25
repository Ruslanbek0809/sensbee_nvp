# Chronos-2 forecaster for the NVP service. It uses the setups chosen on the benchmark's validation periods
# and returns the forecast as deciles.
# Each request gets a recipe. The recipe depends on the SensBee sensors configured in the environment
# (<NAME>_SENSOR_UUID and <NAME>_MY_API_KEY_READ, with the names of benchmark/tasks.py):
# - "mask": a hall's visitors_total. Values-only Chronos-2, and every quantile is set to 0 where the hall's
#   opening profile is below THRESHOLD. The profile is learned from the context itself: the share of its days
#   with visitors, per Berlin weekday and 15-min slot. It is learned because the service doesn't know when a
#   season starts.
# - "stations": a weather station's temperature. Chronos-2 with the other configured stations' temperature as
#   past covariates, and the time of day and solar elevation as known covariates.
#   A station that can't be loaded is left out.
#   The other stations are private and are read with the server's keys, so this recipe is used only when the request
#   carries the station's own configured READ key. Any other caller gets "plain", so no caller's forecast depends on
#   sensors their key doesn't open.
# - "plain": any other sensor or column. Values-only Chronos-2.
# The inputs follow the benchmark:
# - The forecast origin is the current 15-min bucket. It is still filling, so it is dropped,
#   and the context ends with the last complete bucket.
# - One extra day is loaded before the context, so gaps at its start are filled from earlier data,
#   as the benchmark does (it fills them from the whole history).
# The model, the mask and the covariate window are the benchmark's own code (benchmark/models,
# benchmark/covariates.py), so the service runs what was evaluated.
# Quantiles are sorted (the benchmark sorts crossings for every model).
# They are then widened around the median by the recipe's factor (benchmark/bands.py), because Chronos-2's raw 80 %
# band held the truth only about 70-74 % of the time on validation. The median doesn't change.
# For event-based columns they are also clipped at 0, as the LLMTime path clips them. The benchmark scores them
# unclipped.
# torch and chronos-forecasting come from requirements-tsfm.txt and are loaded on the first request.

import hmac
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Optional

import httpx
import numpy as np
import pandas as pd

from benchmark import covariates as cov
from benchmark.bands import BAND_FACTORS, widen
from benchmark.models import CHRONOS2
from benchmark.models.chronos_models import chronos2_forecaster
from benchmark.models.mask_models import closed_steps, recent_mask_forecaster
from benchmark.snapshot import BUCKETS_PER_DAY
from benchmark.tasks import QUANTILE_LEVELS, STEP, TASKS
from src.data_access.data_loader import EVENT_BASED_COLUMNS
from src.data_access.sensbee_client import load_sensor_series_from_api

logger = logging.getLogger(__name__)

THRESHOLD = 0.3  # "0 when closed" threshold, selected on validation for both halls (thesis Phase 5 step 5.0)
RECIPES = {"visitors": "mask", "regular": "stations"}  # keyed by benchmark task kind; anything else is "plain"
# Extra days loaded before the context, so gap filling starts from data (as in the benchmark). Exact for visitor
# columns (their fill reaches back at most 2 h); for other columns it covers gaps at the context's start of up to a day.
MARGIN_DAYS = 1
# The history the recipes were evaluated with and the band factors fitted on. Other lengths work but aren't calibrated
# (model_info["evaluated_history"] says so).
EVALUATED_HISTORY_DAYS = 28

_model = None  # (forecaster, parameters) of the loaded Chronos-2
_lock = threading.Lock()  # one model call at a time, because requests run in worker threads


# A configured sensor's id (suffix "SENSOR_UUID") or READ key (suffix "MY_API_KEY_READ"), stripped; "" if unset.
def _configured(name: str, suffix: str) -> str:
    return (os.getenv(f"{name}_{suffix}") or "").strip()


# Returns the configured name of a SensBee sensor id, or None if no <NAME>_SENSOR_UUID holds it.
def _sensor_name(sensor_id: str) -> Optional[str]:
    for name in TASKS:
        if _configured(name, "SENSOR_UUID") == sensor_id:
            return name
    return None


# Returns the recipe for a sensor and column ("mask", "stations" or "plain"), plus the sensor's configured name.
# "stations" needs the station's configured READ key in the request (compared in constant time); otherwise "plain".
def recipe_for(sensor_id: str, column_name: str, api_key: Optional[str]) -> tuple[str, Optional[str]]:
    name = _sensor_name(sensor_id)
    if name is None or TASKS[name].column != column_name:
        return "plain", name
    recipe = RECIPES.get(TASKS[name].kind, "plain")
    if recipe == "stations":
        configured = _configured(name, "MY_API_KEY_READ")
        if not configured or not hmac.compare_digest((api_key or "").strip().encode(), configured.encode()):
            return "plain", name
    return recipe, name


# Returns the Chronos-2 forecaster (the benchmark adapter at the pinned revision). It is loaded on first use and cached.
def _chronos2() -> tuple:
    global _model
    if _model is None:
        try:
            _model = chronos2_forecaster(*CHRONOS2)
        except ImportError as exc:
            raise ValueError(f"CHRONOS-2 NOT INSTALLED (install requirements-tsfm.txt): {exc}") from exc
        logger.info(f"LOADED Chronos-2 {CHRONOS2[0]} @ {CHRONOS2[1]}")
    return _model


# Loads Chronos-2 now. The service calls it at startup when chronos-forecasting is installed, so no request waits for it.
def preload() -> None:
    with _lock:
        _chronos2()


# Loads one sensor column from SensBee over [origin - history_days - MARGIN_DAYS, now].
# It drops the still-filling bucket at the origin and keeps only the last history_days.
def _load(sensor_id: str, api_key: Optional[str], column_name: str, origin: pd.Timestamp, now: datetime,
          history_days: int) -> pd.Series:
    start = origin - timedelta(days=history_days + MARGIN_DAYS)
    series = load_sensor_series_from_api(sensor_id=sensor_id, api_key=api_key, column_name=column_name,
                                         from_time=start.to_pydatetime(), to_time=now)
    return series[series.index < origin].iloc[-history_days * BUCKETS_PER_DAY:]


# Builds the covariate window for a station's forecast, like the benchmark's "stations" set.
# - Time of day and solar elevation are known covariates.
# - The other configured stations' temperature is a past covariate (covariate_window blanks it over the horizon).
# loads holds the other configured stations' temperature loads, started together with the target's.
# Returns the window, the stations used, and the stations left out (not configured, no data, or HTTP error).
def _stations_window(name: str, context: pd.Series, origin: pd.Timestamp, horizon: int,
                     loads: dict[str, Future]) -> tuple[pd.DataFrame, list[str], list[str]]:
    index = context.index.append(pd.date_range(origin, periods=horizon, freq=STEP))
    columns = cov.sun_columns(index)
    kinds = {column: cov.GROUP_KINDS["sun"] for column in cov.SUN}
    used, dropped = [], []
    for station in cov.STATIONS:
        if station == name:
            continue
        if station not in loads:
            dropped.append(station)
            continue
        try:
            series = loads[station].result()
        except (ValueError, httpx.HTTPError) as exc:
            logger.warning(f"STATION {station} LEFT OUT of the covariates: {type(exc).__name__}")
            dropped.append(station)
            continue
        columns[f"temperature_{station}"] = series.reindex(index).to_numpy()
        kinds[f"temperature_{station}"] = cov.GROUP_KINDS["stations"]
        used.append(station)
    window = cov.covariate_window(pd.DataFrame(columns, index=index), kinds, origin, context.index)
    return window, used, dropped


# Forecasts one sensor column with Chronos-2 and returns deciles. The forecast is issued at `now` (naive UTC),
# which tests can pass in. Returns a dict with:
# - the forecast timestamps,
# - the deciles [horizon_steps, 9], sorted and clipped at 0 for event-based columns,
# - the context's size and last timestamp,
# - model_info (model, revision, recipe and its settings, band factor, origin, context points, whether the history is
#   the evaluated 28 days, and the seconds spent loading data from SensBee and in the model call, without waiting for
#   the lock).
def forecast_chronos2(sensor_id: str, api_key: Optional[str], column_name: str, horizon_steps: int,
                      history_days: int, now: Optional[datetime] = None) -> dict:
    started = perf_counter()
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    origin = pd.Timestamp(now).floor(STEP)
    recipe, name = recipe_for(sensor_id, column_name, api_key)
    # The stations recipe loads the other configured stations' temperature at the same time as the target.
    others = {} if recipe != "stations" else {
        station: _configured(station, "SENSOR_UUID") for station in cov.STATIONS
        if station != name and _configured(station, "SENSOR_UUID")}
    with ThreadPoolExecutor(max_workers=1 + len(others)) as pool:
        target = pool.submit(_load, sensor_id, api_key, column_name, origin, now, history_days)
        loads = {station: pool.submit(_load, station_id, _configured(station, "MY_API_KEY_READ") or None,
                                      "temperature", origin, now, history_days)
                 for station, station_id in others.items()}
        context = target.result()
        if len(context) < 2:
            raise ValueError(f"NOT ENOUGH DATA before {origin} for sensor {sensor_id} ({len(context)} points).")
        timestamps = pd.date_range(origin, periods=horizon_steps, freq=STEP)
        info = {"model_id": CHRONOS2[0], "revision": CHRONOS2[1], "recipe": recipe, "origin": origin.isoformat(),
                "context_points": len(context), "quantile_levels": list(QUANTILE_LEVELS),
                "evaluated_history": history_days == EVALUATED_HISTORY_DAYS}
        window = None
        if recipe == "stations":
            window, info["covariate_stations"], info["stations_left_out"] = _stations_window(
                name, context, origin, horizon_steps, loads)

    loaded = perf_counter()
    with _lock:
        model_started = perf_counter()
        base, _ = _chronos2()
        model = recent_mask_forecaster(lambda: (base, {}), THRESHOLD, "chronos2")[0] if recipe == "mask" else base
        extra = {"covariates": [window]} if recipe == "stations" else {}
        quantiles, _ = model([context], horizon_steps, QUANTILE_LEVELS, 0, **extra)
    info["seconds"] = {"data": round(loaded - started, 3), "model": round(perf_counter() - model_started, 3)}
    if recipe == "mask":
        info.update({"threshold": THRESHOLD,
                     "masked_steps": int(closed_steps(context, horizon_steps, THRESHOLD).sum())})

    quantiles = np.asarray(quantiles, dtype=float)[0]
    if quantiles.shape != (horizon_steps, len(QUANTILE_LEVELS)) or not np.isfinite(quantiles).all():
        raise RuntimeError(f"CHRONOS-2 RETURNED AN INVALID FORECAST (shape {quantiles.shape}).")
    factor = BAND_FACTORS[recipe]
    quantiles = widen(quantiles, QUANTILE_LEVELS, factor)  # sorts, then widens
    info["band_factor"] = factor
    if column_name in EVENT_BASED_COLUMNS:
        quantiles = np.maximum(quantiles, 0.0)
    logger.info(f"CHRONOS-2 FORECAST {recipe} for {column_name}: {len(context)} PTS → {horizon_steps} steps")
    return {"timestamps": [t.isoformat() for t in timestamps], "quantiles": quantiles, "input_points": len(context),
            "last_data_timestamp": context.index[-1].isoformat(), "model_info": info}
