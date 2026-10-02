# Benchmark tasks on a frozen snapshot: the 15-min target series of each sensor, its validation and test periods,
# forecast origins every 6 h with a validity rule (gaps, outages), and the context and target windows.
#
# Targets are built the way the service builds them (src/data_access/sensbee_client.py:145-161): SensBee's AVG/MAX
# bucket, then for visitor counts ffill up to 2 h, 0 after that and clip at 0; the parking counter is forward-filled.
# Weather targets are never filled: an origin whose window has a gap is invalid, and only contexts are forward-filled.
# Every fill runs forward in time and contexts are cut at the origin before filling, so no value at or after an origin
# can reach its context. Timestamps are naive UTC.

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from benchmark.snapshot import BUCKET_FREQ, BUCKETS_PER_DAY, bucket_15min
from src.data_access.data_loader import CLOSURE_GAP_MINUTES, fill_event_series

STEP = pd.Timedelta(BUCKET_FREQ)
BUCKET_MINUTES = int(STEP / pd.Timedelta("1min"))
HORIZON = BUCKETS_PER_DAY                  # 96 steps = 24 h
REPORT_STEPS = (4, 24, 96)                 # metrics on the first 1 h, 6 h and 24 h of each forecast
CONTEXT_DAYS = (1, 7, 14, 28)
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ORIGIN_FREQ = "6h"                         # origins at 00/06/12/18 UTC
GUARD = HORIZON * STEP                     # ± around an origin: complete for weather, free of exclusions for all
MIN_CONTEXT_COVERAGE = 0.9                 # weather: observed share required in every context window
SEASON = BUCKETS_PER_DAY                   # daily season of the MASE/SQL scale and of seasonal naive
SCALE_DAYS = 28                            # history before an origin that the MASE/SQL scale is computed on
VALIDATION_DAYS = 28
CLOSURE_GAP_BUCKETS = CLOSURE_GAP_MINUTES // BUCKET_MINUTES  # visitors: ffill up to 120 min, then 0 (recorded in run.json)


# One sensor's forecasting task. kind: "regular" (weather; gaps stay NaN), "visitors" (event counter that resets
# overnight; service fill rule) or "counter" (drifting entry/exit counter; forward-filled). Periods are naive UTC;
# validation is the VALIDATION_DAYS before test_start. exclusions: (start, end, reason), [start, end), naive UTC.
@dataclass(frozen=True)
class SensorTask:
    column: str
    aggregation: str
    kind: str
    test_start: str
    test_end: str
    exclusions: tuple[tuple[str, str, str], ...] = ()
    secondary: bool = False


# Periods where event sensors were silent because of an outage, not because nothing happened. Found by listing the
# silences > 2 h that start during opening hours (Oct 2025–May 2026); every other one was a regular pause or an
# early closing.
STREAM_OUTAGE = ("2026-04-13 08:30", "2026-04-14 15:00",
                 "visitor stream and parking silent ~30 h from Mon 10:41 local; the weather stations kept reporting")
PARKING_SILENCE = ("2026-04-02 09:45", "2026-04-04 08:45",
                   "parking silent 46.8 h from Maundy Thursday 11:53 local while the visitor stream was alive "
                   "(holiday or parking-only outage; excluded to be safe)")

# Test = 56 days: the last 8 weeks of the snapshot for weather, of the season for the halls (last open day: Eishalle
# 28 Mar, Schwimmhalle 13 May 2026) and before the parking sensor stopped (17 Apr 2026). Parking is secondary: a
# drifting entry/exit counter, reported on its own and left out of the cross-sensor aggregate.
TASKS: dict[str, SensorTask] = {
    "MANEBACH_WEATHER_STATION": SensorTask("temperature", "AVG", "regular", "2026-08-06", "2026-10-01"),
    "STUTZERBACH_WEATHER_STATION": SensorTask("temperature", "AVG", "regular", "2026-08-06", "2026-10-01"),
    "FRAUENWALD_WEATHER_STATION": SensorTask("temperature", "AVG", "regular", "2026-08-06", "2026-10-01"),
    "EISHALLE": SensorTask("visitors_total", "MAX", "visitors", "2026-02-01", "2026-03-29", (STREAM_OUTAGE,)),
    "SCHWIMMHALLE": SensorTask("visitors_total", "MAX", "visitors", "2026-03-19", "2026-05-14", (STREAM_OUTAGE,)),
    "PARKPLATZ_EAZ": SensorTask("Fahrzeuge_auf_Parkplatz", "AVG", "counter", "2026-02-20", "2026-04-17",
                                (STREAM_OUTAGE, PARKING_SILENCE), secondary=True),
}


# Returns [start, end) of a task's "test" or "validation" period.
def period_bounds(task: SensorTask, period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    test_start = pd.Timestamp(task.test_start)
    if period == "test":
        return test_start, pd.Timestamp(task.test_end)
    if period == "validation":
        return test_start - pd.Timedelta(days=VALIDATION_DAYS), test_start
    raise ValueError(f"UNKNOWN PERIOD {period}")


# Builds a sensor's 15-min target on the complete UTC grid from its raw snapshot rows: SensBee's bucket, then the fill
# rule of its kind. Weather keeps its gaps as NaN.
def target_series(raw: pd.DataFrame, task: SensorTask) -> pd.Series:
    buckets = bucket_15min(raw, task.column, task.aggregation)
    series = buckets.reindex(pd.date_range(buckets.index[0], buckets.index[-1], freq=BUCKET_FREQ))
    if task.kind == "visitors":
        series = fill_event_series(series, BUCKET_MINUTES)
    elif task.kind == "counter":
        series = series.ffill()
    return series


# Candidate origins of a period: every ORIGIN_FREQ from its start while the 24-h target window still ends inside it.
def candidate_origins(task: SensorTask, period: str) -> pd.DatetimeIndex:
    start, end = period_bounds(task, period)
    return pd.date_range(start, end - GUARD, freq=ORIGIN_FREQ)


# Share of observed buckets in [start, end), from the cumulative count of observed buckets on the series grid that
# starts at `first`. Buckets outside the series count as missing.
def _observed_share(cumulative: np.ndarray, first: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> float:
    n = len(cumulative) - 1
    a = min(max((start - first) // STEP, 0), n)
    b = min(max((end - first) // STEP, 0), n)
    return float(cumulative[b] - cumulative[a]) / ((end - start) // STEP)


# Applies the validity rule to every candidate origin of a period. An origin is valid when no listed exclusion
# overlaps [origin - 24 h, origin + 24 h), the longest context lies inside the series, both 24-h windows around the
# origin are complete, and every context window is at least MIN_CONTEXT_COVERAGE observed. Filled kinds (visitors,
# counter) have no gaps, so only the exclusions and the series bounds apply to them. Returns one row per candidate:
# the first failing reason, the observed share of each context window, and whether an exclusion lies inside the
# longest context (for a sensitivity check).
def origin_table(series: pd.Series, task: SensorTask, period: str) -> pd.DataFrame:
    cumulative = np.concatenate([[0], np.cumsum(series.notna().to_numpy())])
    first = series.index[0]
    exclusions = [(pd.Timestamp(start), pd.Timestamp(end)) for start, end, _ in task.exclusions]
    longest = pd.Timedelta(days=max(CONTEXT_DAYS))
    rows = []
    for origin in candidate_origins(task, period):
        coverage = {days: _observed_share(cumulative, first, origin - pd.Timedelta(days=days), origin)
                    for days in CONTEXT_DAYS}
        if any(start < origin + GUARD and end > origin - GUARD for start, end in exclusions):
            reason = "excluded"
        elif origin - longest < first:
            reason = "short_history"
        elif _observed_share(cumulative, first, origin, origin + GUARD) < 1:
            reason = "target_gap"
        elif _observed_share(cumulative, first, origin - GUARD, origin) < 1:
            reason = "recent_gap"
        elif min(coverage.values()) < MIN_CONTEXT_COVERAGE:
            reason = "low_coverage"
        else:
            reason = "ok"
        rows.append({
            "origin": origin,
            "valid": reason == "ok",
            "reason": reason,
            **{f"coverage_{days}d": round(share, 4) for days, share in coverage.items()},
            "exclusion_in_context": any(start < origin and end > origin - longest for start, end in exclusions),
        })
    return pd.DataFrame(rows)


# The context of one origin: the series strictly before the origin, forward-filled, its last `steps` buckets
# (everything when steps is None). It is cut at the origin before filling, so nothing at or after it can enter.
def context_window(series: pd.Series, origin: pd.Timestamp, steps: Optional[int]) -> pd.Series:
    past = series.loc[: origin - STEP].ffill().dropna()
    return past if steps is None else past.iloc[-steps:]


# The HORIZON target values from the origin on, NaN where the series has none.
def target_window(series: pd.Series, origin: pd.Timestamp) -> pd.Series:
    return series.reindex(pd.date_range(origin, periods=HORIZON, freq=BUCKET_FREQ))


# The values the MASE/SQL scale of an origin is computed on: the SCALE_DAYS before it (unfilled for weather), with the
# task's exclusions set to NaN so an outage can't distort the scale.
def scale_history(series: pd.Series, origin: pd.Timestamp, task: SensorTask) -> np.ndarray:
    history = series.reindex(pd.date_range(origin - pd.Timedelta(days=SCALE_DAYS), origin, freq=BUCKET_FREQ,
                                           inclusive="left"))
    for start, end, _ in task.exclusions:
        history[(history.index >= pd.Timestamp(start)) & (history.index < pd.Timestamp(end))] = np.nan
    return history.to_numpy(dtype=float)


# Training data for a fitted model: the series strictly before the cutoff. When one model is fitted or tuned on
# several sensors, the cutoff is the earliest test start among them, because the periods of different sensors overlap
# (e.g. Schwimmhalle's validation lies inside Eishalle's test, and both share one event stream).
def training_series(series: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
    return series.loc[: pd.Timestamp(cutoff) - STEP]
