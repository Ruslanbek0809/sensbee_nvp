# Tests for benchmark/tasks.py on synthetic rows (no snapshot, no network): the fill rule of each sensor kind, equality
# with the service loader (only SensbeeClient.get_sensor_data is patched, with autospec), origins and periods, the
# validity reasons, the scale's exclusions, and that nothing at or after an origin reaches its context or scale.

import dataclasses
import math
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from benchmark.metrics import seasonal_scale
from benchmark.tasks import (
    GUARD, TASKS, SensorTask, candidate_origins, context_window, origin_table, period_bounds, scale_history,
    target_series, target_window, training_series,
)
from src.data_access.sensbee_client import SensbeeClient, load_sensor_series_from_api

EPOCH = datetime(1970, 1, 1)
EVENT_ROWS = [("2026-02-10 10:05", 5.0), ("2026-02-10 10:10", 7.0), ("2026-02-10 10:50", -2.0),
              ("2026-02-10 11:05", 4.0), ("2026-02-10 14:10", 9.0)]


# Raw snapshot-like rows from (time, value) pairs.
def _raw(column: str, rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame({"created_at": pd.to_datetime([t for t, _ in rows]), column: [v for _, v in rows]})


# 60 days of 15-min weather-like rows from 2026-01-01 with two planted gaps: the bucket 2026-02-10 12:00 and the whole
# day 2026-02-15.
def _weather_with_gaps() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", "2026-03-02", freq="15min", inclusive="left")
    values = 10 + 5 * np.sin(np.arange(len(times)) * 2 * np.pi / 96)
    keep = (times != pd.Timestamp("2026-02-10 12:00")) & ~((times >= "2026-02-15") & (times < "2026-02-16"))
    return pd.DataFrame({"created_at": times[keep], "temperature": values[keep]})


# Fake SensBee /data/load with time_grouping over (time, value) rows: one record per bucket
# floor(epoch / interval) * interval with rows inside the inclusive [from, to], aggregated with MAX or AVG. Written
# with epoch arithmetic, independent of the pandas code under test.
def _fake_grouped(rows: list[tuple[str, float]], aggregation: str):
    def get_sensor_data(self, sensor_id, api_key=None, from_time=None, to_time=None, limit=None,
                        ordering="ASC", cols=None, time_grouping_seconds=None) -> list[dict]:
        column = cols[0].split(".")[0]
        groups: dict[datetime, list[float]] = {}
        for t, value in rows:
            moment = datetime.fromisoformat(t)
            if from_time <= moment <= to_time:
                seconds = (moment - EPOCH).total_seconds()
                label = EPOCH + timedelta(seconds=math.floor(seconds / time_grouping_seconds) * time_grouping_seconds)
                groups.setdefault(label, []).append(value)
        return [{"grouped_time": label.strftime("%Y-%m-%dT%H:%M:%S"),
                 column: max(values) if aggregation == "MAX" else sum(values) / len(values)}
                for label, values in sorted(groups.items())]
    return get_sensor_data


def test_target_series_applies_the_fill_rule_of_each_kind():
    index = pd.date_range("2026-02-10 10:00", "2026-02-10 14:00", freq="15min")

    visitors = target_series(_raw("v", EVENT_ROWS), SensorTask("v", "MAX", "visitors", "2026-03-01", "2026-03-02"))
    assert list(visitors.index) == list(index)
    # 10:00 = MAX(5, 7); 10:15-10:30 filled; 10:45 = -2 clipped to 0; 11:00 = 4; 11:15-13:00 filled (8 buckets =
    # 2 h); 13:15-13:45 closed → 0; 14:00 = 9.
    assert visitors.tolist() == [7, 7, 7, 0, 4] + [4] * 8 + [0] * 3 + [9]

    counter = target_series(_raw("c", EVENT_ROWS), SensorTask("c", "MAX", "counter", "2026-03-01", "2026-03-02"))
    assert counter.tolist() == [7, 7, 7, -2, 4] + [4] * 11 + [9]  # no limit, no clip

    weather = target_series(_raw("t", EVENT_ROWS), SensorTask("t", "AVG", "regular", "2026-03-01", "2026-03-02"))
    assert weather.iloc[0] == 6.0  # AVG(5, 7)
    assert int(weather.isna().sum()) == 2 + 11  # 10:15-10:30 and 11:15-13:45 stay missing


@pytest.mark.parametrize("column, aggregation, kind", [("visitors_total", "MAX", "visitors"),
                                                       ("temperature", "AVG", "regular")])
def test_contexts_equal_what_the_service_loader_returns(column, aggregation, kind):
    # A 30-min gap, a negative value and a 2.5-h gap (a closure for visitors). The request window starts at an observed
    # bucket (10:00) and ends with a complete one (14:45), where the service's per-window fill and ours must agree.
    rows = [("2026-02-10 09:50", 3.0), ("2026-02-10 10:02", 8.0), ("2026-02-10 10:09", 11.0),
            ("2026-02-10 10:22", 9.0), ("2026-02-10 11:07", -1.0), ("2026-02-10 11:31", 6.0),
            ("2026-02-10 14:40", 12.0), ("2026-02-10 15:05", 10.0)]
    with patch.object(SensbeeClient, "get_sensor_data", autospec=True, side_effect=_fake_grouped(rows, aggregation)):
        service = load_sensor_series_from_api(sensor_id="sensor-uuid", api_key="read-key", column_name=column,
                                              from_time=datetime(2026, 2, 10, 10, 0),
                                              to_time=datetime(2026, 2, 10, 14, 59, 59))

    task = SensorTask(column, aggregation, kind, "2026-03-01", "2026-03-02")
    ours = context_window(target_series(_raw(column, rows), task), pd.Timestamp("2026-02-10 15:00"), len(service))
    assert len(service) == 20
    assert list(ours.index) == list(service.index)
    assert ours.tolist() == pytest.approx(service.tolist())


def test_origins_every_6h_inside_each_period():
    task = TASKS["MANEBACH_WEATHER_STATION"]
    test = candidate_origins(task, "test")
    assert len(test) == 221
    assert test[0] == pd.Timestamp("2026-08-06") and test[-1] == pd.Timestamp("2026-09-30")
    assert set(test.hour) == {0, 6, 12, 18}

    validation = candidate_origins(task, "validation")
    assert validation[0] == pd.Timestamp("2026-07-09")
    assert validation[-1] + GUARD == pd.Timestamp("2026-08-06")  # validation windows end where the test starts


def test_periods_are_56_and_28_days_and_do_not_overlap():
    for name, task in TASKS.items():
        validation_start, validation_end = period_bounds(task, "validation")
        test_start, test_end = period_bounds(task, "test")
        assert test_end - test_start == pd.Timedelta(days=56), name
        assert validation_end == test_start, name
        assert validation_end - validation_start == pd.Timedelta(days=28), name
        assert candidate_origins(task, "test")[-1] + GUARD == test_end, name


def test_origin_validity_reasons():
    task = SensorTask("temperature", "AVG", "regular", "2026-02-01", "2026-03-01",
                      exclusions=(("2026-02-25 10:00", "2026-02-25 11:00", "test outage"),))
    series = target_series(_weather_with_gaps(), task)
    table = origin_table(series, task, "test").set_index("origin")
    reason = table["reason"]

    assert reason[pd.Timestamp("2026-02-05 00:00")] == "ok"
    assert reason[pd.Timestamp("2026-02-10 06:00")] == "target_gap"  # 02-10 12:00 lies in the next 24 h
    assert reason[pd.Timestamp("2026-02-11 00:00")] == "recent_gap"  # ... in the previous 24 h
    low = table.loc[pd.Timestamp("2026-02-20 00:00")]
    assert low["reason"] == "low_coverage"  # 02-15 is missing: 6 of 7 days observed
    assert low["coverage_7d"] == round(6 / 7, 4) and low["coverage_28d"] >= 0.9
    assert reason[pd.Timestamp("2026-02-25 00:00")] == "excluded"
    last = table.loc[pd.Timestamp("2026-02-28 00:00")]
    assert last["reason"] == "ok" and bool(last["exclusion_in_context"])

    # The first validation origin (2026-01-04) would need 28 days before the series' first bucket.
    assert origin_table(series, task, "validation").iloc[0]["reason"] == "short_history"


def test_scale_ignores_excluded_buckets():
    # A counter rising by 1 per bucket has |y_t - y_{t-96}| = 96 everywhere; zeros from an outage inflate the scale
    # unless the outage is excluded.
    times = pd.date_range("2026-01-01", "2026-02-01", freq="15min", inclusive="left")
    values = np.arange(len(times), dtype=float)
    values[(times >= "2026-01-20 08:00") & (times < "2026-01-20 12:00")] = 0.0
    raw = pd.DataFrame({"created_at": times, "c": values})
    origin = pd.Timestamp("2026-01-30")
    plain = SensorTask("c", "AVG", "counter", "2026-01-30", "2026-01-31")
    excluded = dataclasses.replace(plain, exclusions=(("2026-01-20 08:00", "2026-01-20 12:00", "test outage"),))

    assert seasonal_scale(scale_history(target_series(raw, plain), origin, plain), 96) > 96
    assert seasonal_scale(scale_history(target_series(raw, excluded), origin, excluded), 96) == 96.0


@pytest.mark.parametrize("kind", ["regular", "visitors", "counter"])
def test_nothing_at_or_after_the_origin_reaches_context_or_scale(kind):
    origin = pd.Timestamp("2026-02-20 06:00")
    raw = _weather_with_gaps()
    # A gap from 1 h before to 1 h after the origin: a fill that looked ahead (e.g. interpolation) would pull the
    # first value after the gap into the context.
    raw = raw[(raw["created_at"] < origin - pd.Timedelta("1h")) | (raw["created_at"] >= origin + pd.Timedelta("1h"))]
    task = SensorTask("temperature", "MAX", kind, "2026-02-01", "2026-03-01")
    future = raw["created_at"] >= origin
    changed = raw.assign(temperature=np.where(future, raw["temperature"] + 1000, raw["temperature"]))
    series, changed_series = target_series(raw, task), target_series(changed, task)

    for steps in (96, 672, 2688, None):
        context = context_window(series, origin, steps)
        assert context.index[-1] == origin - pd.Timedelta("15min")
        assert not context.isna().any()
        pd.testing.assert_series_equal(context, context_window(changed_series, origin, steps))
    np.testing.assert_array_equal(scale_history(series, origin, task), scale_history(changed_series, origin, task))
    assert training_series(series, origin).index[-1] == origin - pd.Timedelta("15min")

    target = target_window(series, origin)
    assert target.index[0] == origin and len(target) == 96
    # The change is real, just not visible before the origin (equal_nan: weather keeps its gap in the target).
    assert not np.allclose(target, target_window(changed_series, origin), equal_nan=True)
