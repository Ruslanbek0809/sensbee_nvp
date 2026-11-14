# Tests for benchmark/tasks.py on synthetic rows (no snapshot, no network):
# - the fill rule of each sensor kind
# - equality with the service loader (only SensbeeClient.get_sensor_data is patched, with autospec)
# - origins and periods, and the validity reasons
# - the scale's exclusions
# - nothing at or after an origin reaches its context or scale
# - the training items of fitted models (gap splits, minimum length, train_start, exclusions)
# - the task sets: the October set unchanged; the January set's hall rules (P1: reopening day, season end, window,
#   training cut at the window start without the closed season); the external set (6a.12 X3: the DWD trio on the
#   SensBee weather windows) and each set's weather stations

import dataclasses
import math
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from benchmark.metrics import seasonal_scale
from benchmark.tasks import (
    EXTERNAL_TASKS, GUARD, STREAM_OUTAGE, TASKS, SensorTask, candidate_origins, context_window, january_tasks,
    open_days, origin_table, period_bounds, reopening_day, scale_history, season_end, set_stations, target_series,
    target_window, task_set, training_items, training_series,
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


# A longer context passed to origin_table is checked like the default ones: a week-long gap 6–7 weeks before the
# origin leaves every default window complete but the 56-day window only 49/56 observed. A 56-day context also needs
# more history.
def test_longer_contexts_are_checked_only_when_passed():
    times = pd.date_range("2026-01-01", "2026-03-02", freq="15min", inclusive="left")
    keep = ~((times >= "2026-01-05") & (times < "2026-01-12"))
    raw = pd.DataFrame({"created_at": times[keep], "temperature": np.full(int(keep.sum()), 10.0)})
    task = SensorTask("temperature", "AVG", "regular", "2026-02-01", "2026-03-01")
    series = target_series(raw, task)
    default = origin_table(series, task, "test").set_index("origin")
    longer = origin_table(series, task, "test", (1, 7, 14, 28, 56)).set_index("origin")
    origin = pd.Timestamp("2026-02-28 00:00")

    assert default.loc[origin, "reason"] == "ok" and "coverage_56d" not in default.columns
    assert longer.loc[origin, "reason"] == "low_coverage"
    assert longer.loc[origin, "coverage_56d"] == round(49 / 56, 4)
    assert longer.loc[pd.Timestamp("2026-02-05 00:00"), "reason"] == "short_history"  # 56 days start in Dec 2025
    assert default.loc[pd.Timestamp("2026-02-05 00:00"), "reason"] == "low_coverage"  # its 28 days hold the gap


def test_scale_ignores_excluded_buckets():
    # A counter rising by 1 per bucket has |y_t - y_{t-96}| = 96 everywhere. Zeros from an outage inflate the scale
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


# The 60 days of _weather_with_gaps() plus a 3-h gap on 2026-01-10 (filled) and a 2-day gap from
# 2026-01-20 (a split), cut at 2026-02-25: pieces 2026-01-01 → 01-20 (19 d) and 01-22 → 02-25 (34 d).
def test_training_items_split_long_gaps_and_fill_short_ones():
    raw = _weather_with_gaps()
    times = raw["created_at"]
    short = (times >= "2026-01-10 06:00") & (times < "2026-01-10 09:00")
    long = (times >= "2026-01-20") & (times < "2026-01-22")
    raw = raw[~short & ~long]
    task = SensorTask("temperature", "AVG", "regular", "2026-02-25", "2026-03-01")
    series = target_series(raw, task)
    items = training_items(series, task, pd.Timestamp("2026-02-25"))

    assert [(str(i.index[0]), str(i.index[-1])) for i in items] == [
        ("2026-01-01 00:00:00", "2026-01-19 23:45:00"), ("2026-01-22 00:00:00", "2026-02-24 23:45:00")]
    for item in items:
        assert not item.isna().any()
        assert (item.index.to_series().diff().dropna() == pd.Timedelta("15min")).all()  # complete grid
    # The 3-h gap is forward-filled with the last value before it; the planted 1-bucket and 1-day gaps (2026-02-10,
    # 2026-02-15) are inside the second piece and filled the same way.
    gap = items[0].loc["2026-01-10 06:00":"2026-01-10 08:45"]
    assert (gap == series.loc["2026-01-10 05:45"]).all()
    assert items[1].loc["2026-02-15"].nunique() == 1


def test_training_items_drop_short_pieces_and_respect_train_start_and_exclusions():
    times = pd.date_range("2025-12-01", "2026-03-01", freq="15min", inclusive="left")
    raw = pd.DataFrame({"created_at": times, "c": np.arange(len(times), dtype=float)})
    # A 1-h outage on 2026-01-15 splits even though it is short; train_start cuts December; the piece 2026-02-20 →
    # 02-25 after a 2-day gap is shorter than 14 days and dropped.
    raw = raw[(raw["created_at"] < "2026-02-18") | (raw["created_at"] >= "2026-02-20")]
    task = SensorTask("c", "AVG", "regular", "2026-02-25", "2026-03-01",
                      exclusions=(("2026-01-15 10:00", "2026-01-15 11:00", "test outage"),), train_start="2026-01-01")
    items = training_items(target_series(raw, task), task, pd.Timestamp("2026-02-25"))

    assert [(str(i.index[0]), str(i.index[-1])) for i in items] == [
        ("2026-01-01 00:00:00", "2026-01-15 09:45:00"), ("2026-01-15 11:00:00", "2026-02-17 23:45:00")]


@pytest.mark.parametrize("kind", ["regular", "visitors", "counter"])
def test_training_items_never_reach_the_cutoff(kind):
    raw = _weather_with_gaps()
    cutoff = pd.Timestamp("2026-02-20 06:00")
    task = SensorTask("temperature", "MAX", kind, "2026-02-21", "2026-03-01")
    changed = raw.assign(temperature=np.where(raw["created_at"] >= cutoff, raw["temperature"] + 1000,
                                              raw["temperature"]))
    items = training_items(target_series(raw, task), task, cutoff)
    changed_items = training_items(target_series(changed, task), task, cutoff)

    assert items[-1].index[-1] == cutoff - pd.Timedelta("15min")
    assert len(items) == len(changed_items)
    for a, b in zip(items, changed_items):
        pd.testing.assert_series_equal(a, b)


def test_hall_training_starts_at_the_season_start():
    assert TASKS["EISHALLE"].train_start == TASKS["SCHWIMMHALLE"].train_start == "2025-10-10"
    assert all(task.train_start is None for name, task in TASKS.items() if task.kind != "visitors")


# Task sets (6a.11): the January set from a snapshot, with the hall rules of protocol P1 (plan
# docs/plans/2026-11-14-phase-6a-11-task-sets-january-rules.md, decisions D1-D10). The synthetic halls of conftest.py
# carry the October snapshot's stray days (2025-07-08, 2025-10-10, 2026-09-08, 09-17, 09-29) and late-May trickle days.
JANUARY_UNTIL = pd.Timestamp("2027-01-10")
WEATHER = ("MANEBACH_WEATHER_STATION", "STUTZERBACH_WEATHER_STATION", "FRAUENWALD_WEATHER_STATION")


# Open days (>= 16 nonzero buckets on a UTC date) of a synthetic hall.
def _open(hall_raw, counts, **kwargs) -> pd.Series:
    return open_days(target_series(hall_raw(counts, **kwargs), TASKS["SCHWIMMHALLE"]))


# The January set over the two synthetic halls (Schwimmhalle optionally reopening on `reopen`).
def _january(hall_raw, hall_days, reopen=None):
    frames = {"EISHALLE": hall_raw(hall_days("ice")), "SCHWIMMHALLE": hall_raw(hall_days("swim", reopen))}
    return january_tasks(frames, JANUARY_UNTIL)


def test_the_reopening_day_skips_the_stray_days(hall_raw, hall_days):
    swim, ice = _open(hall_raw, hall_days("swim")), _open(hall_raw, hall_days("ice"))
    june_2025, june_2026 = pd.Timestamp("2025-06-01"), pd.Timestamp("2026-06-01")

    assert reopening_day(swim, june_2026) == pd.Timestamp("2026-09-19")  # not 09-08, 09-17 or the lone 08-10
    assert reopening_day(ice, june_2026) is None                         # one count on 09-29 is no reopening
    # Not Eishalle's lone full test day 2025-07-08, nor the 9-bucket 2025-10-10.
    assert reopening_day(swim, june_2025) == reopening_day(ice, june_2025) == pd.Timestamp("2025-10-11")
    # The 7-day span must lie inside the data: with data to 09-23 only 5 days from 09-19 exist.
    assert reopening_day(_open(hall_raw, hall_days("swim"), until="2026-09-24"), june_2026) is None
    full_week = _open(hall_raw, hall_days("swim"), until="2026-09-26")
    assert reopening_day(full_week, june_2026) == pd.Timestamp("2026-09-19")


def test_the_season_ends_on_the_last_open_day_that_ends_a_full_week(hall_raw, hall_days):
    june_2026 = pd.Timestamp("2026-06-01")
    # Not 2026-05-29 (16 buckets) after the trickle days of late May.
    assert season_end(_open(hall_raw, hall_days("swim")), june_2026) == pd.Timestamp("2026-05-16")
    assert season_end(_open(hall_raw, hall_days("ice")), june_2026) == pd.Timestamp("2026-03-28")


def test_the_january_set_from_a_snapshot(hall_raw, hall_days):
    tasks, notes = _january(hall_raw, hall_days)

    assert sorted(tasks) == sorted([*WEATHER, "SCHWIMMHALLE"])  # parking excluded, Eishalle not reopened
    for name in WEATHER:
        assert tasks[name] == dataclasses.replace(TASKS[name], test_start="2026-10-01", test_end="2027-01-10")
    swim = tasks["SCHWIMMHALLE"]
    assert (swim.column, swim.aggregation, swim.kind) == ("visitors_total", "MAX", "visitors")
    assert (swim.train_start, swim.season_start, swim.test_start, swim.test_end) == (
        "2025-10-11", "2026-09-19", "2026-10-17", "2027-01-10")
    assert swim.exclusions == (STREAM_OUTAGE, ("2026-05-17", "2026-09-19", "closed season (P1)"))
    assert notes["SCHWIMMHALLE"]["status"] == "ok"
    assert notes["SCHWIMMHALLE"]["reopenings"] == ["2025-10-11", "2026-09-19"]
    assert notes["SCHWIMMHALLE"]["season_ends"] == ["2026-05-16"]
    assert notes["EISHALLE"]["status"] == "not reopened"
    assert notes["PARKPLATZ_EAZ"]["status"] == "excluded"
    assert period_bounds(swim, "test") == (pd.Timestamp("2026-10-17"), JANUARY_UNTIL)


# Window start = reopening + 28 days, but not before 2026-10-01; a window shorter than 14 days is not scored.
@pytest.mark.parametrize("reopen, start", [
    ("2026-08-20", "2026-10-01"), ("2026-11-20", "2026-12-18"), ("2026-11-29", "2026-12-27"),
    ("2026-11-30", None), ("2026-12-20", None)])
def test_a_hall_window_starts_28_days_after_reopening_and_needs_14_days(hall_raw, hall_days, reopen, start):
    tasks, notes = _january(hall_raw, hall_days, reopen)
    if start is None:
        assert "SCHWIMMHALLE" not in tasks and notes["SCHWIMMHALLE"]["status"] == "too short"
    else:
        assert tasks["SCHWIMMHALLE"].test_start == start and tasks["SCHWIMMHALLE"].season_start == reopen


def test_no_january_hall_training_row_at_or_after_the_cutoff(hall_raw, hall_days):
    tasks, _ = _january(hall_raw, hall_days)
    task = tasks["SCHWIMMHALLE"]
    cutoff = period_bounds(task, "test")[0]
    raw = hall_raw(hall_days("swim"))
    changed = raw.assign(visitors_total=np.where(raw["created_at"] >= cutoff, 99.0, raw["visitors_total"]))
    items = training_items(target_series(raw, task), task, cutoff)
    changed_items = training_items(target_series(changed, task), task, cutoff)

    assert items[-1].index[-1] == cutoff - pd.Timedelta("15min")
    assert len(items) == len(changed_items)
    for a, b in zip(items, changed_items):
        pd.testing.assert_series_equal(a, b)


def test_no_closed_season_row_in_a_january_hall_training_item(hall_raw, hall_days):
    tasks, _ = _january(hall_raw, hall_days)
    task = tasks["SCHWIMMHALLE"]
    items = training_items(target_series(hall_raw(hall_days("swim")), task), task, period_bounds(task, "test")[0])
    stamps = pd.DatetimeIndex(np.concatenate([item.index for item in items]))

    # The summer with its trickle and stray days (2026-05-17 .. 09-18) and everything before the first reopening
    # (the 2025-07-08 test day, 2025-10-10) are left out; the outage of April 2026 splits the old season.
    assert not ((stamps >= pd.Timestamp("2026-05-17")) & (stamps < pd.Timestamp("2026-09-19"))).any()
    assert [(str(i.index[0]), str(i.index[-1])) for i in items] == [
        ("2025-10-11 00:00:00", "2026-04-13 08:15:00"), ("2026-04-14 15:00:00", "2026-05-16 23:45:00"),
        ("2026-09-19 00:00:00", "2026-10-16 23:45:00")]


def test_the_october_set_is_the_static_tasks(tmp_path):
    assert task_set("october", tmp_path) == (TASKS, {})
    assert all(task.season_start is None for task in TASKS.values())
    with pytest.raises(ValueError, match="UNKNOWN TASK SET"):
        task_set("march", tmp_path)


# The external set: the three DWD stations as weather tasks with exactly the SensBee weather windows, so the two
# can be compared period for period. It needs no snapshot.
def test_the_external_set_is_the_dwd_trio_on_the_sensbee_weather_windows(tmp_path):
    tasks, notes = task_set("external", tmp_path)
    assert list(tasks) == ["DWD_MARTINRODA", "DWD_SCHMUECKE", "DWD_NEUHAUS"] and notes == {}
    assert tasks is EXTERNAL_TASKS
    sensbee = TASKS["MANEBACH_WEATHER_STATION"]
    for task in tasks.values():
        assert task == sensbee
        for period in ("test", "validation"):
            assert period_bounds(task, period) == period_bounds(sensbee, period)
    assert period_bounds(sensbee, "validation") == (pd.Timestamp("2026-07-09"), pd.Timestamp("2026-08-06"))
    assert period_bounds(sensbee, "test") == (pd.Timestamp("2026-08-06"), pd.Timestamp("2026-10-01"))


# The neighbours of the "stations" covariate sets are all weather stations of the whole set.
def test_each_task_set_has_its_own_weather_stations():
    sensbee = ("MANEBACH_WEATHER_STATION", "STUTZERBACH_WEATHER_STATION", "FRAUENWALD_WEATHER_STATION")
    assert set_stations("october") == set_stations("january") == sensbee
    assert set_stations("external") == ("DWD_MARTINRODA", "DWD_SCHMUECKE", "DWD_NEUHAUS")
    with pytest.raises(ValueError, match="UNKNOWN TASK SET"):
        set_stations("march")
