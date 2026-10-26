# Tests for benchmark/covariates.py and the runner's covariate path on synthetic data (no network):
# - Berlin time across the DST switch
# - the holiday table (local dates, Easter-based holidays computed here)
# - solar elevation against the solstice geometry
# - the opening profile is fitted only before the cutoff
# - the leakage guard of covariate_window: raising every raw value at or after the origin changes nothing that a
#   non-oracle forecast sees
# - the runner hands each forecast its own window, through the split by length and the one-by-one retry
# - a trainer gets only known columns before the cutoff
# - the "0 when closed" mask (Phase 4 step 4.3): zero exactly where the learned profile is below the threshold, and
#   the base forecast untouched elsewhere
# - its Phase 5 form (D5-2), whose profile comes from each context: hand-checked shares of 4 Mondays, and a run
#   without any covariate set
# - Phase 6a: the holiday table to 2027-04-01; training rows' opening profile from earlier weeks only (D-E8) and the
#   training items starting where it is defined; "stations96" (neighbours 24 h earlier, known) and "wx" (Manebach
#   weather for the halls, past) leak nothing at or after the origin

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark import covariates as cov
from benchmark.models.mask_models import mask_forecaster, recent_mask_forecaster
from benchmark.runner import _predict, naive_forecaster, run
from benchmark.snapshot import MANIFEST_NAME, write_raw_csv
from benchmark.tasks import HORIZON, QUANTILE_LEVELS, TASKS, SensorTask, target_series

TIMES = pd.date_range("2026-01-01", "2026-02-10", freq="15min", inclusive="left")  # 40 days of 15-min rows
STATION = "MANEBACH_WEATHER_STATION"


# Easter Sunday of a year (anonymous Gregorian algorithm, Meeus/Jones/Butcher).
def _easter(year: int) -> date:
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = (h + l_ - 7 * m + 114) % 31 + 1
    return date(year, month, day)


# Weather raw rows at TIMES: temperature plus the four covariate variables, each a distinct ramp.
def _weather_raw(offset: float = 0.0) -> pd.DataFrame:
    n = len(TIMES)
    return pd.DataFrame({"created_at": TIMES, "temperature": np.arange(n) * 0.01 + offset,
                         **{v: np.arange(n) * (j + 2) + offset for j, v in enumerate(cov.WEATHER_VARIABLES)}})


# Visitor raw rows at TIMES: 5 visitors on Mondays 09:00-10:00 Berlin time (08:00-09:00 UTC in winter), else 0.
def _visitor_raw() -> pd.DataFrame:
    local = cov.berlin(TIMES)
    v = np.where((local.weekday == 0) & (local.hour == 9), 5.0, 0.0)
    return pd.DataFrame({"created_at": TIMES, "visitors_total": v})


def test_berlin_time_across_the_dst_switch():
    index = pd.DatetimeIndex(["2026-03-29 00:45", "2026-03-29 01:00", "2026-10-25 00:45", "2026-10-25 01:00"])
    local = cov.berlin(index)
    assert [(t.hour, t.minute) for t in local] == [(1, 45), (3, 0), (2, 45), (2, 0)]
    features = cov.calendar(index[:2])
    assert np.allclose(features["tod_sin"], np.sin(2 * np.pi * np.array([105, 180]) / 1440))


def test_holiday_flags_follow_the_berlin_date():
    index = pd.DatetimeIndex(["2026-04-02 21:30", "2026-04-02 22:30",  # 23:30 / 00:30 CEST: Good Friday starts
                              "2025-12-31 23:30",                      # 00:30 CET on New Year's Day
                              "2026-01-06 12:00",                      # Epiphany: not a Thuringian holiday
                              "2026-02-15 12:00", "2026-02-16 12:00", "2026-02-21 12:00", "2026-02-22 12:00",
                              "2026-05-15 12:00", "2026-04-07 12:00"])
    features = cov.calendar(index)
    assert features["public_holiday"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert features["school_holiday"].tolist() == [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]  # 31 Dec: Christmas break


def test_the_easter_holidays_of_the_table_match_the_computed_easter():
    easter = _easter(2026)
    assert easter == date(2026, 4, 5) and _easter(2027) == date(2027, 3, 28)
    moving = {easter + timedelta(days=d) for d in (-2, 1, 39, 50)}  # Good Friday, Easter Monday, Ascension, Whit Monday
    moving |= {_easter(2027) + timedelta(days=d) for d in (-2, 1)}  # the table ends 2027-04-01, before Ascension 2027
    table = {t.date() for t in cov.PUBLIC_HOLIDAYS}
    assert moving <= table
    fixed = {(9, 20), (10, 3), (10, 31), (12, 25), (12, 26), (1, 1), (5, 1)}  # ThürFGtG § 2(1)
    expected = {date(y, m, d) for y in (2025, 2026, 2027) for m, d in fixed
                if cov.HOLIDAYS_FROM <= pd.Timestamp(date(y, m, d)) < cov.HOLIDAYS_TO}
    assert table - moving == expected


def test_calendar_refuses_dates_outside_the_holiday_table():
    with pytest.raises(ValueError, match="OUTSIDE THE HOLIDAY TABLE"):
        cov.calendar(pd.DatetimeIndex(["2025-06-30 12:00"]))
    with pytest.raises(ValueError, match="OUTSIDE THE HOLIDAY TABLE"):
        cov.calendar(pd.DatetimeIndex(["2027-03-31 22:30"]))  # 00:30 CEST on 1 Apr


# At solar noon the elevation is 90° − latitude + declination. The declination is ±23.44° at the solstices.
# Ilmenau lies at 10.9° E, so solar noon is about 44 min before 12:00 UTC.
# The equation of time moves that by under 17 min.
@pytest.mark.parametrize("day, declination", [("2026-06-21", 23.44), ("2025-12-21", -23.44)])
def test_solar_elevation_at_the_solstices(day, declination):
    index = pd.date_range(day, periods=24 * 60, freq="1min")
    elevation = cov.solar_elevation(index)
    assert abs(elevation.max() - (90 - cov.ILMENAU[0] + declination)) < 0.3
    noon = index[int(np.argmax(elevation))]
    assert pd.Timestamp(f"{day} 10:59") <= noon <= pd.Timestamp(f"{day} 11:33")
    assert elevation[0] < 0  # midnight UTC


def test_the_opening_profile_uses_only_data_before_the_cutoff():
    task = SensorTask("visitors_total", "MAX", "visitors", "2026-02-01", "2026-02-05",
                      (("2026-01-12 00:00", "2026-01-13 00:00", "a Monday excluded"),), train_start="2026-01-03")
    raw = _visitor_raw()
    outage = (raw["created_at"] >= "2026-01-12") & (raw["created_at"] < "2026-01-13")
    raw.loc[outage, "visitors_total"] = 0.0  # silent on the excluded Monday: 2/3 instead of 1 if it counted
    series = target_series(raw, task)
    cutoff = pd.Timestamp("2026-01-25")  # Mondays before it from train_start: 5, 12 (excluded) and 19 January
    profile = cov.open_profile(series, task, cutoff)
    assert profile.shape == (7, 96)
    open_slots = np.zeros((7, 96))
    open_slots[0, 36:40] = 1  # Monday 09:00-09:45 local
    assert np.array_equal(profile, open_slots)
    later = raw.copy()
    later.loc[later["created_at"] >= cutoff, "visitors_total"] = 99.0
    assert np.array_equal(cov.open_profile(target_series(later, task), task, cutoff), profile)


# Builds a set for STATION from the given raw frames and returns its window for one origin and 1-day context.
def _window(name: str, frames: dict[str, pd.DataFrame], origin: pd.Timestamp) -> pd.DataFrame:
    task = TASKS[STATION]
    series = target_series(frames[STATION], task)
    frame, kinds, _ = cov.build(name, STATION, frames, series, task, pd.Timestamp("2026-02-01"))
    context_index = pd.date_range(origin - pd.Timedelta("1D"), origin, freq="15min", inclusive="left")
    return cov.covariate_window(frame, kinds, origin, context_index)


def test_nothing_at_or_after_the_origin_reaches_a_non_oracle_window():
    origin = pd.Timestamp("2026-02-02 06:00")
    frames = {s: _weather_raw(i) for i, s in enumerate(cov.STATIONS)}
    for station in cov.STATIONS:  # a 2-h gap right before the origin, so the context is forward-filled
        frames[station] = frames[station][(frames[station]["created_at"] < origin - pd.Timedelta("2h"))
                                          | (frames[station]["created_at"] >= origin)]
    changed = {}
    for station, raw in frames.items():
        later = raw.copy()
        numeric = later.columns.drop("created_at")
        later.loc[later["created_at"] >= origin, numeric] += 1000.0
        changed[station] = later
    for name in ("sun", "met", "stations"):
        window = _window(name, frames, origin)
        pd.testing.assert_frame_equal(window, _window(name, changed, origin))
        past_only = [c for c in window.columns if c not in cov.SUN]
        assert window.loc[origin:, past_only].isna().all().all()  # the future of past-only columns is hidden
        assert window.loc[:origin - pd.Timedelta("15min"), past_only].notna().all().all()  # gap forward-filled
        assert len(window) == 96 + HORIZON
    oracle, oracle_changed = _window("met_oracle", frames, origin), _window("met_oracle", changed, origin)
    pd.testing.assert_frame_equal(oracle.iloc[:96], oracle_changed.iloc[:96])  # the past is the same
    assert (oracle_changed.loc[origin:, "humidity"] - oracle.loc[origin:, "humidity"] == 1000).all()


def test_only_the_oracle_set_has_oracle_columns():
    frames = {s: _weather_raw(i) for i, s in enumerate(cov.STATIONS)}
    task = TASKS[STATION]
    series = target_series(frames[STATION], task)
    for name, spec in cov.COVARIATE_SETS.items():
        if spec.kind != "regular":
            continue
        _, kinds, _ = cov.build(name, STATION, frames, series, task, pd.Timestamp("2026-02-01"))
        assert ("oracle" in kinds.values()) == spec.oracle == (name == "met_oracle")
        assert sorted(set(kinds.values())) == list(spec.column_kinds)
    _, kinds, _ = cov.build("stations", STATION, frames, series, task, pd.Timestamp("2026-02-01"))
    assert sorted(kinds) == sorted([*cov.SUN, "temperature_FRAUENWALD_WEATHER_STATION",
                                    "temperature_STUTZERBACH_WEATHER_STATION"])


# A window lists its known and oracle columns as future columns, never the past-only ones, so a model gets one input
# schema per set even when an oracle column's future is all NaN.
def test_a_window_lists_its_future_columns_by_kind():
    frames = {s: _weather_raw(i) for i, s in enumerate(cov.STATIONS)}
    origin = pd.Timestamp("2026-02-02 06:00")
    assert _window("met", frames, origin).attrs["future_columns"] == list(cov.SUN)
    later = frames[STATION]["created_at"] >= origin
    frames[STATION].loc[later, list(cov.WEATHER_VARIABLES)] = np.nan  # no observed future of the oracle variables
    oracle = _window("met_oracle", frames, origin)
    assert oracle.attrs["future_columns"] == [*cov.SUN, *cov.WEATHER_VARIABLES]
    assert oracle.loc[origin:, list(cov.WEATHER_VARIABLES)].isna().all().all()


# Writes a snapshot with one visitor sensor "V" (the Monday pattern) and returns its folder and task (4-day test).
def _visitor_snapshot(tmp_path: Path) -> tuple[Path, dict[str, SensorTask]]:
    snapshot_dir = tmp_path / "snapshot"
    sha256 = write_raw_csv(_visitor_raw(), snapshot_dir / "raw" / "V.csv.gz")
    (snapshot_dir / MANIFEST_NAME).write_text(json.dumps(
        {"snapshot_id": "test-snapshot", "sensors": {"V": {"file": "raw/V.csv.gz", "sha256": sha256}}}))
    return snapshot_dir, {"V": SensorTask("visitors_total", "MAX", "visitors", "2026-02-01", "2026-02-05",
                                          train_start="2026-01-03")}


# Naive forecasts that record, per context, its last timestamp and its covariate window's index and columns.
def _recording_forecaster(calls: list):
    def forecast(contexts, horizon, levels, seed, covariates):
        for context, window in zip(contexts, covariates):
            calls.append((context.index, window.index, list(window.columns)))
        return naive_forecaster(contexts, horizon, levels, seed)
    return forecast


def test_each_forecast_gets_its_own_window_and_the_run_records_the_set(tmp_path):
    snapshot_dir, tasks = _visitor_snapshot(tmp_path)
    calls = []
    run_dir = run(_recording_forecaster(calls), "rec+cal_open", snapshot_dir, tmp_path / "runs", tasks=tasks,
                  context_days=(1, 7), covariates="cal_open")
    assert len(calls) == 13 * 2
    for context_index, window_index, columns in calls:
        origin = context_index[-1] + pd.Timedelta("15min")
        assert window_index.equals(context_index.append(pd.date_range(origin, periods=HORIZON, freq="15min")))
        assert columns == [*cov.CALENDAR, "open_share"]
    record = json.loads((run_dir / "run.json").read_text())
    assert record["covariates"] == {"set": "cal_open", "kind": "visitors", "groups": ["calendar", "open"],
                                    "oracle": False}
    assert record["counts"]["V"]["covariates"]["columns"]["open_share"] == "known"
    assert np.array(record["counts"]["V"]["covariates"]["open_profile"]).shape == (7, 96)


def test_a_set_of_another_kind_is_refused_before_running(tmp_path):
    snapshot_dir, tasks = _visitor_snapshot(tmp_path)
    with pytest.raises(ValueError, match="IS FOR regular SENSORS"):
        run(naive_forecaster, "naive+sun", snapshot_dir, tmp_path / "runs", tasks=tasks, covariates="sun")


# The split by length and the one-by-one retry keep each covariate window with its context.
def test_windows_stay_paired_through_the_length_split_and_the_retry():
    contexts = [pd.Series(np.arange(n, dtype=float), index=pd.date_range(start, periods=n, freq="15min"))
                for n, start in ((4, "2026-01-01"), (6, "2026-01-02"), (4, "2026-01-03"))]
    windows = [pd.DataFrame({"x": [float(i)]}) for i in range(3)]
    seen = []

    def forecast(batch, horizon, levels, seed, covariates):
        if len(batch) > 1:
            raise RuntimeError("batch refused")  # forces the retry
        seen.append((batch[0].index[0], covariates[0]["x"].iloc[0]))
        return naive_forecaster(batch, horizon, levels, seed)

    _, status = _predict(forecast, contexts, QUANTILE_LEVELS, 0, windows)
    assert status == ["ok"] * 3
    assert sorted(seen) == [(pd.Timestamp("2026-01-01"), 0.0), (pd.Timestamp("2026-01-02"), 1.0),
                            (pd.Timestamp("2026-01-03"), 2.0)]


# "met" has past-only columns next to the known ones: the trainer must get the known ones only, before the cutoff.
def test_a_trainer_gets_only_known_columns_before_the_cutoff(tmp_path):
    snapshot_dir = tmp_path / "snapshot"
    sha256 = write_raw_csv(_weather_raw(), snapshot_dir / "raw" / "W.csv.gz")
    (snapshot_dir / MANIFEST_NAME).write_text(json.dumps(
        {"snapshot_id": "test-snapshot", "sensors": {"W": {"file": "raw/W.csv.gz", "sha256": sha256}}}))
    tasks = {"W": SensorTask("temperature", "AVG", "regular", "2026-02-01", "2026-02-05")}
    received = []

    def trainer(sensor, items, seed, covariates):
        received.append(covariates)
        return _recording_forecaster([]), {}

    run(None, "t+met", snapshot_dir, tmp_path / "runs", tasks=tasks, context_days=(1,), trainer=trainer,
        covariates="met")
    (frame,) = received
    assert list(frame.columns) == list(cov.SUN)
    assert frame.index[-1] == pd.Timestamp("2026-02-01") - pd.Timedelta("15min")


def test_the_mask_zeroes_exactly_the_closed_slots_and_keeps_the_rest():
    context = pd.Series(np.arange(4, dtype=float) + 1, index=pd.date_range("2026-01-05", periods=4, freq="15min"))
    share = np.where(np.arange(HORIZON) % 2 == 0, 0.05, 0.5)
    window = pd.DataFrame({"open_share": np.r_[np.ones(4), share]},
                          index=context.index.append(pd.date_range("2026-01-05 01:00", periods=HORIZON, freq="15min")))
    forecast, params = mask_forecaster(lambda: (naive_forecaster, {"point_only": True}), 0.1, "naive")
    quantiles, status = forecast([context], HORIZON, QUANTILE_LEVELS, 0, covariates=[window])
    assert status == ["ok"] and params["threshold"] == 0.1 and params["point_only"] is True
    assert params["base_model"] == "naive"
    assert (quantiles[0, share < 0.1] == 0).all() and (quantiles[0, share >= 0.1] == 4.0).all()


def test_a_masked_run_forecasts_zero_outside_the_learned_opening_hours(tmp_path):
    snapshot_dir, tasks = _visitor_snapshot(tmp_path)
    forecast, params = mask_forecaster(lambda: (lambda c, h, l, s: (np.full((len(c), h, len(l)), 7.0), ["ok"] * len(c)),
                                                {"point_only": False}), 0.5, "base")
    run_dir = run(forecast, "mask+cal_open", snapshot_dir, tmp_path / "runs", tasks=tasks, context_days=(1,),
                  covariates="cal_open", model_params=params)
    forecasts = pd.read_csv(run_dir / "forecasts.csv.gz", parse_dates=["timestamp"])
    local = cov.berlin(pd.DatetimeIndex(forecasts["timestamp"]))
    open_slot = (local.weekday == 0) & (local.hour == 9)  # the only slots with visitors before the cutoff
    assert open_slot.any() and (forecasts.loc[open_slot, "q0.5"] == 7.0).all()
    assert (forecasts.loc[~open_slot, [f"q{q:g}" for q in QUANTILE_LEVELS]] == 0).all().all()


# A base forecaster of 7 everywhere, scored as probabilistic.
def _sevens():
    return lambda c, h, l, s: (np.full((len(c), h, len(l)), 7.0), ["ok"] * len(c)), {"point_only": False}


# 28 days from Monday 5 Jan (4 Mondays). Local 09:00 (08:00 UTC) has visitors on all 4 Mondays, 09:15 on 1 of them
# (share 0.25), 09:30 on 2 (0.5), nothing else. The horizon starts on Monday 2 Feb at 00:00 UTC, so step 32 is local
# 09:00. θ 0.3 masks the 0.25 slot and θ 0.2 keeps it. Every never-open slot is 0 either way.
@pytest.mark.parametrize("threshold, kept", [(0.3, [32, 34]), (0.2, [32, 33, 34])])
def test_the_recent_mask_takes_its_profile_from_the_context(threshold, kept):
    index = pd.date_range("2026-01-05", periods=28 * 96, freq="15min")
    values = pd.Series(0.0, index=index)
    mondays = [pd.Timestamp(f"2026-01-{d:02d}") for d in (5, 12, 19, 26)]
    for k, monday in enumerate(mondays):
        values[monday + pd.Timedelta("8h")] = 5.0
        if k == 0:
            values[monday + pd.Timedelta("8h15min")] = 5.0
        if k < 2:
            values[monday + pd.Timedelta("8h30min")] = 5.0
    forecast, params = recent_mask_forecaster(_sevens, threshold, "base")
    quantiles, status = forecast([values], HORIZON, QUANTILE_LEVELS, 0)
    assert status == ["ok"] and params["threshold"] == threshold and params["base_model"] == "base"
    expected = np.zeros(HORIZON)
    expected[kept] = 7.0
    assert np.array_equal(quantiles[0, :, 4], expected) and (quantiles[0] == expected[:, None]).all()


def test_a_recent_mask_run_needs_no_covariate_set(tmp_path):
    snapshot_dir, tasks = _visitor_snapshot(tmp_path)
    forecast, params = recent_mask_forecaster(_sevens, 0.5, "base")
    run_dir = run(forecast, "mask_recent", snapshot_dir, tmp_path / "runs", tasks=tasks, context_days=(7,),
                  model_params=params)
    forecasts = pd.read_csv(run_dir / "forecasts.csv.gz", parse_dates=["timestamp"])
    local = cov.berlin(pd.DatetimeIndex(forecasts["timestamp"]))
    open_slot = (local.weekday == 0) & (local.hour == 9)  # every 7-day context holds exactly one Monday 09:00-09:45
    assert open_slot.any() and (forecasts.loc[open_slot, "q0.5"] == 7.0).all()
    assert (forecasts.loc[~open_slot, [f"q{q:g}" for q in QUANTILE_LEVELS]] == 0).all().all()
    assert json.loads((run_dir / "run.json").read_text())["covariates"] is None


def test_the_extended_holiday_table_flags_both_school_years():
    index = pd.DatetimeIndex(["2025-07-15 12:00", "2025-08-09 12:00", "2026-12-31 12:00", "2027-01-04 12:00",
                              "2027-02-01 12:00", "2027-03-26 12:00"])
    features = cov.calendar(index)
    assert features["school_holiday"].tolist() == [1, 0, 1, 0, 1, 1]
    assert features["public_holiday"].tolist() == [0, 0, 0, 0, 0, 1]  # Good Friday 2027


# D-E8 by hand on the Monday pattern (5 visitors 09:00-10:00 Berlin), with the Monday of 19 January closed.
# A Monday 09:00 row gets the share of open Mondays in the weeks before its own: 5 Jan has no earlier Monday (NaN),
# 12 Jan → 1 (5 Jan), 19 Jan → 1 (its own closed day doesn't count), 26 Jan → 2/3. The cutoff profile would give
# every one of them 3/4.
def test_training_rows_get_the_opening_profile_of_earlier_weeks_only():
    task = SensorTask("visitors_total", "MAX", "visitors", "2026-02-01", "2026-02-05", train_start="2026-01-03")
    raw = _visitor_raw()
    raw.loc[(raw["created_at"] >= "2026-01-19") & (raw["created_at"] < "2026-01-20"), "visitors_total"] = 0.0
    series = target_series(raw, task)
    mondays = pd.DatetimeIndex(["2026-01-05 08:00", "2026-01-12 08:00", "2026-01-19 08:00", "2026-01-26 08:00"])
    share = cov.training_open_share(series, task, pd.Timestamp("2026-01-31"), mondays)
    assert np.isnan(share[0]) and share[1:].tolist() == pytest.approx([1.0, 1.0, 2 / 3])
    assert cov.open_profile(series, task, pd.Timestamp("2026-01-31"))[0, 36] == pytest.approx(3 / 4)


# The runner hands a cal_open trainer the earlier-weeks profile and starts its items where every known column is
# defined: 10 Jan 00:00 UTC. Monday to Friday of the week of 5 Jan have no earlier day of their weekday, and neither
# do Saturday 10 Jan's first local slots (9 Jan 23:00-23:45 UTC), because train_start, 3 Jan 00:00 UTC, is 01:00 in
# Berlin. 7 days of items are dropped.
def test_a_cal_open_trainer_gets_earlier_weeks_and_trimmed_items(tmp_path):
    snapshot_dir, tasks = _visitor_snapshot(tmp_path)
    received = []

    def trainer(sensor, items, seed, covariates):
        received.append((items, covariates))
        return _recording_forecaster([]), {}

    run_dir = run(None, "t+cal_open", snapshot_dir, tmp_path / "runs", tasks=tasks, context_days=(1,),
                  trainer=trainer, covariates="cal_open")
    ((items, frame),) = received
    start = pd.Timestamp("2026-01-10 00:00")
    assert frame.index[0] == start and items[0].index[0] == start and frame.notna().all().all()
    series = target_series(_visitor_raw(), tasks["V"])
    expected = cov.training_open_share(series, tasks["V"], pd.Timestamp("2026-02-01"), frame.index)
    assert np.array_equal(frame["open_share"].to_numpy(), expected)
    fit = json.loads((Path(run_dir) / "run.json").read_text())["counts"]["V"]["fit"]
    assert fit["covariates_from"].startswith("2026-01-10T00:00") and fit["steps_before_covariates"] == 7 * 96


# stations96: at a future step t the neighbour column holds that station's last temperature at or before t - 96
# steps, and raising every raw value at or after the origin changes nothing in the window. The neighbour has a gap
# from 30 h before to 2 h after the origin, so filling backwards, or shifting by 96 observations instead of 24 h,
# would fail.
def test_stations96_holds_the_neighbours_a_day_earlier_and_leaks_nothing():
    origin = pd.Timestamp("2026-02-02 06:00")
    neighbour = "FRAUENWALD_WEATHER_STATION"
    frames = {s: _weather_raw(i) for i, s in enumerate(cov.STATIONS)}
    gap = (frames[neighbour]["created_at"] >= origin - pd.Timedelta("30h")) & (
        frames[neighbour]["created_at"] < origin + pd.Timedelta("2h"))
    frames[neighbour] = frames[neighbour][~gap]
    changed = {}
    for station, raw in frames.items():
        later = raw.copy()
        later.loc[later["created_at"] >= origin, "temperature"] += 1000.0
        changed[station] = later
    window = _window("stations96", frames, origin)
    pd.testing.assert_frame_equal(window, _window("stations96", changed, origin))
    column = f"temperature_{neighbour}_lag96"
    raw = frames[neighbour].set_index("created_at")["temperature"].reindex(TIMES).ffill()
    future = window.loc[origin:, column]
    assert np.allclose(future.to_numpy(), raw.reindex(future.index - pd.Timedelta("1D")).to_numpy())
    assert column in window.attrs["future_columns"] and window[column].notna().all()


# wx: a hall gets Manebach's temperature and precipitation rate, Stützerbach's where Manebach has none, as past-only
# columns blanked after the origin.
def test_wx_gives_a_hall_the_station_weather_as_past_only():
    origin = pd.Timestamp("2026-02-02 06:00")
    manebach = _weather_raw().assign(precipitation_rate=0.5)
    manebach = manebach[(manebach["created_at"] < "2026-01-20") | (manebach["created_at"] >= "2026-01-21")]
    stutzerbach = _weather_raw(100.0).assign(precipitation_rate=7.0)
    frames = {"V": _visitor_raw(), "MANEBACH_WEATHER_STATION": manebach, "STUTZERBACH_WEATHER_STATION": stutzerbach,
              "FRAUENWALD_WEATHER_STATION": _weather_raw(200.0).assign(precipitation_rate=9.0)}
    task = SensorTask("visitors_total", "MAX", "visitors", "2026-02-01", "2026-02-05", train_start="2026-01-03")
    frame, kinds, info = cov.build("wx", "V", frames, target_series(_visitor_raw(), task), task,
                                   pd.Timestamp("2026-02-01"))
    assert kinds == {"hall_temperature": "past", "hall_precipitation_rate": "past"}
    assert (frame.loc["2026-01-20", "hall_precipitation_rate"] == 7.0).all()  # Manebach's gap, from Stützerbach
    assert (frame.loc["2026-01-21", "hall_precipitation_rate"] == 0.5).all()
    assert info["hall_weather_sources"]["hall_temperature"]["STUTZERBACH_WEATHER_STATION"] == pytest.approx(1 / 40)
    _, _, day = cov.build("wx", "V", frames, target_series(_visitor_raw(), task), task, pd.Timestamp("2026-02-01"),
                          (pd.Timestamp("2026-01-20"), pd.Timestamp("2026-01-21")))  # shares over the span only
    assert day["hall_weather_sources"]["hall_temperature"]["STUTZERBACH_WEATHER_STATION"] == 1.0
    context_index = pd.date_range(origin - pd.Timedelta("1D"), origin, freq="15min", inclusive="left")
    window = cov.covariate_window(frame, kinds, origin, context_index)
    assert window.loc[origin:].isna().all().all() and window.loc[:origin - pd.Timedelta("15min")].notna().all().all()
