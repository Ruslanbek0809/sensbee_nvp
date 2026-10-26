# Covariates for the benchmark harness (Phase 4): named covariate sets, the covariate frame of a sensor on its 15-min
# grid, and the window one forecast may see. Every column has a kind:
# - "known": known before the origin for the whole horizon (calendar, holidays, solar elevation, and the opening
#   profile learned from the data before the cutoff).
# - "past": observed values (other weather variables, other stations), visible only before the origin.
# - "oracle": observed future values, an upper bound only (reported apart from the other models).
# covariate_window() is the one place that decides what a forecast sees.
# - The frame is cut at the origin before it is forward-filled (the context rule of benchmark/tasks.py).
# - The future rows of "past" columns are blanked.
# - Only an oracle set has oracle columns. Their future rows keep the observed values (gaps stay NaN).
#
# Calendar features use Europe/Berlin time (SensBee timestamps are naive UTC).
# Holidays come from a dated table that covers HOLIDAYS_FROM to HOLIDAYS_TO.
# Calendar frames are built only inside that range. A window that needs a row outside it raises instead of getting a
# silent 0.
# - Public holidays: Thüringer Feier- und Gedenktagsgesetz (ThürFGtG) § 2(1), statewide: New Year, Good Friday, Easter
#   Monday, 1 May, Ascension, Whit Monday, 20 September (Weltkindertag), 3 October, Reformation Day (31 October),
#   Christmas Day and 26 December (umwelt-online.de, last amendment 26.03.2019, checked 2026-10-14).
#   Corpus Christi is left out. It is a holiday only where an ordinance names a mainly Catholic municipality
#   (§ 2(2)), and Ilmenau is not one.
#   Easter 2026 is 5 April (tests/benchmark/test_covariates.py computes it).
# - School holidays: KMK "Ferien im Schuljahr 2025/2026", Thüringen row, Stand 09.10.2025
#   (kmk.org/fileadmin/Dateien/pdf/Ferienkalender/FER2025_26.pdf, checked 2026-10-14).
#   The table gives the first and last holiday day. 15.05.2026 is a school-free day.
#   Extended 2026-10-26 (Phase 6a) to 2025-07-01 .. 2027-04-01: KMK "Ferien im Schuljahr 2024/2025", Thüringen row,
#   Stand 17.05.2023 (FER2024_25.pdf: summer 28.06.-08.08.2025) and "Ferien im Schuljahr 2026/2027", Thüringen row,
#   Stand 09.10.2025 (FER2026_27.pdf: autumn 12.10.-24.10.2026, Christmas 23.12.2026-02.01.2027,
#   winter 01.02.-06.02.2027, Easter 22.03.-03.04.2027), both checked 2026-10-26. Easter 2027 is 28 March.
# Solar elevation: NOAA Global Monitoring Division, "General Solar Position Calculations" (gml.noaa.gov/grad/solcalc/
# solareqns.PDF, checked 2026-10-14). It is computed at one point for Ilmenau (de.wikipedia.org/wiki/Ilmenau:
# 50.687 N, 10.914 E, checked 2026-10-14). The weather stations lie within about 10 km, which moves the sun angle by
# a fraction of a degree.
# The opening profile (D4-1): per Berlin weekday and 15-min local slot, the share of days with visitors_total > 0.
# It is taken from the hall's training data before the cutoff (from task.train_start, without the task's exclusions).
# It is fitted once per sensor and period, like a trained model. Every forecast window uses that cutoff profile
# (all of it lies before the origin).
# A trained model's training rows get a profile from earlier weeks only (D-E8, 2026-10-22; training_open_share):
# otherwise a training row's own day would be part of its feature. Training starts where every known covariate is
# defined (training_covariates), so the first days, with no earlier week of that weekday, are left out.
# Phase 6a sets (fixed 2026-10-26, before any run):
# - "stations96" (weather): the other stations' temperature 96 steps (24 h) earlier, forward-filled. At a future step
#   t it holds a value from before t - 96 steps, which is before the origin, so it is known for the whole horizon.
#   It gives a model without a past-covariate input (AutoGluon's tabular models drop past covariates) the neighbours.
# - "wx" (halls): temperature and precipitation_rate (past) as an input for the visitor counts: Manebach's, and in
#   each 15-min bucket Manebach lacks, Stützerbach's, else Frauenwald's (owner decision 2026-10-26: no single station
#   covers both halls' windows; Manebach misses much of Dec 2025-Jan 2026). The share from each station is recorded.
# series_open_profile() computes the same share from any series. Phase 5 uses it on each 28-day context, because the
# service can compute that profile without knowing when a season starts (D5-2).

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from benchmark.snapshot import BUCKET_FREQ, bucket_15min
from benchmark.tasks import STEP, TASKS, SensorTask, _excluded, target_window, training_history

logger = logging.getLogger(__name__)

KINDS = ("known", "past", "oracle")
BERLIN = "Europe/Berlin"
ILMENAU = (50.687222, 10.914167)  # latitude, longitude (degrees north, east)
HOLIDAYS_FROM, HOLIDAYS_TO = pd.Timestamp("2025-07-01"), pd.Timestamp("2027-04-01")  # [from, to), Berlin dates
PUBLIC_HOLIDAYS = tuple(pd.Timestamp(d) for d in (
    "2025-09-20", "2025-10-03", "2025-10-31", "2025-12-25", "2025-12-26", "2026-01-01", "2026-04-03", "2026-04-06",
    "2026-05-01", "2026-05-14", "2026-05-25", "2026-09-20", "2026-10-03", "2026-10-31", "2026-12-25", "2026-12-26",
    "2027-01-01", "2027-03-26", "2027-03-29"))
SCHOOL_HOLIDAYS = tuple((pd.Timestamp(a), pd.Timestamp(b)) for a, b in (  # first and last day, inclusive
    ("2025-06-28", "2025-08-08"),
    ("2025-10-06", "2025-10-18"), ("2025-12-22", "2026-01-03"), ("2026-02-16", "2026-02-21"),
    ("2026-04-07", "2026-04-17"), ("2026-05-15", "2026-05-15"), ("2026-07-04", "2026-08-14"),
    ("2026-10-12", "2026-10-24"), ("2026-12-23", "2027-01-02"), ("2027-02-01", "2027-02-06"),
    ("2027-03-22", "2027-04-03")))
WEATHER_VARIABLES = ("humidity", "global_radiation", "pressure_rel", "wind_speed_avg")  # fixed before any run
STATIONS = tuple(name for name, task in TASKS.items() if task.kind == "regular")
HALL_WEATHER_STATIONS = ("MANEBACH_WEATHER_STATION",       # inside Ilmenau; its gaps are filled from the others
                         "STUTZERBACH_WEATHER_STATION", "FRAUENWALD_WEATHER_STATION")
HALL_WEATHER_VARIABLES = ("temperature", "precipitation_rate")  # fixed 2026-10-26, before any run
LAG_STEPS = 96                                            # stations96: the neighbours 24 h earlier
CALENDAR = ("tod_sin", "tod_cos", "dow_sin", "dow_cos", "public_holiday", "school_holiday")
SUN = ("tod_sin", "tod_cos", "solar_elevation")
# The kind of every column a feature group produces.
GROUP_KINDS = {"calendar": "known", "open": "known", "sun": "known", "met": "past", "stations": "past",
               "met_oracle": "oracle", "stations96": "known", "wx": "past"}


# A named covariate set: the task kind it applies to ("visitors" or "regular") and its feature groups.
@dataclass(frozen=True)
class CovariateSet:
    kind: str
    groups: tuple[str, ...]

    # Kinds of the columns this set produces.
    @property
    def column_kinds(self) -> tuple[str, ...]:
        return tuple(sorted({GROUP_KINDS[group] for group in self.groups}))

    @property
    def oracle(self) -> bool:
        return "oracle" in self.column_kinds


# Fixed before any run (Phase 4 plan, section A).
COVARIATE_SETS: dict[str, CovariateSet] = {
    "cal": CovariateSet("visitors", ("calendar",)),
    "cal_open": CovariateSet("visitors", ("calendar", "open")),
    "sun": CovariateSet("regular", ("sun",)),
    "met": CovariateSet("regular", ("sun", "met")),
    "stations": CovariateSet("regular", ("sun", "stations")),
    "met_oracle": CovariateSet("regular", ("sun", "met_oracle")),
    "stations96": CovariateSet("regular", ("sun", "stations96")),   # Phase 6a sets (header)
    "wx": CovariateSet("visitors", ("wx",)),
    "cal_open_wx": CovariateSet("visitors", ("calendar", "open", "wx")),
    "cal_parking": CovariateSet("counter", ("calendar",)),  # parking's calendar inputs (tuned trained models)
}


# Naive-UTC timestamps as Europe/Berlin local time.
def berlin(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_localize("UTC").tz_convert(BERLIN)


# Time of day of Berlin-local timestamps as (sin, cos).
def _time_of_day(local: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    tod = 2 * np.pi * (local.hour * 60 + local.minute).to_numpy() / 1440
    return np.sin(tod), np.cos(tod)


# The "sun" group's columns (SUN) for naive-UTC timestamps: time of day in Berlin and solar elevation.
def sun_columns(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    tod_sin, tod_cos = _time_of_day(berlin(index))
    return {"tod_sin": tod_sin, "tod_cos": tod_cos, "solar_elevation": solar_elevation(index)}


# Calendar features in Berlin time: time of day and weekday as sin/cos, plus the public and school holiday flags
# of the local date. Raises when a timestamp's local date lies outside the holiday table.
def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = berlin(index)
    days = local.tz_localize(None).normalize()
    if len(days) and (days.min() < HOLIDAYS_FROM or days.max() >= HOLIDAYS_TO):
        raise ValueError(f"DATES {days.min().date()}..{days.max().date()} OUTSIDE THE HOLIDAY TABLE "
                         f"{HOLIDAYS_FROM.date()}..{HOLIDAYS_TO.date()}")
    tod_sin, tod_cos = _time_of_day(local)
    dow = 2 * np.pi * local.weekday.to_numpy() / 7
    school = np.zeros(len(index), dtype=bool)
    for first, last in SCHOOL_HOLIDAYS:
        school |= (days >= first) & (days <= last)
    return pd.DataFrame({"tod_sin": tod_sin, "tod_cos": tod_cos, "dow_sin": np.sin(dow),
                         "dow_cos": np.cos(dow), "public_holiday": days.isin(PUBLIC_HOLIDAYS).astype(float),
                         "school_holiday": school.astype(float)}, index=index)


# Solar elevation in degrees at Ilmenau at naive-UTC timestamps (NOAA general solar position equations, see the
# header).
def solar_elevation(index: pd.DatetimeIndex) -> np.ndarray:
    latitude, longitude = ILMENAU
    year_days = np.where(index.is_leap_year, 366, 365)
    gamma = 2 * np.pi / year_days * (index.dayofyear - 1 + (index.hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma)
            + 0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    tst = index.hour * 60 + index.minute + index.second / 60 + eqtime + 4 * longitude  # timezone 0: UTC
    ha = np.radians(tst / 4 - 180)
    lat = np.radians(latitude)
    cos_zenith = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    return 90 - np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))


# Weekday (0 = Monday) and 15-min slot (0-95) of naive-UTC timestamps in Berlin time.
def _local_slots(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    local = berlin(index)
    return local.weekday.to_numpy(), (local.hour * 4 + local.minute // 15).to_numpy()


# Opening profile of a series, shape [7 weekdays, 96 slots]: the share of its days with a value > 0 in that Berlin
# weekday and 15-min slot. Slots the series doesn't cover are NaN.
def series_open_profile(series: pd.Series) -> np.ndarray:
    weekday, slot = _local_slots(series.index)
    share = (series.to_numpy() > 0).astype(float)
    profile = pd.Series(share).groupby([weekday, slot]).mean()
    return profile.reindex(pd.MultiIndex.from_product([range(7), range(96)])).to_numpy().reshape(7, 96)


# Opening profile of a hall, shape [7 weekdays, 96 slots]: the share of days with visitors > 0 in the training data
# before the cutoff (from task.train_start, without the task's exclusions).
def open_profile(series: pd.Series, task: SensorTask, cutoff: pd.Timestamp) -> np.ndarray:
    history = training_history(series, task, cutoff)
    return series_open_profile(history[~_excluded(history.index, task)])


# Opening-profile values of training rows from earlier weeks only (D-E8): a row in Berlin week w (Monday to Sunday)
# gets the share of days with visitors > 0 in its weekday and slot over the hall's training data (from
# task.train_start, without the exclusions, before the cutoff) in the weeks before w. NaN where that weekday and slot
# have no earlier day, and for rows outside the training weeks.
def training_open_share(series: pd.Series, task: SensorTask, cutoff: pd.Timestamp,
                        index: pd.DatetimeIndex) -> np.ndarray:
    history = training_history(series, task, cutoff)
    history = history[~_excluded(history.index, task)]

    def weeks(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        local = berlin(idx).tz_localize(None).normalize()
        return local - pd.to_timedelta(local.weekday, unit="D")

    starts = pd.DatetimeIndex(sorted(set(weeks(history.index))))
    weekday, slot = _local_slots(history.index)
    position = starts.get_indexer(weeks(history.index))
    opened = np.zeros((len(starts), 7, 96))
    days = np.zeros((len(starts), 7, 96))
    np.add.at(opened, (position, weekday, slot), (history.to_numpy() > 0).astype(float))
    np.add.at(days, (position, weekday, slot), 1.0)
    before_opened = np.cumsum(opened, axis=0) - opened  # weeks strictly before each week
    before_days = np.cumsum(days, axis=0) - days
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(before_days > 0, before_opened / before_days, np.nan)
    row_weekday, row_slot = _local_slots(index)
    row_position = starts.get_indexer(weeks(index))
    out = np.full(len(index), np.nan)
    inside = row_position >= 0
    out[inside] = share[row_position[inside], row_weekday[inside], row_slot[inside]]
    return out


# The profile value of each timestamp (its Berlin weekday and slot).
def open_share(index: pd.DatetimeIndex, profile: np.ndarray) -> np.ndarray:
    weekday, slot = _local_slots(index)
    return profile[weekday, slot]


# The covariate frame of one sensor for a set. It returns every column on the sensor's grid (calendar sets only
# inside the holiday table), the kind of each column, and info for run.json (the opening profile).
# cutoff: the period start, where the opening profile is fitted.
# Weather columns are SensBee AVG buckets, unfilled. The runner checks that the set matches the sensor's task kind.
# span: [start, end) over which the stations96 observed shares and the wx source shares are recorded (the runner
# passes the evaluated windows; None = the whole frame).
def build(name: str, sensor: str, frames: dict[str, pd.DataFrame], series: pd.Series, task: SensorTask,
          cutoff: pd.Timestamp,
          span: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None) -> tuple[pd.DataFrame, dict[str, str], dict]:
    spec = COVARIATE_SETS[name]
    index = series.index

    # Share of True in a boolean series over the span.
    def share(flags: pd.Series) -> float:
        if span is not None:
            flags = flags[(flags.index >= span[0]) & (flags.index < span[1])]
        return round(float(flags.mean()), 4)

    if "calendar" in spec.groups:
        local_days = berlin(index).tz_localize(None).normalize()
        index = index[(local_days >= HOLIDAYS_FROM) & (local_days < HOLIDAYS_TO)]
    columns: dict[str, np.ndarray] = {}
    kinds: dict[str, str] = {}
    info: dict = {}
    for group in spec.groups:
        if group == "calendar":
            features = calendar(index)
            for column in CALENDAR:
                columns[column], kinds[column] = features[column].to_numpy(), GROUP_KINDS[group]
        elif group == "open":
            profile = open_profile(series, task, cutoff)
            columns["open_share"], kinds["open_share"] = open_share(index, profile), GROUP_KINDS[group]
            info["open_profile"] = np.round(profile, 4).tolist()
        elif group == "sun":
            columns.update(sun_columns(index))
            kinds.update({column: GROUP_KINDS[group] for column in SUN})
        elif group in ("met", "met_oracle"):
            for variable in WEATHER_VARIABLES:
                columns[variable] = bucket_15min(frames[sensor], variable, "AVG").reindex(index).to_numpy()
                kinds[variable] = GROUP_KINDS[group]
        elif group == "stations":
            for station in STATIONS:
                if station != sensor:
                    column = f"temperature_{station}"
                    columns[column] = bucket_15min(frames[station], "temperature", "AVG").reindex(index).to_numpy()
                    kinds[column] = GROUP_KINDS[group]
        elif group == "stations96":
            grid = pd.date_range(index[0], index[-1], freq=BUCKET_FREQ)
            for station in STATIONS:
                if station != sensor:
                    column = f"temperature_{station}_lag{LAG_STEPS}"
                    lagged = bucket_15min(frames[station], "temperature", "AVG").reindex(grid).shift(LAG_STEPS)
                    info.setdefault("lagged_observed_share", {})[column] = share(lagged.notna())
                    columns[column] = lagged.ffill().reindex(index).to_numpy()  # earlier values only
                    kinds[column] = GROUP_KINDS[group]
        elif group == "wx":
            for variable in HALL_WEATHER_VARIABLES:
                column, combined, sources = f"hall_{variable}", pd.Series(np.nan, index=index), {}
                for station in HALL_WEATHER_STATIONS:
                    values = bucket_15min(frames[station], variable, "AVG").reindex(index)
                    fill = combined.isna() & values.notna()
                    sources[station] = share(fill)
                    combined[fill] = values[fill]
                info.setdefault("hall_weather_sources", {})[column] = sources
                columns[column], kinds[column] = combined.to_numpy(), GROUP_KINDS[group]
    return pd.DataFrame(columns, index=index), kinds, info


# What one forecast may see: rows for the context's timestamps (the frame cut at the origin, then forward-filled,
# as contexts are) and the HORIZON steps from the origin on, with the future of "past" columns blanked.
# The window's attrs["future_columns"] lists the known and oracle columns.
# So a model gets the same input schema for every window of a set
# (an oracle column whose future is all NaN in one window stays a future column).
# Raises when a known column has a missing value (e.g. outside the holiday table).
def covariate_window(frame: pd.DataFrame, kinds: dict[str, str], origin: pd.Timestamp,
                     context_index: pd.DatetimeIndex) -> pd.DataFrame:
    past = frame.loc[: origin - STEP].ffill().reindex(context_index)
    future = target_window(frame, origin)
    future[[column for column, kind in kinds.items() if kind == "past"]] = np.nan
    window = pd.concat([past, future])
    known = [column for column, kind in kinds.items() if kind == "known"]
    if window[known].isna().any().any():
        raise ValueError(f"KNOWN COVARIATES MISSING in the window of origin {origin}")
    window.attrs["future_columns"] = [column for column, kind in kinds.items() if kind != "past"]
    return window


# Observed share of each non-known column in [start, end) (before any filling), for run.json.
def coverage(frame: pd.DataFrame, kinds: dict[str, str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    rows = frame.reindex(pd.date_range(start, end, freq=BUCKET_FREQ, inclusive="left"))
    return {column: round(float(rows[column].notna().mean()), 4) for column, kind in kinds.items() if kind != "known"}


# Known columns of a covariate frame strictly before the cutoff: what a trainer may use for its training items.
# open_share is replaced by its earlier-weeks values (training_open_share, D-E8).
# The frame starts at its first row where every column is defined, and the trainer's items are cut to it
# (runner._fit). A gap after that start is an error.
# series, task: the sensor's target and task, needed only when the frame has open_share.
def training_covariates(frame: pd.DataFrame, kinds: dict[str, str], cutoff: pd.Timestamp,
                        series: Optional[pd.Series] = None,
                        task: Optional[SensorTask] = None) -> Optional[pd.DataFrame]:
    known = [column for column, kind in kinds.items() if kind == "known"]
    if not known:
        return None
    rows = frame.loc[: cutoff - STEP, known].copy()
    if "open_share" in known:
        rows["open_share"] = training_open_share(series, task, cutoff, rows.index)
    defined = rows.notna().all(axis=1).to_numpy()
    if not defined.any():
        raise ValueError("NO TRAINING ROW HAS EVERY KNOWN COVARIATE")
    rows = rows.iloc[int(np.argmax(defined)):]
    if rows.isna().any().any():
        missing = rows.columns[rows.isna().any()].tolist()
        raise ValueError(f"KNOWN COVARIATES MISSING inside the training rows: {missing}")
    return rows
