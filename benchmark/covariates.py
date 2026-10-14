# Covariates for the benchmark harness (Phase 4): named covariate sets, the covariate frame of a sensor on its 15-min
# grid, and the window one forecast may see. Every column has a kind:
# - "known": known before the origin for the whole horizon (calendar, holidays, solar elevation, and the opening
#   profile learned from the data before the cutoff);
# - "past": observed values (other weather variables, other stations), visible only before the origin;
# - "oracle": observed future values, an upper bound only (reported apart from the other models).
# covariate_window() is the one place that decides what a forecast sees: the frame is cut at the origin before it is
# forward-filled (the context rule of benchmark/tasks.py), and the future rows of "past" columns are blanked. Only an
# oracle set has oracle columns, whose future rows keep the observed values (gaps stay NaN).
#
# Calendar features use Europe/Berlin time (SensBee timestamps are naive UTC). Holidays come from a dated table that
# covers HOLIDAYS_FROM to HOLIDAYS_TO; calendar frames are built only inside that range, and a window that needs a row
# outside it raises instead of getting a silent 0.
# - Public holidays: Thüringer Feier- und Gedenktagsgesetz (ThürFGtG) § 2(1), statewide: New Year, Good Friday, Easter
#   Monday, 1 May, Ascension, Whit Monday, 20 September (Weltkindertag), 3 October, Reformation Day (31 October),
#   Christmas Day and 26 December (umwelt-online.de, last amendment 26.03.2019, checked 2026-10-14). Corpus Christi
#   is a holiday only where an ordinance names a mainly Catholic municipality (§ 2(2)), not Ilmenau, so it is left out.
#   Easter 2026 is 5 April (tests/benchmark/test_covariates.py computes it).
# - School holidays: KMK "Ferien im Schuljahr 2025/2026", Thüringen row, Stand 09.10.2025
#   (kmk.org/fileadmin/Dateien/pdf/Ferienkalender/FER2025_26.pdf, checked 2026-10-14): first and last holiday day;
#   15.05.2026 is a school-free day.
# Solar elevation: NOAA Global Monitoring Division, "General Solar Position Calculations" (gml.noaa.gov/grad/solcalc/
# solareqns.PDF, checked 2026-10-14), at one point for Ilmenau (de.wikipedia.org/wiki/Ilmenau: 50.687 N, 10.914 E,
# checked 2026-10-14); the weather stations lie within about 10 km, which moves the sun angle by a fraction of a degree.
# The opening profile (D4-1): per Berlin weekday and 15-min local slot, the share of days with visitors_total > 0 in the
# hall's training data before the cutoff (from task.train_start, without the task's exclusions), fitted once per sensor
# and period like a trained model.

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from benchmark.snapshot import BUCKET_FREQ, bucket_15min
from benchmark.tasks import HORIZON, STEP, TASKS, SensorTask, _excluded, training_series

logger = logging.getLogger(__name__)

KINDS = ("known", "past", "oracle")
BERLIN = "Europe/Berlin"
ILMENAU = (50.687222, 10.914167)  # latitude, longitude (degrees north, east)
HOLIDAYS_FROM, HOLIDAYS_TO = pd.Timestamp("2025-10-01"), pd.Timestamp("2026-10-01")  # [from, to), Berlin dates
PUBLIC_HOLIDAYS = tuple(pd.Timestamp(d) for d in (
    "2025-10-03", "2025-10-31", "2025-12-25", "2025-12-26", "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-01",
    "2026-05-14", "2026-05-25", "2026-09-20"))
SCHOOL_HOLIDAYS = tuple((pd.Timestamp(a), pd.Timestamp(b)) for a, b in (  # first and last day, inclusive
    ("2025-10-06", "2025-10-18"), ("2025-12-22", "2026-01-03"), ("2026-02-16", "2026-02-21"),
    ("2026-04-07", "2026-04-17"), ("2026-05-15", "2026-05-15"), ("2026-07-04", "2026-08-14")))
WEATHER_VARIABLES = ("humidity", "global_radiation", "pressure_rel", "wind_speed_avg")  # fixed before any run
STATIONS = tuple(name for name, task in TASKS.items() if task.kind == "regular")
CALENDAR = ("tod_sin", "tod_cos", "dow_sin", "dow_cos", "public_holiday", "school_holiday")
SUN = ("tod_sin", "tod_cos", "solar_elevation")


# A named covariate set: the task kind it applies to ("visitors" or "regular") and its feature groups.
@dataclass(frozen=True)
class CovariateSet:
    kind: str
    groups: tuple[str, ...]

    @property
    def oracle(self) -> bool:
        return "met_oracle" in self.groups

    # Kinds of the columns this set produces.
    @property
    def column_kinds(self) -> tuple[str, ...]:
        kinds = {"calendar": "known", "open": "known", "sun": "known", "met": "past", "stations": "past",
                 "met_oracle": "oracle"}
        return tuple(sorted({kinds[group] for group in self.groups}))


# Fixed before any run (Phase 4 plan, section A).
COVARIATE_SETS: dict[str, CovariateSet] = {
    "cal": CovariateSet("visitors", ("calendar",)),
    "cal_open": CovariateSet("visitors", ("calendar", "open")),
    "sun": CovariateSet("regular", ("sun",)),
    "met": CovariateSet("regular", ("sun", "met")),
    "stations": CovariateSet("regular", ("sun", "stations")),
    "met_oracle": CovariateSet("regular", ("sun", "met_oracle")),
}


# Naive-UTC timestamps as Europe/Berlin local time.
def berlin(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_localize("UTC").tz_convert(BERLIN)


# Calendar features in Berlin time: time of day and weekday as sin/cos, and the public and school holiday flags of the
# local date. Raises when a timestamp's local date lies outside the holiday table.
def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = berlin(index)
    days = local.tz_localize(None).normalize()
    if len(days) and (days.min() < HOLIDAYS_FROM or days.max() >= HOLIDAYS_TO):
        raise ValueError(f"DATES {days.min().date()}..{days.max().date()} OUTSIDE THE HOLIDAY TABLE "
                         f"{HOLIDAYS_FROM.date()}..{HOLIDAYS_TO.date()}")
    tod = 2 * np.pi * (local.hour * 60 + local.minute).to_numpy() / 1440
    dow = 2 * np.pi * local.weekday.to_numpy() / 7
    school = np.zeros(len(index), dtype=bool)
    for first, last in SCHOOL_HOLIDAYS:
        school |= (days >= first) & (days <= last)
    return pd.DataFrame({"tod_sin": np.sin(tod), "tod_cos": np.cos(tod), "dow_sin": np.sin(dow),
                         "dow_cos": np.cos(dow), "public_holiday": days.isin(PUBLIC_HOLIDAYS).astype(float),
                         "school_holiday": school.astype(float)}, index=index)


# Solar elevation in degrees at naive-UTC timestamps (NOAA general solar position equations, see the header).
def solar_elevation(index: pd.DatetimeIndex, latitude: float = ILMENAU[0],
                    longitude: float = ILMENAU[1]) -> np.ndarray:
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


# Opening profile of a hall: [7 weekdays, 96 slots] share of days with visitors > 0 in the training data before the
# cutoff (from task.train_start, without the task's exclusions).
def open_profile(series: pd.Series, task: SensorTask, cutoff: pd.Timestamp) -> np.ndarray:
    history = training_series(series, cutoff)
    if task.train_start is not None:
        history = history.loc[pd.Timestamp(task.train_start):]
    history = history[~_excluded(history.index, task)]
    weekday, slot = _local_slots(history.index)
    share = (history.to_numpy() > 0).astype(float)
    profile = pd.Series(share).groupby([weekday, slot]).mean()
    return profile.reindex(pd.MultiIndex.from_product([range(7), range(96)])).to_numpy().reshape(7, 96)


# The profile value of each timestamp (its Berlin weekday and slot).
def open_share(index: pd.DatetimeIndex, profile: np.ndarray) -> np.ndarray:
    weekday, slot = _local_slots(index)
    return profile[weekday, slot]


# The covariate frame of one sensor for a set: every column on the sensor's grid (calendar sets only inside the
# holiday table), its kind, and info for run.json (the opening profile). cutoff: the period start, where the opening
# profile is fitted. Weather columns are SensBee AVG buckets, unfilled.
def build(name: str, sensor: str, frames: dict[str, pd.DataFrame], series: pd.Series, task: SensorTask,
          cutoff: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, str], dict]:
    spec = COVARIATE_SETS[name]
    if task.kind != spec.kind:
        raise ValueError(f"COVARIATE SET {name} IS FOR {spec.kind} SENSORS, NOT {sensor} ({task.kind})")
    index = series.index
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
                columns[column], kinds[column] = features[column].to_numpy(), "known"
        elif group == "open":
            profile = open_profile(series, task, cutoff)
            columns["open_share"], kinds["open_share"] = open_share(index, profile), "known"
            info["open_profile"] = np.round(profile, 4).tolist()
        elif group == "sun":
            local = berlin(index)
            tod = 2 * np.pi * (local.hour * 60 + local.minute).to_numpy() / 1440
            columns["tod_sin"], columns["tod_cos"] = np.sin(tod), np.cos(tod)
            columns["solar_elevation"] = np.asarray(solar_elevation(index))
            kinds.update({column: "known" for column in SUN})
        elif group in ("met", "met_oracle"):
            kind = "oracle" if group == "met_oracle" else "past"
            for variable in WEATHER_VARIABLES:
                columns[variable] = bucket_15min(frames[sensor], variable, "AVG").reindex(index).to_numpy()
                kinds[variable] = kind
        elif group == "stations":
            for station in STATIONS:
                if station != sensor:
                    column = f"temperature_{station}"
                    columns[column] = bucket_15min(frames[station], "temperature", "AVG").reindex(index).to_numpy()
                    kinds[column] = "past"
    return pd.DataFrame(columns, index=index), kinds, info


# What one forecast may see: rows for the context's timestamps (the frame cut at the origin, then forward-filled, as
# contexts are) and the HORIZON steps from the origin on, with the future of "past" columns blanked. The window's
# attrs["future_columns"] lists the known and oracle columns, so a model gets the same input schema for every window
# of a set (an oracle column whose future is all NaN in one window stays a future column). Raises when a known column
# has a missing value (e.g. outside the holiday table).
def covariate_window(frame: pd.DataFrame, kinds: dict[str, str], origin: pd.Timestamp,
                     context_index: pd.DatetimeIndex) -> pd.DataFrame:
    past = frame.loc[: origin - STEP].ffill().reindex(context_index)
    future = frame.reindex(pd.date_range(origin, periods=HORIZON, freq=BUCKET_FREQ))
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
def training_covariates(frame: pd.DataFrame, kinds: dict[str, str],
                        cutoff: pd.Timestamp) -> Optional[pd.DataFrame]:
    known = [column for column, kind in kinds.items() if kind == "known"]
    return frame.loc[: cutoff - STEP, known] if known else None
