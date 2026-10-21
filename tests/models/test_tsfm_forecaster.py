# Tests for src/models/tsfm_forecaster.py without torch or weights:
# - the recipe per configured sensor
# - the dropped still-filling bucket at the origin
# - the "0 when closed" mask from the context's own profile
# - the stations covariates (past columns blanked over the horizon, a station without data left out)
# - sorted deciles clipped at 0 for visitor columns only
# - a clear error when chronos isn't installed
# Only the SensBee HTTP call and the Chronos-2 builder are patched (autospec). So the real loader, fill rules,
# covariate window and mask run.
# The integration test loads the real Chronos-2 (bench venv, weights cached): pytest -m integration.

from datetime import datetime
from typing import Callable, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import src.models.tsfm_forecaster as tsfm
from benchmark import covariates as cov
from benchmark.tasks import QUANTILE_LEVELS, TASKS
from src.data_access.sensbee_client import SensbeeClient

STATION, OTHER, THIRD = "MANEBACH_WEATHER_STATION", "STUTZERBACH_WEATHER_STATION", "FRAUENWALD_WEATHER_STATION"


# Every benchmark sensor configured as uuid-<NAME> / key-<NAME>, whatever the local env file holds, and no model cached.
@pytest.fixture(autouse=True)
def configured(monkeypatch):
    for name in TASKS:
        monkeypatch.setenv(f"{name}_SENSOR_UUID", f"uuid-{name}")
        monkeypatch.setenv(f"{name}_MY_API_KEY_READ", f"key-{name}")
    monkeypatch.setattr(tsfm, "_model", None)


# Fake SensBee /data/load with time_grouping: one record per 15-min bucket from floor(from) to floor(to), valued by
# the sensor's function of the bucket label; a sensor without a function has no data.
def _server(values: dict[str, Callable[[pd.Timestamp], float]]):
    def get_sensor_data(self, sensor_id, api_key=None, from_time=None, to_time=None, limit=None, ordering="ASC",
                        cols=None, time_grouping_seconds=None) -> list[dict]:
        column = cols[0].split(".")[0]
        if sensor_id not in values:
            return []
        buckets = pd.date_range(pd.Timestamp(from_time).floor("15min"), pd.Timestamp(to_time).floor("15min"),
                                freq="15min")
        return [{"grouped_time": t.strftime("%Y-%m-%dT%H:%M:%S"), column: values[sensor_id](t)} for t in buckets]
    return get_sensor_data


# A stand-in for the Chronos-2 adapter: records its calls and returns `deciles` (default 7.0) at every step.
def _model(calls: list, deciles: Optional[list[float]] = None):
    def forecast(contexts, horizon, levels, seed, covariates=None):
        calls.append((contexts, covariates))
        row = np.full(len(levels), 7.0) if deciles is None else np.array(deciles, dtype=float)
        return np.tile(row, (len(contexts), horizon, 1)), ["ok"] * len(contexts)
    return forecast


# Runs forecast_chronos2 against the fake server and model.
def _forecast(values: dict, calls: list, sensor_id: str, column: str, now: datetime, history_days: int = 1,
              deciles: Optional[list[float]] = None) -> dict:
    with patch.object(SensbeeClient, "get_sensor_data", autospec=True, side_effect=_server(values)), \
         patch.object(tsfm, "chronos2_forecaster", autospec=True, return_value=(_model(calls, deciles), {})):
        return tsfm.forecast_chronos2(sensor_id, "read-key", column, 96, history_days, now=now)


@pytest.mark.parametrize("sensor_id, column, recipe", [
    ("uuid-EISHALLE", "visitors_total", "mask"), ("uuid-SCHWIMMHALLE", "visitors_total", "mask"),
    ("uuid-EISHALLE", "visitor_change", "plain"), (f"uuid-{STATION}", "temperature", "stations"),
    (f"uuid-{STATION}", "humidity", "plain"), ("uuid-PARKPLATZ_EAZ", "Fahrzeuge_auf_Parkplatz", "plain"),
    ("some-other-sensor", "temperature", "plain")])
def test_the_recipe_follows_the_configured_sensor_and_column(sensor_id, column, recipe):
    assert tsfm.recipe_for(sensor_id, column)[0] == recipe


def test_the_still_filling_bucket_at_the_origin_is_dropped():
    calls = []
    out = _forecast({"x": lambda t: 999.0 if t == pd.Timestamp("2026-02-10 10:00") else 1.0}, calls, "x",
                    "temperature", now=datetime(2026, 2, 10, 10, 7))
    ((context,), covariates), = calls
    assert covariates is None and len(context) == 96 and (context == 1.0).all()
    assert context.index[-1] == pd.Timestamp("2026-02-10 09:45")
    assert out["timestamps"][0] == "2026-02-10T10:00:00" and out["last_data_timestamp"] == "2026-02-10T09:45:00"
    assert out["model_info"]["recipe"] == "plain" and out["model_info"]["origin"] == "2026-02-10T10:00:00"


# Visitors only on Mondays 09:00-09:45 Berlin time (08:00-08:45 UTC in winter); the forecast starts on Monday 9 Feb
# 00:00 UTC, so with 7 days of history the context holds one such Monday and only steps 32-35 stay open.
def test_a_hall_forecast_is_zero_where_the_context_profile_is_closed():
    local = lambda t: cov.berlin(pd.DatetimeIndex([t]))[0]  # noqa: E731
    calls = []
    out = _forecast({"uuid-EISHALLE": lambda t: 5.0 if (local(t).weekday() == 0 and local(t).hour == 9) else 0.0},
                    calls, "uuid-EISHALLE", "visitors_total", now=datetime(2026, 2, 9, 0, 7), history_days=7)
    expected = np.zeros(96)
    expected[32:36] = 7.0
    assert np.array_equal(out["quantiles"], np.tile(expected[:, None], (1, len(QUANTILE_LEVELS))))
    assert out["model_info"]["threshold"] == tsfm.THRESHOLD and out["model_info"]["masked_steps"] == 92


def test_a_station_gets_the_other_stations_as_past_covariates():
    calls = []
    now = datetime(2026, 2, 10, 10, 7)
    out = _forecast({f"uuid-{STATION}": lambda t: 1.0, f"uuid-{OTHER}": lambda t: t.hour + t.minute / 60}, calls,
                    f"uuid-{STATION}", "temperature", now=now)  # the third station has no data
    ((context,), (window,)), = calls
    column = f"temperature_{OTHER}"
    assert list(window.columns) == [*cov.SUN, column] and window.attrs["future_columns"] == list(cov.SUN)
    past, future = window.iloc[:len(context)], window.iloc[len(context):]
    assert np.allclose(past[column], context.index.hour + context.index.minute / 60)
    assert future[column].isna().all() and future.index[0] == pd.Timestamp("2026-02-10 10:00")
    assert out["model_info"]["covariate_stations"] == [OTHER]
    assert out["model_info"]["stations_left_out"] == [THIRD]


def test_an_unconfigured_station_is_left_out_without_a_request(monkeypatch):
    monkeypatch.delenv(f"{THIRD}_SENSOR_UUID")
    calls = []
    with patch.object(SensbeeClient, "get_sensor_data", autospec=True,
                      side_effect=_server({f"uuid-{STATION}": lambda t: 1.0, f"uuid-{OTHER}": lambda t: 2.0})) as get, \
         patch.object(tsfm, "chronos2_forecaster", autospec=True, return_value=(_model(calls), {})):
        out = tsfm.forecast_chronos2(f"uuid-{STATION}", "k", "temperature", 96, 1, now=datetime(2026, 2, 10, 10, 7))
    assert {c.kwargs["sensor_id"] for c in get.call_args_list} == {f"uuid-{STATION}", f"uuid-{OTHER}"}
    assert out["model_info"]["stations_left_out"] == [THIRD]


@pytest.mark.parametrize("column, low", [("visitors_total", 0.0), ("temperature", -9.0)])
def test_deciles_are_sorted_and_clipped_at_zero_for_visitor_columns_only(column, low):
    crossing = [3.0, -9.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 7.0]
    out = _forecast({"x": lambda t: 1.0}, [], "x", column, now=datetime(2026, 2, 10, 10, 7), deciles=crossing)
    expected = np.maximum(np.sort(crossing), low)
    assert np.array_equal(out["quantiles"], np.tile(expected, (96, 1)))


def test_a_non_finite_forecast_is_an_error():
    with pytest.raises(RuntimeError, match="INVALID FORECAST"):
        _forecast({"x": lambda t: 1.0}, [], "x", "temperature", now=datetime(2026, 2, 10, 10, 7),
                  deciles=[np.nan] * 9)


def test_a_missing_chronos_install_is_a_clear_error():
    with patch.object(SensbeeClient, "get_sensor_data", autospec=True, side_effect=_server({"x": lambda t: 1.0})), \
         patch.object(tsfm, "chronos2_forecaster", autospec=True, side_effect=ImportError("No module named 'torch'")):
        with pytest.raises(ValueError, match="CHRONOS-2 NOT INSTALLED"):
            tsfm.forecast_chronos2("x", "k", "temperature", 96, 1, now=datetime(2026, 2, 10, 10, 7))


# The real model through the service path, on the RP's 14-day Stützerbach fixture served by the fake server.
@pytest.mark.integration
def test_the_real_chronos2_returns_sorted_finite_deciles():
    pytest.importorskip("chronos")
    from benchmark.models import CHRONOS2
    from benchmark.models.chronos_models import chronos2_forecaster
    from src.data_access.data_loader import load_sensor_series_from_json
    series = load_sensor_series_from_json("data/temp_14day_sensbee_data.json", column_name="temperature")
    now = (series.index[-1] + pd.Timedelta("22min")).to_pydatetime()  # the bucket after the last one is filling
    with patch.object(SensbeeClient, "get_sensor_data", autospec=True,
                      side_effect=_server({"x": lambda t: float(series.get(t, np.nan))})):
        out = tsfm.forecast_chronos2("x", "k", "temperature", 96, 7, now=now)
    assert out["quantiles"].shape == (96, 9) and np.isfinite(out["quantiles"]).all()
    assert (np.diff(out["quantiles"], axis=1) >= 0).all() and out["input_points"] == 7 * 96
    forecast, _ = chronos2_forecaster(*CHRONOS2)
    direct, _ = forecast([series.iloc[-7 * 96:]], 96, QUANTILE_LEVELS, 0)
    assert np.allclose(out["quantiles"], np.sort(direct[0], axis=-1))
