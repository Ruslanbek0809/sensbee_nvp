# Tests for the /forecast endpoint through FastAPI's TestClient: request validation, the new Chronos-2 fields, the
# unchanged LLM response, the model preload at startup, and the security rules (no key fallback, no key in responses
# or logs, cache access by key, the provider and model allow-list). The real forecast_single_column runs; only the SensBee
# loader, the LLM call and the Chronos-2 forecaster are patched (autospec), so no network, model or paid API is used.

import logging
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.models.tsfm_forecaster as tsfm
from src.data_access.sensbee_client import SensbeeClient
from src.service import forecast_jobs
from src.service.main import app, forecast_cache

client = TestClient(app)
LLM_FIELDS = {"sensor_id", "column", "forecast_values", "forecast_timestamps", "input_points", "forecast_points",
              "horizon_hours", "last_data_timestamp", "generated_at", "provider", "use_normalization",
              "include_context", "cached"}
NEW_FIELDS = {"forecast_lower", "forecast_upper", "forecast_quantiles", "model_info"}


# Every test starts and ends with an empty cache.
@pytest.fixture(autouse=True)
def empty_cache():
    forecast_cache.clear()
    yield
    forecast_cache.clear()


# Every test may use the hosted providers (off unless HOSTED_PROVIDERS lists them); the switch has its own test.
@pytest.fixture(autouse=True)
def hosted_providers_enabled(monkeypatch):
    monkeypatch.setenv("HOSTED_PROVIDERS", "groq, openai, mistral")


# What forecast_chronos2 returns for `steps` steps: increasing deciles 0..8 plus the step number.
def _chronos2_result(steps: int) -> dict:
    timestamps = [t.isoformat() for t in pd.date_range("2026-02-10 10:00", periods=steps, freq="15min")]
    quantiles = np.arange(9.0)[None, :] + np.arange(steps)[:, None]
    return {"timestamps": timestamps, "quantiles": quantiles, "input_points": 2688,
            "last_data_timestamp": "2026-02-10T09:45:00",
            "model_info": {"model_id": "amazon/chronos-2", "recipe": "stations", "covariate_stations": ["X"]}}


# POSTs a forecast request for one sensor's temperature, with `fields` added.
def _post(**fields) -> httpx.Response:
    return client.post("/forecast", json={"sensor_id": "sensor-uuid", "column": "temperature", **fields})


def test_chronos2_response_has_the_median_the_interval_and_the_model():
    with patch.object(forecast_jobs, "forecast_chronos2", autospec=True, return_value=_chronos2_result(96)) as tsfm:
        response = _post(provider="chronos-2", horizon_hours=24)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == LLM_FIELDS | NEW_FIELDS
    assert tsfm.call_args.args[3:] == (96, 28)
    assert body["provider"] == "chronos-2" and body["forecast_points"] == 96
    assert body["forecast_values"] == body["forecast_quantiles"]["0.5"]
    assert all(lo <= mid <= hi for lo, mid, hi in zip(body["forecast_lower"], body["forecast_values"],
                                                       body["forecast_upper"]))
    assert body["model_info"]["recipe"] == "stations"
    cached = client.get("/forecast/cache/sensor-uuid/temperature").json()
    assert cached["cached"] is True and cached["forecast_quantiles"] == body["forecast_quantiles"]


def test_llm_response_is_unchanged_and_has_no_quantiles():
    series = pd.Series(np.linspace(-2.0, 3.0, 96), index=pd.date_range("2026-02-10", periods=96, freq="15min"))
    with patch.object(forecast_jobs, "load_sensor_series_from_api", autospec=True, return_value=series) as load, \
         patch.object(forecast_jobs, "nvp_llms_forecast", autospec=True, return_value=np.zeros(24)):
        response = _post(horizon_hours=6)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == LLM_FIELDS | NEW_FIELDS
    assert all(body[field] is None for field in NEW_FIELDS)
    assert body["provider"] == "groq" and body["forecast_values"] == [0.0] * 24
    assert load.call_args.kwargs["window_hours"] == 7 * 24  # the old default of 7 days


@pytest.mark.parametrize("provider", ["groq", "chronos-2"])
@pytest.mark.parametrize("history_days", [0, 29])
def test_history_days_outside_1_to_28_is_rejected(provider, history_days):
    assert _post(provider=provider, history_days=history_days).status_code == 422


def test_llm_provider_with_more_than_14_days_is_a_bad_request():
    with patch.object(forecast_jobs, "load_sensor_series_from_api", autospec=True) as load:
        response = _post(provider="groq", history_days=20)

    assert response.status_code == 400 and "too long for LLM providers" in response.json()["detail"]
    load.assert_not_called()


# The whole path below the endpoint, with only the SensBee HTTP call and the Chronos-2 builder patched: a configured
# hall gets the "0 when closed" recipe, and its model_info comes back as plain JSON.
def test_a_hall_forecast_goes_through_the_real_forecaster(monkeypatch):
    monkeypatch.setenv("EISHALLE_SENSOR_UUID", "hall-uuid")
    monkeypatch.setattr(tsfm, "_model", None)

    def get_sensor_data(self, sensor_id, api_key=None, from_time=None, to_time=None, limit=None, ordering="ASC",
                        cols=None, time_grouping_seconds=None) -> list[dict]:
        buckets = pd.date_range(pd.Timestamp(from_time).floor("15min"), pd.Timestamp(to_time).floor("15min"),
                                freq="15min")
        return [{"grouped_time": t.strftime("%Y-%m-%dT%H:%M:%S"), "visitors_total": 5.0} for t in buckets]

    def forecast(contexts, horizon, levels, seed, covariates=None):
        return np.full((len(contexts), horizon, len(levels)), 3.0), ["ok"] * len(contexts)

    with patch.object(SensbeeClient, "get_sensor_data", autospec=True, side_effect=get_sensor_data), \
         patch.object(tsfm, "chronos2_forecaster", autospec=True, return_value=(forecast, {})):
        response = client.post("/forecast", json={"sensor_id": "hall-uuid", "column": "visitors_total",
                                                  "provider": "chronos-2", "history_days": 7})

    assert response.status_code == 200
    body = response.json()
    assert body["model_info"]["recipe"] == "mask" and body["model_info"]["masked_steps"] == 0  # open every slot
    assert body["input_points"] == 7 * 96 and body["forecast_values"] == [3.0] * 96


def test_chronos2_without_its_install_is_a_bad_request():
    with patch.object(forecast_jobs, "forecast_chronos2", autospec=True,
                      side_effect=ValueError("CHRONOS-2 NOT INSTALLED (install requirements-tsfm.txt)")):
        response = _post(provider="chronos-2")

    assert response.status_code == 400 and "NOT INSTALLED" in response.json()["detail"]


# With chronos-forecasting installed (the WITH_TSFM build) the model is loaded once at startup; without it, never.
@pytest.mark.parametrize("installed", [True, False])
def test_startup_preloads_chronos2_only_when_installed(monkeypatch, installed):
    monkeypatch.setattr(tsfm, "_model", None)
    with patch("src.service.main.find_spec", autospec=True, return_value=object() if installed else None), \
         patch.object(tsfm, "chronos2_forecaster", autospec=True, return_value=(lambda *args: None, {})) as build:
        with TestClient(app):
            pass
    assert build.call_count == (1 if installed else 0)


CALLER_KEY = "CALLER-KEY-123"
SERVER_KEY = "DUMMY-SERVER-KEY"
STATION_KEY = "DUMMY-STATION-KEY"


# Removes every configured sensor from the environment (python-dotenv may have loaded real ones) and sets a dummy
# server key, so a test sees only dummy keys, also in its failure output.
@pytest.fixture
def dummy_keys(monkeypatch):
    for name in tsfm.TASKS:
        monkeypatch.delenv(f"{name}_SENSOR_UUID", raising=False)
        monkeypatch.delenv(f"{name}_MY_API_KEY_READ", raising=False)
    monkeypatch.setattr("src.data_access.sensbee_client.DEFAULT_API_KEY", SERVER_KEY)


# SensBee answering every load with HTTP 500. Only the transport is patched, so httpx's client code and log run.
def _upstream_500(self, request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, request=request)


@pytest.mark.parametrize("provider", ["groq", "chronos-2"])
def test_a_request_without_a_key_sends_no_key_upstream(dummy_keys, provider):
    def get_sensor_data(self, sensor_id, api_key=None, from_time=None, to_time=None, limit=None, ordering="ASC",
                        cols=None, time_grouping_seconds=None) -> list[dict]:
        buckets = pd.date_range(pd.Timestamp(from_time).floor("15min"), pd.Timestamp(to_time).floor("15min"),
                                freq="15min")
        return [{"grouped_time": t.strftime("%Y-%m-%dT%H:%M:%S"), "temperature": 5.0} for t in buckets]

    def forecast(contexts, horizon, levels, seed, covariates=None):
        return np.full((len(contexts), horizon, len(levels)), 3.0), ["ok"] * len(contexts)

    with patch.object(SensbeeClient, "get_sensor_data", autospec=True, side_effect=get_sensor_data) as get, \
         patch.object(forecast_jobs, "nvp_llms_forecast", autospec=True, return_value=np.zeros(24)), \
         patch.object(tsfm, "chronos2_forecaster", autospec=True, return_value=(forecast, {})):
        response = _post(provider=provider, horizon_hours=6)

    assert response.status_code == 200
    assert [call.kwargs["api_key"] for call in get.call_args_list] == [None]


# An upstream error must not put a key into the response or any log line (httpx's request line included): not the
# caller's, not the server's, and not the other stations' keys that the stations recipe reads with.
@pytest.mark.parametrize("provider, sensor_id, api_key", [
    ("groq", "sensor-uuid", None),
    ("groq", "sensor-uuid", CALLER_KEY),
    ("chronos-2", "sensor-uuid", None),
    ("chronos-2", "manebach-uuid", CALLER_KEY),  # stations recipe: the other station is read with STATION_KEY
])
def test_an_upstream_error_leaks_no_key(dummy_keys, monkeypatch, caplog, provider, sensor_id, api_key):
    monkeypatch.setenv("MANEBACH_WEATHER_STATION_SENSOR_UUID", "manebach-uuid")
    monkeypatch.setenv("MANEBACH_WEATHER_STATION_MY_API_KEY_READ", CALLER_KEY)
    monkeypatch.setenv("STUTZERBACH_WEATHER_STATION_SENSOR_UUID", "stutzerbach-uuid")
    monkeypatch.setenv("STUTZERBACH_WEATHER_STATION_MY_API_KEY_READ", STATION_KEY)
    caplog.set_level(logging.INFO)
    with patch.object(httpx.HTTPTransport, "handle_request", autospec=True, side_effect=_upstream_500) as send:
        response = client.post("/forecast", json={"sensor_id": sensor_id, "column": "temperature",
                                                  "provider": provider, "api_key": api_key})

    assert response.status_code == 502
    sent_keys = {call.args[1].url.params.get("key") for call in send.call_args_list}
    assert sent_keys == ({api_key, STATION_KEY} if sensor_id == "manebach-uuid" else {api_key})
    for key in (CALLER_KEY, SERVER_KEY, STATION_KEY):
        assert key not in response.text and key not in caplog.text


# POSTs a Chronos-2 forecast for one sensor's temperature (the forecaster patched), with or without a READ key.
def _cache_forecast(sensor_id: str, api_key=None) -> None:
    with patch.object(forecast_jobs, "forecast_chronos2", autospec=True, return_value=_chronos2_result(96)):
        response = client.post("/forecast", json={"sensor_id": sensor_id, "column": "temperature",
                                                  "provider": "chronos-2", "api_key": api_key})
    assert response.status_code == 200


# The header that carries the READ key for the cache endpoints.
def _key_header(key) -> dict:
    return {} if key is None else {"X-API-Key": key}


@pytest.mark.parametrize("key", [None, "WRONG-KEY"])
def test_a_cache_read_without_the_sensors_key_is_refused(key):
    _cache_forecast("private-uuid", CALLER_KEY)

    refused = client.get("/forecast/cache/private-uuid/temperature", headers=_key_header(key))
    allowed = client.get("/forecast/cache/private-uuid/temperature", headers=_key_header(CALLER_KEY))

    assert refused.status_code == 404 and "forecast_values" not in refused.text
    assert allowed.status_code == 200 and allowed.json()["cached"] is True


# The list and delete endpoints see only the entries made with the caller's key (keyless callers: keyless entries).
def test_cache_list_and_delete_see_only_the_callers_entries():
    _cache_forecast("private-uuid", CALLER_KEY)
    _cache_forecast("public-uuid")

    listed = client.get("/forecast/cache").json()
    listed_with_key = client.get("/forecast/cache", headers=_key_header(CALLER_KEY)).json()
    cleared = client.delete("/forecast/cache").json()

    assert listed["count"] == 1 and [item["sensor_id"] for item in listed["forecasts"]] == ["public-uuid"]
    assert [item["sensor_id"] for item in listed_with_key["forecasts"]] == ["private-uuid"]
    assert cleared["cleared"] == 1
    assert client.get("/forecast/cache/private-uuid/temperature", headers=_key_header(CALLER_KEY)).status_code == 200


# Unknown providers and models are refused before any data is loaded or any model is called. A hosted provider takes
# only the server's configured model; "local" only its presets (any other name would be loaded from the Hub with
# trust_remote_code); "chronos-2" no model name.
@pytest.mark.parametrize("provider, model", [
    ("no-such-provider", None),
    ("groq", "some-other-model"),
    ("mistral", "some-other-model"),
    ("openai", "some-other-model"),
    ("local", "someone/any-hub-repo"),
    ("chronos-2", "some-other-model"),
])
def test_an_unknown_provider_or_model_is_refused(monkeypatch, provider, model):
    monkeypatch.setenv("GROQ_MODEL", "configured-groq-model")
    with patch.object(forecast_jobs, "load_sensor_series_from_api", autospec=True) as load, \
         patch.object(forecast_jobs, "forecast_chronos2", autospec=True) as chronos2, \
         patch.object(forecast_jobs, "nvp_llms_forecast", autospec=True) as llm:
        response = _post(provider=provider, model=model)

    assert 400 <= response.status_code < 500
    load.assert_not_called()
    chronos2.assert_not_called()
    llm.assert_not_called()


@pytest.mark.parametrize("provider, model", [("groq", None), ("GROQ", None), ("groq", "configured-groq-model"),
                                             ("local", "mistral-7b")])
def test_the_configured_model_and_the_local_presets_are_allowed(monkeypatch, provider, model):
    monkeypatch.setenv("GROQ_MODEL", "configured-groq-model")
    series = pd.Series(np.linspace(-2.0, 3.0, 96), index=pd.date_range("2026-02-10", periods=96, freq="15min"))
    with patch.object(forecast_jobs, "load_sensor_series_from_api", autospec=True, return_value=series), \
         patch.object(forecast_jobs, "nvp_llms_forecast", autospec=True, return_value=np.zeros(24)) as llm:
        response = _post(provider=provider, model=model, horizon_hours=6)

    assert response.status_code == 200 and llm.call_args.kwargs["model"] == model


# Hosted providers spend the server's paid keys, so they are refused unless HOSTED_PROVIDERS lists them.
@pytest.mark.parametrize("enabled, provider, allowed", [
    (None, "groq", False),
    ("", "mistral", False),
    ("mistral", "groq", False),
    ("mistral", "mistral", True),
    (" Groq ,openai", "openai", True),
])
def test_hosted_providers_are_off_unless_enabled(monkeypatch, enabled, provider, allowed):
    if enabled is None:
        monkeypatch.delenv("HOSTED_PROVIDERS")
    else:
        monkeypatch.setenv("HOSTED_PROVIDERS", enabled)
    series = pd.Series(np.linspace(-2.0, 3.0, 96), index=pd.date_range("2026-02-10", periods=96, freq="15min"))
    with patch.object(forecast_jobs, "load_sensor_series_from_api", autospec=True, return_value=series) as load, \
         patch.object(forecast_jobs, "nvp_llms_forecast", autospec=True, return_value=np.zeros(24)):
        response = _post(provider=provider, horizon_hours=6)

    assert response.status_code == (200 if allowed else 400)
    assert load.called is allowed
