# Tests for the /forecast endpoint through FastAPI's TestClient: request validation, the new Chronos-2 fields, and
# the unchanged LLM response. The real forecast_single_column runs; only the SensBee loader, the LLM call and the
# Chronos-2 forecaster are patched (autospec), so no network, model or paid API is used.

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
