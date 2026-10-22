# Tests for forecast_single_column(): the requested history must reach the SensBee client as a time window.
# provider "chronos-2" goes to the Chronos-2 forecaster instead of the LLM, and returns the median plus the deciles.
# history_days defaults to 7 days for LLM providers and 28 for Chronos-2; LLM providers refuse more than 14.
# The SensBee loader is patched with autospec, so a call with parameters it doesn't accept fails like in production.

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.service import forecast_jobs


# Builds a 15-min series like the SensBee client returns.
def _series(points: int = 96) -> pd.Series:
    index = pd.date_range("2026-02-10 00:00", periods=points, freq="15min")
    return pd.Series(np.linspace(-2.0, 3.0, points), index=index)


def test_history_days_is_passed_as_window_hours():
    with patch.object(forecast_jobs, "load_sensor_series_from_api", autospec=True, return_value=_series()) as load, \
         patch.object(forecast_jobs, "nvp_llms_forecast", autospec=True, return_value=np.zeros(24)):
        forecast_jobs.forecast_single_column(
            sensor_id="sensor-uuid",
            api_key="read-key",
            column_name="temperature",
            horizon_hours=6,
            history_days=7,
        )

    kwargs = load.call_args.kwargs
    assert kwargs["window_hours"] == 7 * 24
    assert "limit" not in kwargs
    assert kwargs["sensor_id"] == "sensor-uuid"
    assert kwargs["api_key"] == "read-key"
    assert kwargs["column_name"] == "temperature"


def test_forecast_timestamps_continue_after_last_observation():
    series = _series()
    with patch.object(forecast_jobs, "load_sensor_series_from_api", autospec=True, return_value=series), \
         patch.object(forecast_jobs, "nvp_llms_forecast", autospec=True, return_value=np.zeros(24)) as forecast:
        result = forecast_jobs.forecast_single_column(
            sensor_id="sensor-uuid",
            api_key=None,
            column_name="temperature",
            horizon_hours=6,
            history_days=1,
        )

    assert forecast.call_args.kwargs["horizon"] == 24  # 6 h at 15-min steps
    assert result["forecast_points"] == 24
    assert result["forecast_timestamps"][0] == (series.index[-1] + pd.Timedelta(minutes=15)).isoformat()


# What forecast_chronos2 returns for `steps` steps: decile i at step t is 10 * i + t, so every column is distinct.
def _chronos2_result(steps: int) -> dict:
    timestamps = [t.isoformat() for t in pd.date_range("2026-02-10 10:00", periods=steps, freq="15min")]
    quantiles = 10.0 * np.arange(9)[None, :] + np.arange(steps)[:, None]
    return {"timestamps": timestamps, "quantiles": quantiles, "input_points": 2688,
            "last_data_timestamp": "2026-02-10T09:45:00", "model_info": {"recipe": "plain"}}


def test_chronos2_returns_the_median_and_the_deciles_without_the_llm():
    with patch.object(forecast_jobs, "forecast_chronos2", autospec=True, return_value=_chronos2_result(24)) as tsfm, \
         patch.object(forecast_jobs, "nvp_llms_forecast", autospec=True) as llm:
        result = forecast_jobs.forecast_single_column(
            sensor_id="sensor-uuid", api_key="read-key", column_name="temperature", horizon_hours=6,
            history_days=None, provider="chronos-2")

    tsfm.assert_called_once_with("sensor-uuid", "read-key", "temperature", 24, 28)  # 6 h at 15-min steps, 28 days
    llm.assert_not_called()
    steps = np.arange(24.0)
    assert result["forecast_values"] == (steps + 40).tolist()  # q0.5 is the 5th of 9 levels
    assert result["forecast_lower"] == steps.tolist() and result["forecast_upper"] == (steps + 80).tolist()
    assert list(result["forecast_quantiles"]) == ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    assert result["forecast_quantiles"]["0.3"] == (steps + 20).tolist()
    assert result["forecast_points"] == 24 and result["forecast_timestamps"][0] == "2026-02-10T10:00:00"
    assert result["input_points"] == 2688 and result["model_info"] == {"recipe": "plain"}


@pytest.mark.parametrize("provider, requested, used", [
    ("groq", None, 7), ("groq", 14, 14), ("chronos-2", None, 28), ("chronos-2", 3, 3)])
def test_history_days_defaults_per_provider(provider, requested, used):
    with patch.object(forecast_jobs, "load_sensor_series_from_api", autospec=True, return_value=_series()) as load, \
         patch.object(forecast_jobs, "nvp_llms_forecast", autospec=True, return_value=np.zeros(24)), \
         patch.object(forecast_jobs, "forecast_chronos2", autospec=True, return_value=_chronos2_result(24)) as tsfm:
        forecast_jobs.forecast_single_column(
            sensor_id="sensor-uuid", api_key=None, column_name="temperature", horizon_hours=6,
            history_days=requested, provider=provider)

    if provider == "chronos-2":
        assert tsfm.call_args.args[4] == used
    else:
        assert load.call_args.kwargs["window_hours"] == used * 24


def test_llm_providers_refuse_more_than_14_days():
    with patch.object(forecast_jobs, "load_sensor_series_from_api", autospec=True, return_value=_series()) as load:
        with pytest.raises(ValueError, match="too long for LLM providers"):
            forecast_jobs.forecast_single_column(
                sensor_id="sensor-uuid", api_key=None, column_name="temperature", horizon_hours=6,
                history_days=15, provider="groq")
    load.assert_not_called()
