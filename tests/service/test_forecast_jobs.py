# Tests for forecast_single_column(): the requested history must reach the SensBee client as a time window.
# The SensBee loader is patched with autospec, so a call with parameters it doesn't accept fails like in production.

from unittest.mock import patch

import numpy as np
import pandas as pd

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
