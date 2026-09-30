# Tests for forecast_from_api(): it must request a history window the SensBee client understands.
# The loader is imported inside the function, so it is patched where it's defined, with autospec.

from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models import nvp_llms


# Builds a 15-min series like the SensBee client returns.
def _series(points: int = 96) -> pd.Series:
    index = pd.date_range("2026-02-10 00:00", periods=points, freq="15min")
    return pd.Series(np.linspace(-2.0, 3.0, points), index=index)


def test_history_days_is_passed_as_window_hours():
    with patch("src.data_access.sensbee_client.load_sensor_series_from_api", autospec=True, return_value=_series()) as load, \
         patch.object(nvp_llms, "nvp_llms_forecast", autospec=True, return_value=np.zeros(96)):
        nvp_llms.forecast_from_api(sensor_id="sensor-uuid", column_name="temperature", horizon_hours=24, history_days=3)

    kwargs = load.call_args.kwargs
    assert kwargs["window_hours"] == 3 * 24
    assert "limit" not in kwargs
