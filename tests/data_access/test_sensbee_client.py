# Tests for load_sensor_series_from_api(): the 15-min grid must match SensBee's bucket labels and the default window must end at UTC now.
# Only the HTTP layer is patched (with autospec), so the real resampling and gap filling run on the fake server's data.

import logging
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from src.data_access.sensbee_client import SensbeeClient, load_sensor_series_from_api


# Fake SensBee /data/load with time_grouping: one record per bucket, bucket = floor(epoch / interval) * interval
# (sensor_mgmt/src/database/data_db.rs). Values are 1, 2, 3, ... in time order.
def _fake_get_sensor_data(self, sensor_id, api_key=None, from_time=None, to_time=None, limit=None,
                          ordering="ASC", cols=None, time_grouping_seconds=None) -> list[dict]:
    column = cols[0].split(".")[0]
    freq = f"{time_grouping_seconds}s"
    buckets = pd.date_range(pd.Timestamp(from_time).floor(freq), pd.Timestamp(to_time).floor(freq), freq=freq)
    return [
        {"grouped_time": ts.strftime("%Y-%m-%dT%H:%M:%S"), column: float(i + 1)}
        for i, ts in enumerate(buckets)
    ]


# Makes datetime.now() return Berlin local time (UTC+1/+2) for one test, whatever the machine's timezone.
@pytest.fixture
def berlin_local_time(monkeypatch):
    monkeypatch.setenv("TZ", "Europe/Berlin")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.parametrize("column_name", ["temperature", "visitors_total"])
def test_grid_matches_server_buckets_when_request_is_off_the_quarter_hour(column_name):
    with patch.object(SensbeeClient, "get_sensor_data", autospec=True, side_effect=_fake_get_sensor_data):
        series = load_sensor_series_from_api(
            sensor_id="sensor-uuid",
            api_key="read-key",
            column_name=column_name,
            from_time=datetime(2026, 2, 10, 10, 37, 12),
            to_time=datetime(2026, 2, 10, 12, 37, 12),
        )

    expected_index = pd.date_range("2026-02-10 10:30", "2026-02-10 12:30", freq="15min")
    assert list(series.index) == list(expected_index)
    assert series.tolist() == [float(i) for i in range(1, 10)]


def test_default_window_ends_at_utc_now(berlin_local_time):
    with patch.object(SensbeeClient, "get_sensor_data", autospec=True, side_effect=_fake_get_sensor_data) as get:
        series = load_sensor_series_from_api(
            sensor_id="sensor-uuid",
            api_key="read-key",
            column_name="temperature",
            window_hours=2,
        )
    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)

    kwargs = get.call_args.kwargs
    assert abs((kwargs["to_time"] - utc_now).total_seconds()) < 60
    assert kwargs["from_time"] == kwargs["to_time"] - timedelta(hours=2)
    assert series.index[-1] <= utc_now


def test_read_key_is_not_logged(caplog):
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = []
    caplog.set_level(logging.DEBUG, logger="src.data_access.sensbee_client")
    with patch.object(httpx.Client, "get", autospec=True, return_value=response) as get:
        SensbeeClient().get_sensor_data(sensor_id="sensor-uuid", api_key="secret-read-key", limit=5)

    assert get.call_args.kwargs["params"]["key"] == "secret-read-key"
    assert "secret-read-key" not in caplog.text
