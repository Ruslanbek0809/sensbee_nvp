# Tests for load_sensor_series_from_json() through the real loader on a small JSON file in tmp_path: event-based
# columns must be clipped at 0 like the service path (src/data_access/sensbee_client.py), so fixtures and live data
# give the same series.

import json

from src.data_access.data_loader import load_sensor_series_from_json


def test_event_based_negative_counts_are_clipped_to_zero(tmp_path):
    path = tmp_path / "visitors.json"
    path.write_text(json.dumps([
        {"created_at": "2026-02-10T10:05:00.000", "visitors_total": 5},
        {"created_at": "2026-02-10T10:20:00.000", "visitors_total": -3},  # counter anomaly
        {"created_at": "2026-02-10T10:35:00.000", "visitors_total": 4},
    ]))

    series = load_sensor_series_from_json(str(path), "visitors_total")

    assert series.tolist() == [5.0, 0.0, 4.0]


def test_regular_columns_keep_negative_values(tmp_path):
    path = tmp_path / "weather.json"
    path.write_text(json.dumps([
        {"created_at": "2026-02-10T10:00:00.000", "temperature": -3.5},
        {"created_at": "2026-02-10T10:15:00.000", "temperature": -2.0},
    ]))

    assert load_sensor_series_from_json(str(path), "temperature").tolist() == [-3.5, -2.0]
