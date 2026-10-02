# Tests for the frozen snapshot: 15-min bucketing like SensBee, month paging, hash-checked CSV.gz round trip, and a
# full snapshot run against a fake server (no network; SensbeeClient.get_sensor_data is patched with autospec).

import gzip
import json
import math
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from benchmark import snapshot
from benchmark.snapshot import (
    MANIFEST_NAME, bucket_15min, load_snapshot, quality_stats, read_raw_csv, write_raw_csv,
)
from scripts import snapshot_sensbee
from src.data_access.sensbee_client import SensbeeClient

EPOCH = datetime(1970, 1, 1)
FAKE_KEY = "fake-read-key-123"
FAKE_UUID = "00000000-fake-uuid-0000-000000000000"


# Fake SensBee /data/load over a fixed list of raw rows: inclusive from/to, limit, and with time_grouping one record
# per bucket floor(epoch / interval) * interval with AVG or MAX (sensor_mgmt/src/database/data_db.rs). Written with
# plain epoch arithmetic, independent of the pandas code under test.
def _fake_server(raw_rows: list[dict]):
    def get_sensor_data(self, sensor_id, api_key=None, from_time=None, to_time=None, limit=None,
                        ordering="ASC", cols=None, time_grouping_seconds=None) -> list[dict]:
        rows = [r for r in raw_rows
                if (from_time is None or r["created_at"] >= from_time) and (to_time is None or r["created_at"] <= to_time)]
        if time_grouping_seconds is None:
            out = [{**r, "created_at": r["created_at"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]} for r in rows]
            return out[:limit]
        groups: dict[datetime, list[dict]] = {}
        for r in rows:
            seconds = (r["created_at"] - EPOCH).total_seconds()
            label = EPOCH + timedelta(seconds=math.floor(seconds / time_grouping_seconds) * time_grouping_seconds)
            groups.setdefault(label, []).append(r)
        out = []
        for label in sorted(groups):
            record = {"grouped_time": label.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}
            for spec in cols:
                column, aggregation = spec.split(".")
                values = [r[column] for r in groups[label] if r.get(column) is not None]
                record[column] = (None if not values else max(values) if aggregation == "MAX"
                                  else sum(values) / len(values))
            out.append(record)
        return out[:limit]
    return get_sensor_data


# Raw visitor-like rows around a month boundary, one of them exactly at 2026-02-01 00:00:00.
def _visitor_rows() -> list[dict]:
    times = ["2026-01-31T23:50:12.345", "2026-01-31T23:52:00.000", "2026-02-01T00:00:00.000",
             "2026-02-01T00:07:30.500", "2026-02-01T00:21:00.000"]
    return [{"created_at": datetime.fromisoformat(t), "visitor_change": 1, "visitor_type": "in", "visitors_total": i + 3}
            for i, t in enumerate(times)]


# Raw weather-like rows every 7 minutes, so buckets hold 2-3 rows and AVG is a real mean.
def _weather_rows() -> list[dict]:
    start = datetime(2026, 1, 31, 22, 0)
    return [{"created_at": start + timedelta(minutes=7 * i), "temperature": round(-1.3 + 0.17 * i, 2),
             "humidity": 80.0 + (i % 5)} for i in range(60)]


def test_bucket_15min_matches_the_server_rule():
    raw = pd.DataFrame({
        "created_at": pd.to_datetime(["2026-02-10 10:07:00", "2026-02-10 10:14:59.999", "2026-02-10 10:15:00",
                                      "2026-02-10 10:20:00", "2026-02-10 10:44:00"], format="ISO8601"),
        "value": [1.0, 3.0, 5.0, None, 7.0],
    })
    labels = list(pd.to_datetime(["2026-02-10 10:00", "2026-02-10 10:15", "2026-02-10 10:30"]))

    avg = bucket_15min(raw, "value", "AVG")
    assert list(avg.index) == labels
    assert avg.tolist() == [2.0, 5.0, 7.0]  # (1 + 3) / 2; the null at 10:20 is skipped like SQL AVG
    assert bucket_15min(raw, "value", "MAX").tolist() == [3.0, 5.0, 7.0]


def test_quality_stats_counts_missing_buckets_and_zeros():
    index = pd.to_datetime(["2026-02-10 00:00", "2026-02-10 00:15", "2026-02-10 01:30"])
    stats = quality_stats(pd.Series([0.0, 4.0, 0.0], index=index))

    assert stats["buckets_expected"] == 7  # 00:00 ... 01:30
    assert stats["buckets_present"] == 3
    assert stats["longest_missing_run_hours"] == 1.0  # 00:30, 00:45, 01:00, 01:15
    assert stats["zero_share"] == round(2 / 3, 4)


def test_fetch_raw_keeps_a_month_boundary_row_once():
    rows = _visitor_rows()
    with patch.object(SensbeeClient, "get_sensor_data", autospec=True, side_effect=_fake_server(rows)) as get:
        raw = snapshot_sensbee.fetch_raw(
            SensbeeClient(), FAKE_UUID, FAKE_KEY, pd.Timestamp("2026-01-31"), pd.Timestamp("2026-02-02"))

    assert get.call_count == 2  # January window and February window
    assert len(raw) == len(rows)
    assert (raw["created_at"] == pd.Timestamp("2026-02-01 00:00:00")).sum() == 1


def test_fetch_raw_raises_when_a_window_fills_the_limit(monkeypatch):
    monkeypatch.setattr(snapshot_sensbee, "WINDOW_LIMIT", 2)
    with patch.object(SensbeeClient, "get_sensor_data", autospec=True, side_effect=_fake_server(_visitor_rows())):
        with pytest.raises(RuntimeError, match="limit"):
            snapshot_sensbee.fetch_raw(
                SensbeeClient(), FAKE_UUID, FAKE_KEY, pd.Timestamp("2026-01-31"), pd.Timestamp("2026-02-02"))


def test_cross_check_reports_changed_and_one_sided_buckets():
    rows = _weather_rows()
    honest = _fake_server(rows)

    # Per month window, the server changes one bucket's temperature and drops another bucket entirely
    def tampered(self, sensor_id, **kwargs):
        out = honest(self, sensor_id, **kwargs)
        if kwargs.get("time_grouping_seconds") and len(out) > 1:
            out = [dict(r) for r in out]
            out[0]["temperature"] += 0.5
            del out[1]
        return out

    since, until = pd.Timestamp("2026-01-31"), pd.Timestamp("2026-02-02")
    with patch.object(SensbeeClient, "get_sensor_data", autospec=True, side_effect=tampered) as get:
        raw = snapshot_sensbee.fetch_raw(SensbeeClient(), FAKE_UUID, FAKE_KEY, since, until)
        result = snapshot_sensbee.cross_check(SensbeeClient(), FAKE_UUID, FAKE_KEY, raw, since, until)

    assert get.call_count == 4  # 2 raw + 2 grouped month windows
    assert result["temperature"]["mismatches"] == 4  # 2 windows × (1 changed + 1 dropped)
    assert result["humidity"]["mismatches"] == 2  # 2 windows × 1 dropped
    assert result["temperature"]["max_abs_diff"] == pytest.approx(0.5)


def test_raw_csv_round_trip_and_hash_check(tmp_path):
    raw = pd.DataFrame(_weather_rows())
    raw["created_at"] = pd.to_datetime(raw["created_at"])
    raw.loc[3, "temperature"] = None
    path = tmp_path / "raw" / "S.csv.gz"
    digest = write_raw_csv(raw, path)
    assert write_raw_csv(raw, tmp_path / "again.csv.gz") == digest  # same data → same bytes

    (tmp_path / MANIFEST_NAME).write_text(json.dumps(
        {"snapshot_id": "test", "sensors": {"S": {"file": "raw/S.csv.gz", "sha256": digest}}}))
    pd.testing.assert_frame_equal(load_snapshot(tmp_path)["S"], raw)
    pd.testing.assert_frame_equal(read_raw_csv(path), raw)

    data = bytearray(path.read_bytes())
    data[-5] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA-256 MISMATCH"):
        load_snapshot(tmp_path)


def test_snapshot_run_cross_checks_and_writes_no_secrets(tmp_path):
    rows = {"EISHALLE": _visitor_rows(), "MANEBACH_WEATHER_STATION": _weather_rows()}

    def by_sensor(self, sensor_id, **kwargs):
        return _fake_server(rows[sensor_id.split("|")[1]])(self, sensor_id, **kwargs)

    sensors = {name: (f"{FAKE_UUID}|{name}", FAKE_KEY) for name in rows}
    with patch.object(SensbeeClient, "get_sensor_data", autospec=True, side_effect=by_sensor):
        out_dir = snapshot_sensbee.take_snapshot(
            sensors, until=pd.Timestamp("2026-02-02"), out_root=tmp_path, since=pd.Timestamp("2026-01-31"))

        manifest = json.loads((out_dir / MANIFEST_NAME).read_text())
        for name, entry in manifest["sensors"].items():
            assert entry["rows"] == len(rows[name])
            for column, check in entry["cross_check"].items():
                assert check["buckets_compared"] > 0, column
                assert check["mismatches"] == 0, column
        assert set(manifest["sensors"]["EISHALLE"]["cross_check"]) == {"visitor_change", "visitors_total"}
        assert manifest["sensors"]["EISHALLE"]["cross_check"]["visitors_total"]["aggregation"] == "MAX"
        assert len(load_snapshot(out_dir)) == 2

        with pytest.raises(FileExistsError):
            snapshot_sensbee.take_snapshot(
                sensors, until=pd.Timestamp("2026-02-02"), out_root=tmp_path, since=pd.Timestamp("2026-01-31"))

    for path in out_dir.rglob("*"):
        if path.is_file():
            text = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
            assert FAKE_KEY not in text and FAKE_UUID not in text, path.name


def test_git_state_is_unknown_without_a_git_checkout(tmp_path, monkeypatch):
    # A copy without .git (e.g. deployed to the cluster) must not claim a clean tree.
    monkeypatch.setattr(snapshot, "PROJECT_ROOT", tmp_path)
    assert snapshot.git_state() == {"repo": "sensbee_nvp", "commit": None, "dirty": None}
