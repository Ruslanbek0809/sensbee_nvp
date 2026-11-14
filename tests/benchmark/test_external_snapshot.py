# Tests for scripts/snapshot_external.py (step 6a.12, X2) on hand-written DWD product files (no network; the
# download function is patched with autospec):
# - the product parser: -999 becomes missing, MESS_DATUM is naive UTC, malformed or pre-2000 stamps are refused
# - historical and recent overlap: the historical (quality-checked) value wins
# - a full snapshot from fake zip files: [since, until) kept, licence and attribution in the manifest, the source
#   files' hashes recorded, and every raw file's hash re-checked on load
# - a rebuild from a snapshot's kept source files (--from-source): no download, the same raw files, and a changed
#   source file refused; the data-quality report

import hashlib
import io
import json
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from benchmark.snapshot import MANIFEST_NAME, load_snapshot
from benchmark.tasks import EXTERNAL_TASKS
from scripts import snapshot_external

HEADER = "STATIONS_ID;MESS_DATUM;  QN;PP_10;TT_10;TM5_10;RF_10;TD_10;eor"


# A DWD product file: the header and one row per (MESS_DATUM, QN, TT_10), padded like the published files.
def _product(rows: list[tuple[str, int, float]], station: str = "3204") -> str:
    lines = [HEADER] + [f"{station:>11};{stamp};{qn:>5};  950.1;{tt:>6};  14.0;  80.0;  11.8;eor"
                        for stamp, qn, tt in rows]
    return "\r\n".join(lines) + "\r\n"


# A zip archive as DWD ships it: the product file next to a metadata file.
def _zip(product: str, name: str = "produkt_zehn_min_tu_20250701_20261113_03204.txt") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Metadaten_Geographie_03204.txt", "Stations_id;Stationshoehe\r\n3204;429\r\n")
        archive.writestr(name, product.encode("latin-1"))
    return buffer.getvalue()


def test_the_parser_reads_utc_stamps_and_makes_minus_999_missing():
    frame = snapshot_external.parse_product(_product(
        [("202608061200", 3, 15.3), ("202608061210", 3, -999), ("202608061220", 1, 15.6)]))
    assert list(frame.columns) == ["created_at", "temperature", "qn"]
    assert frame["created_at"].dt.tz is None
    assert list(frame["created_at"]) == [pd.Timestamp("2026-08-06 12:00"), pd.Timestamp("2026-08-06 12:10"),
                                         pd.Timestamp("2026-08-06 12:20")]
    assert frame["temperature"].iloc[0] == 15.3 and np.isnan(frame["temperature"].iloc[1])
    assert frame["temperature"].iloc[2] == 15.6
    assert list(frame["qn"]) == [3, 3, 1]


@pytest.mark.parametrize("stamp, match", [("2026080612", "MESS_DATUM"), ("199912312350", "BEFORE 2000")])
def test_the_parser_refuses_stamps_without_minutes_and_before_2000(stamp, match):
    with pytest.raises(ValueError, match=match):
        snapshot_external.parse_product(_product([(stamp, 3, 1.0), ("202608061210", 3, 2.0)]))


def test_the_parser_refuses_a_duplicate_stamp():
    with pytest.raises(ValueError, match="DUPLICATE"):
        snapshot_external.parse_product(_product([("202608061200", 3, 1.0), ("202608061200", 3, 2.0)]))


def test_in_the_overlap_the_historical_value_wins():
    historical = snapshot_external.parse_product(_product(
        [("202512312330", 3, 1.0), ("202512312340", 3, 2.0), ("202512312350", 3, 3.0)]))
    recent = snapshot_external.parse_product(_product(
        [("202512312340", 1, 20.0), ("202512312350", 1, 30.0), ("202601010000", 1, 40.0)]))
    merged, overlap = snapshot_external.merge_sources(historical, recent)
    assert overlap == 2
    assert list(merged["temperature"]) == [1.0, 2.0, 3.0, 40.0]
    assert list(merged["qn"]) == [3, 3, 3, 1]
    assert merged["created_at"].is_monotonic_increasing and merged["created_at"].is_unique


def test_the_product_file_is_the_one_produkt_member_of_the_zip():
    assert snapshot_external.product_text(_zip(_product([("202608061200", 3, 1.0)]))).startswith("STATIONS_ID")
    with pytest.raises(ValueError, match="PRODUCT FILE"):
        snapshot_external.product_text(_zip("x", name="readme.txt"))


# Fake downloads: per station a historical zip (one row before since, one inside, one in the overlap) and a recent
# zip (the overlap with another value, one row inside, one at the cutoff), plus the station list.
def _fake_sources() -> dict[str, bytes]:
    files = {snapshot_external.DWD_URL + "/" + snapshot_external.STATION_LIST: "Stations_id von_datum\r\n".encode()}
    for i, station_id in enumerate(snapshot_external.STATIONS.values()):
        hist = _product([("202412312350", 3, 0.5), ("202512312340", 3, 1.0 + i), ("202512312350", 3, 2.0 + i)])
        recent = _product([("202512312350", 1, 99.0), ("202601010000", 1, 3.0 + i), ("202601020000", 1, 9.0)])
        files[snapshot_external.source_url("historical", station_id)] = _zip(hist)
        files[snapshot_external.source_url("recent", station_id)] = _zip(recent)
    return files


def test_a_snapshot_keeps_since_to_until_records_its_sources_and_is_hash_checked_on_load(tmp_path):
    files = _fake_sources()

    def fake_fetch(client, url):
        return files[url], "Sat, 14 Nov 2026 01:15:03 GMT"

    with patch.object(snapshot_external, "fetch", autospec=True, side_effect=fake_fetch):
        out_dir = snapshot_external.take_snapshot(tmp_path, since=pd.Timestamp("2025-01-01"),
                                                  until=pd.Timestamp("2026-01-02"))
        with pytest.raises(FileExistsError):
            snapshot_external.take_snapshot(tmp_path, since=pd.Timestamp("2025-01-01"),
                                            until=pd.Timestamp("2026-01-02"))

    assert out_dir.name == "external-2026-01-02"
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text())
    assert manifest["until"] == "2026-01-02T00:00:00" and manifest["since"] == "2025-01-01T00:00:00"
    assert manifest["source"]["licence"] == "CC BY 4.0"
    assert manifest["source"]["attribution"] == "Source: Deutscher Wetterdienst"
    assert manifest["source"]["attribution_derived"] == (
        "Based on data from Deutscher Wetterdienst, averaged over individual values")
    assert list(manifest["sensors"]) == list(EXTERNAL_TASKS) == list(snapshot_external.STATIONS)
    for entry in manifest["sensors"].values():
        assert entry["columns"] == ["temperature", "qn"] and entry["target_column"] == "temperature"
        assert entry["overlap_rows"] == 1 and entry["qn_counts"] == {"1": 1, "3": 2}
        for source in entry["sources"]:
            data = files[source["url"]]
            assert source["sha256"] == hashlib.sha256(data).hexdigest() and source["bytes"] == len(data)
            assert (out_dir / source["file"]).read_bytes() == data  # the downloaded file is kept

    frames = load_snapshot(out_dir)
    first = frames["DWD_MARTINRODA"]
    assert list(first["created_at"]) == [pd.Timestamp("2025-12-31 23:40"), pd.Timestamp("2025-12-31 23:50"),
                                         pd.Timestamp("2026-01-01 00:00")]
    assert list(first["temperature"]) == [1.0, 2.0, 3.0]  # historical wins at 23:50; 2024 and the cutoff row dropped

    raw = out_dir / manifest["sensors"]["DWD_SCHMUECKE"]["file"]
    raw.write_bytes(raw.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 MISMATCH"):
        load_snapshot(out_dir)


# Builds a snapshot from the fake downloads into root and returns its folder.
def _snapshot(root, files):
    def fake_fetch(client, url):
        return files[url], "Sat, 14 Nov 2026 01:15:03 GMT"

    with patch.object(snapshot_external, "fetch", autospec=True, side_effect=fake_fetch):
        return snapshot_external.take_snapshot(root, since=pd.Timestamp("2025-01-01"), until=pd.Timestamp("2026-01-02"))


def test_a_rebuild_from_the_kept_sources_gives_the_same_raw_files_without_downloading(tmp_path):
    original = _snapshot(tmp_path / "a", _fake_sources())
    with patch.object(snapshot_external, "fetch", autospec=True, side_effect=AssertionError("no download")):
        rebuilt = snapshot_external.take_snapshot(tmp_path / "b", since=pd.Timestamp("2025-01-01"),
                                                  until=pd.Timestamp("2026-01-02"), source_dir=original)
    first = json.loads((original / MANIFEST_NAME).read_text())
    again = json.loads((rebuilt / MANIFEST_NAME).read_text())
    assert again["rebuilt_from"] == {"snapshot_id": "external-2026-01-02",
                                     "manifest_sha256": hashlib.sha256((original / MANIFEST_NAME).read_bytes())
                                     .hexdigest()}
    assert again["source"]["retrieved_utc"] == first["source"]["retrieved_utc"]
    assert again["source"]["station_list"] == first["source"]["station_list"]
    for name, entry in first["sensors"].items():
        assert again["sensors"][name]["sha256"] == entry["sha256"]
        assert again["sensors"][name]["sources"] == entry["sources"]
    assert len(load_snapshot(rebuilt)) == 3


def test_a_rebuild_refuses_a_changed_source_file(tmp_path):
    original = _snapshot(tmp_path / "a", _fake_sources())
    kept = original / json.loads((original / MANIFEST_NAME).read_text())["sensors"]["DWD_NEUHAUS"]["sources"][1]["file"]
    kept.write_bytes(kept.read_bytes() + b"x")
    with pytest.raises(ValueError, match="SHA-256 MISMATCH"):
        snapshot_external.take_snapshot(tmp_path / "b", since=pd.Timestamp("2025-01-01"),
                                        until=pd.Timestamp("2026-01-02"), source_dir=original)
    assert not (tmp_path / "b").exists()


def test_the_quality_report_lists_every_station_and_the_attribution(tmp_path):
    out_dir = _snapshot(tmp_path, _fake_sources())
    report = (out_dir / "quality_report.md").read_text()
    assert "Source: Deutscher Wetterdienst" in report and "CC BY 4.0" in report
    assert "Based on data from Deutscher Wetterdienst, averaged over individual values" in report
    for name, station_id in snapshot_external.STATIONS.items():
        assert f"| {name} | {station_id} | 3 |" in report
