#!/usr/bin/env python3
# Downloads a frozen snapshot of three DWD weather stations near Ilmenau (step 6a.12, X2): the public replication set
# of the "external" task set (benchmark/tasks.py:EXTERNAL_TASKS).
#
# Source: DWD Climate Data Center, "10-minute station observations of air temperature for Germany", v24.03
# (description and terms of use read 2026-11-14):
# - per station a historical/ zip (quality control completed, versioned yearly) and a recent/ zip (the last 500 days
#   up to yesterday, quality control not completed, values may still change);
# - the product file inside each zip is ";"-separated with MESS_DATUM (YYYYMMDDHHMM), QN (quality level) and TT_10
#   (air temperature 2 m above ground, an instantaneous value); -999 marks a missing value;
# - stamps are UTC from 2000 on (MEZ before), so rows before 2000 are refused;
# - licence CC BY 4.0; attribution "Source: Deutscher Wetterdienst" next to the data, and for derived values (the
#   15-min averages and everything computed from them) e.g. "Based on data from Deutscher Wetterdienst, averaged over
#   individual values" (dwd.de, "Rules for acknowledging the DWD as source").
# Where historical and recent overlap, the historical (quality-checked) row is kept (owner decision 2026-11-14).
# Every row with a quality level is kept; the manifest counts the levels per station.
#
# The snapshot has the layout and manifest schema of scripts/snapshot_sensbee.py, so benchmark.snapshot.load_snapshot
# re-checks every raw file's SHA-256. The downloaded zips and the station list are kept in source/ with their hashes,
# and quality_report.md summarises the stations.
# --from-source <snapshot dir> rebuilds a snapshot from that snapshot's kept source files instead of downloading
# (each file's SHA-256 is checked against its manifest first). Same code and files give the same raw files, so this
# checks a snapshot against the committed code; the recent/ files on the server change daily, a new download may not.
#
# Usage (from sensbee_nvp/):
#   venv/bin/python scripts/snapshot_external.py --out ../../benchmark_data/snapshots
#   venv/bin/python scripts/snapshot_external.py --out <spike dir> \
#       --from-source ../../benchmark_data/snapshots/external-2026-10-01

import argparse
import hashlib
import io
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import httpx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.snapshot import (  # noqa: E402
    MANIFEST_NAME, TIME_COL, bucket_15min, git_state, quality_stats, sha256_file, write_raw_csv,
)

DWD_URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/air_temperature"
STATIONS: dict[str, str] = {"DWD_MARTINRODA": "03204", "DWD_SCHMUECKE": "04501", "DWD_NEUHAUS": "03513"}
FILES = {"historical": "historical/10minutenwerte_TU_{id}_20200101_20251231_hist.zip",
         "recent": "recent/10minutenwerte_TU_{id}_akt.zip"}
STATION_LIST = "recent/zehn_min_tu_Beschreibung_Stationen.txt"
SINCE, UNTIL = "2025-01-01", "2026-10-01"   # the SensBee weather stations' span and cutoff (owner decision 2026-11-14)
MISSING = -999
TIMEOUT_S = 60
SOURCE = {
    "provider": "Deutscher Wetterdienst (DWD), Climate Data Center",
    "dataset": "10-minute station observations of air temperature for Germany, Version v24.03",
    "variable": "TT_10, air temperature 2 m above ground, instantaneous, degrees C",
    "licence": "CC BY 4.0",
    "attribution": "Source: Deutscher Wetterdienst",
    "attribution_derived": "Based on data from Deutscher Wetterdienst, averaged over individual values",
    "terms_url": "https://opendata.dwd.de/climate_environment/CDC/Terms_of_use.pdf",
    "attribution_url": "https://www.dwd.de/EN/service/legal_notice/templates_dwd_as_source.html",
    "description_url": f"{DWD_URL}/DESCRIPTION_obsgermany_climate_10min_air_temperature_en.pdf",
    "overlap_rule": "historical (quality-checked) rows win over recent rows at the same MESS_DATUM",
}


# The URL of a station's historical or recent zip.
def source_url(folder: str, station_id: str) -> str:
    return f"{DWD_URL}/{FILES[folder].format(id=station_id)}"


# Downloads one file. Returns its bytes and the server's Last-Modified header.
def fetch(client: httpx.Client, url: str) -> tuple[bytes, Optional[str]]:
    response = client.get(url)
    response.raise_for_status()
    return response.content, response.headers.get("last-modified")


# The text of the one product file (produkt_*.txt) in a DWD zip. Raises when there is none or more than one.
def product_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [name for name in archive.namelist() if name.startswith("produkt_") and name.endswith(".txt")]
        if len(names) != 1:
            raise ValueError(f"EXPECTED ONE PRODUCT FILE produkt_*.txt, FOUND {names} among {archive.namelist()}")
        return archive.read(names[0]).decode("latin-1")


# Parses a DWD product file into raw rows: created_at (naive UTC), temperature (TT_10, -999 → NaN) and qn.
# Raises on a MESS_DATUM that isn't 12 digits, on rows before 2000 (MEZ stamps) and on duplicate stamps.
def parse_product(text: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    frame.columns = [column.strip() for column in frame.columns]
    stamps = frame["MESS_DATUM"].str.strip()
    malformed = stamps[~stamps.str.fullmatch(r"\d{12}")]
    if len(malformed):
        raise ValueError(f"MESS_DATUM IS NOT YYYYMMDDHHMM (12 digits): {list(malformed[:3])}")
    created = pd.to_datetime(stamps, format="%Y%m%d%H%M")
    if (created < pd.Timestamp("2000-01-01")).any():
        raise ValueError(f"ROWS BEFORE 2000 (MEZ STAMPS, NOT UTC), first {created.min()}")
    if created.duplicated().any():
        raise ValueError(f"DUPLICATE MESS_DATUM {list(created[created.duplicated()][:3])}")
    temperature = pd.to_numeric(frame["TT_10"].str.strip())
    return pd.DataFrame({TIME_COL: created, "temperature": temperature.where(temperature != MISSING),
                         "qn": pd.to_numeric(frame["QN"].str.strip()).astype(int)})


# Joins a station's historical and recent rows. At a stamp both have, the historical row is kept.
# Returns the rows in time order and the number of recent rows dropped for the overlap.
def merge_sources(historical: pd.DataFrame, recent: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    later = recent[~recent[TIME_COL].isin(historical[TIME_COL])]
    merged = pd.concat([historical, later]).sort_values(TIME_COL).reset_index(drop=True)
    return merged, len(recent) - len(later)


# A downloaded file's record for the manifest.
def _record(url: str, data: bytes, modified: Optional[str], file: str) -> dict:
    return {"url": url, "file": file, "bytes": len(data), "last_modified": modified,
            "sha256": hashlib.sha256(data).hexdigest()}


# A reader of the source files kept in an earlier snapshot, by URL: it returns their bytes and recorded
# Last-Modified like fetch, after checking each file's SHA-256 against that snapshot's manifest (ValueError on a
# mismatch).
def kept_sources(source_dir: Path) -> Callable[[httpx.Client, str], tuple[bytes, Optional[str]]]:
    manifest = json.loads((source_dir / MANIFEST_NAME).read_text())
    records = [manifest["source"]["station_list"],
               *[source for entry in manifest["sensors"].values() for source in entry["sources"]]]
    by_url = {record["url"]: record for record in records}

    def read(client: httpx.Client, url: str) -> tuple[bytes, Optional[str]]:
        record = by_url[url]
        digest = sha256_file(source_dir / record["file"])
        if digest != record["sha256"]:
            raise ValueError(f"SHA-256 MISMATCH for {record['file']}: manifest {record['sha256']}, file {digest}")
        return (source_dir / record["file"]).read_bytes(), record["last_modified"]
    return read


# Formats the manifest as a short Markdown data-quality report, with the licence and attribution lines.
def quality_report(manifest: dict) -> str:
    code, source = manifest["code"], manifest["source"]
    lines = [
        f"# External snapshot {manifest['snapshot_id']}",
        "",
        f"Rows in [{manifest['since']}, {manifest['until']}) naive UTC · retrieved {source['retrieved_utc']} "
        f"· written {manifest['created_at_utc']} · code {code['commit']}{' (dirty)' if code['dirty'] else ''}"
        + (f" · rebuilt from {manifest['rebuilt_from']['snapshot_id']}" if "rebuilt_from" in manifest else ""),
        "",
        f"{source['dataset']}; {source['variable']}. Licence {source['licence']}. Raw values: "
        f"\"{source['attribution']}\". Values derived from the 15-min buckets: \"{source['attribution_derived']}\". "
        f"Overlap: {source['overlap_rule']}.",
        "",
        "| Sensor | DWD id | Rows | First row | Last row | Missing TT_10 | Overlap rows | QN counts "
        "| Buckets present/expected | Coverage | Longest missing run (h) | ACF 1 d |",
        "|" + "---|" * 12,
    ]
    for name, e in manifest["sensors"].items():
        q = e["quality"]
        qn = ", ".join(f"{level}: {n}" for level, n in e["qn_counts"].items())
        lines.append(f"| {name} | {e['station_id']} | {e['rows']} | {e['first_row']} | {e['last_row']} "
                     f"| {e['missing_temperature_rows']} | {e['overlap_rows']} | {qn} "
                     f"| {q.get('buckets_present', 0)}/{q.get('buckets_expected', 0)} | {q.get('coverage', '–')} "
                     f"| {q.get('longest_missing_run_hours', '–')} | {q.get('autocorr_1_day', '–')} |")
    return "\n".join(lines) + "\n"


# Downloads, converts and writes one snapshot of the rows in [since, until). Nothing is written until every file
# has been read and parsed. With source_dir, the files come from that snapshot's source/ (kept_sources), not the
# server, and the manifest records which snapshot it was rebuilt from. Returns the new snapshot folder
# (external-<until>).
def take_snapshot(out_root: Path, since: pd.Timestamp, until: pd.Timestamp,
                  source_dir: Optional[Path] = None) -> Path:
    out_dir = Path(out_root) / f"external-{until:%Y-%m-%d}"
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} EXISTS; snapshots are never overwritten")
    get = fetch if source_dir is None else kept_sources(Path(source_dir))
    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    rebuilt_from = {}
    if source_dir is not None:
        earlier = json.loads((Path(source_dir) / MANIFEST_NAME).read_text())
        retrieved = earlier["source"]["retrieved_utc"]
        rebuilt_from = {"rebuilt_from": {"snapshot_id": earlier["snapshot_id"],
                                         "manifest_sha256": sha256_file(Path(source_dir) / MANIFEST_NAME)}}
    files: dict[str, bytes] = {}
    raws: dict[str, pd.DataFrame] = {}
    entries: dict[str, dict] = {}
    with httpx.Client(timeout=TIMEOUT_S, follow_redirects=True) as client:
        list_url = f"{DWD_URL}/{STATION_LIST}"
        data, modified = get(client, list_url)
        station_list = _record(list_url, data, modified, f"source/{Path(STATION_LIST).name}")
        files[station_list["file"]] = data
        for name, station_id in STATIONS.items():
            parsed, sources = {}, []
            for folder in FILES:
                url = source_url(folder, station_id)
                data, modified = get(client, url)
                record = _record(url, data, modified, f"source/{url.rsplit('/', 1)[1]}")
                files[record["file"]] = data
                parsed[folder] = parse_product(product_text(data))
                sources.append({**record, "rows": len(parsed[folder]),
                                "first": parsed[folder][TIME_COL].min().isoformat(),
                                "last": parsed[folder][TIME_COL].max().isoformat()})
            merged, overlap = merge_sources(parsed["historical"], parsed["recent"])
            raw = merged[(merged[TIME_COL] >= since) & (merged[TIME_COL] < until)].reset_index(drop=True)
            raws[name] = raw
            entries[name] = {
                "file": f"raw/{name}.csv.gz",
                "rows": len(raw),
                "columns": [column for column in raw.columns if column != TIME_COL],
                "first_row": raw[TIME_COL].min().isoformat() if not raw.empty else None,
                "last_row": raw[TIME_COL].max().isoformat() if not raw.empty else None,
                "target_column": "temperature",
                "target_aggregation": "AVG",
                "quality": quality_stats(bucket_15min(raw, "temperature", "AVG")),
                "station_id": station_id,
                "sources": sources,
                "overlap_rows": overlap,
                "missing_temperature_rows": int(raw["temperature"].isna().sum()),
                "qn_counts": {str(level): int(n) for level, n in sorted(raw["qn"].value_counts().items())},
            }
            print(f"FETCHED {name} ({station_id}): {len(raw)} rows, overlap {overlap}", file=sys.stderr)

    out_dir.mkdir(parents=True)
    for file, data in files.items():
        (out_dir / file).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / file).write_bytes(data)
    for name, raw in raws.items():
        entries[name]["sha256"] = write_raw_csv(raw, out_dir / entries[name]["file"])
    manifest = {
        "snapshot_id": out_dir.name,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "since": since.isoformat(),
        "until": until.isoformat(),
        "timestamps": "naive UTC",
        "base_url": DWD_URL,
        "source": {**SOURCE, "retrieved_utc": retrieved, "station_list": station_list},
        "code": git_state(),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "httpx": httpx.__version__},
        "sensors": entries,
        **rebuilt_from,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    (out_dir / "quality_report.md").write_text(quality_report(manifest))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen snapshot of three DWD stations near Ilmenau (public data)")
    parser.add_argument("--out", type=Path, required=True, help="parent folder; a new external-<until> folder is created")
    parser.add_argument("--since", default=SINCE, help=f"first day kept, naive UTC (default {SINCE})")
    parser.add_argument("--until", default=UNTIL, help=f"exclusive cutoff day, naive UTC (default {UNTIL})")
    parser.add_argument("--from-source", type=Path,
                        help="rebuild from this snapshot's kept source files instead of downloading")
    args = parser.parse_args()
    out_dir = take_snapshot(args.out, pd.Timestamp(args.since).floor("D"), pd.Timestamp(args.until).floor("D"),
                            args.from_source)
    print(f"SNAPSHOT WRITTEN to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
