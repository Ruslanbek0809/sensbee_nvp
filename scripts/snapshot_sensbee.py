#!/usr/bin/env python3
# Downloads a frozen snapshot of the SensBee sensors used in the thesis: raw rows (all columns) as CSV.gz, a manifest
# with a SHA-256 hash per file, and a data-quality report. Read-only against the API.
#
# Rows are fetched in calendar-month windows. The server treats from/to as inclusive, so rows at exactly a window's
# end are dropped here and come back with the next window. A cross-check compares our 15-min aggregation of the raw
# rows with SensBee's own time_grouping buckets, for every numeric column and month. Sensors are named by their env
# prefix; READ keys and sensor UUIDs are never written or printed.
#
# Usage (from sensbee_nvp/):
#   venv/bin/python scripts/snapshot_sensbee.py --out ../../benchmark_data/snapshots
#   venv/bin/python scripts/snapshot_sensbee.py --out <dir> --sensors EISHALLE --since 2026-02-01 --until 2026-03-01

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.snapshot import (  # noqa: E402
    BUCKETS_PER_DAY, MANIFEST_NAME, TIME_COL, bucket_15min, quality_stats, write_raw_csv,
)
from src.data_access.sensbee_client import (  # noqa: E402
    COLUMN_AGGREGATIONS, DEFAULT_AGGREGATION, DEFAULT_BASE_URL, SensbeeClient,
)

# Sensors in the snapshot (env prefix → target column for the quality report); roadmap decision D3.
TARGET_COLUMNS: dict[str, str] = {
    "MANEBACH_WEATHER_STATION": "temperature",
    "STUTZERBACH_WEATHER_STATION": "temperature",
    "FRAUENWALD_WEATHER_STATION": "temperature",
    "EISHALLE": "visitors_total",
    "SCHWIMMHALLE": "visitors_total",
    "PARKPLATZ_EAZ": "Fahrzeuge_auf_Parkplatz",
}
WINDOW_LIMIT = 200_000                     # raw rows per month window; the busiest month in the 2026-09-29 audit had 44,617
BUCKET_LIMIT = 31 * BUCKETS_PER_DAY + 10   # 15-min buckets per month window, plus a buffer
MATCH_ATOL = 1e-6                          # Postgres AVG vs pandas mean can differ in the last float digits


# Splits [since, until) at calendar-month starts: [(since, next month start), ..., (last month start, until)].
def month_windows(since: pd.Timestamp, until: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = [t for t in pd.date_range(since, until, freq="MS") if since < t < until]
    edges = [since, *starts, until] if since < until else []
    return list(zip(edges[:-1], edges[1:]))


# Fetches [since, until) month by month and returns one frame. The server's from/to are inclusive, so rows at exactly
# a window's end are dropped here and come back with the next window. A window that fills the limit raises.
def fetch_windows(client: SensbeeClient, sensor_id: str, api_key: Optional[str], since: pd.Timestamp,
                  until: pd.Timestamp, time_col: str, limit: int, **params) -> pd.DataFrame:
    frames = []
    for start, end in month_windows(since, until):
        rows = client.get_sensor_data(sensor_id=sensor_id, api_key=api_key, from_time=start, to_time=end,
                                      limit=limit, ordering="ASC", **params)
        if len(rows) >= limit:
            raise RuntimeError(f"WINDOW {start} → {end} RETURNED {len(rows)} ROWS (= limit), data may be truncated")
        if rows:
            df = pd.DataFrame(rows)
            df[time_col] = pd.to_datetime(df[time_col], format="ISO8601")
            frames.append(df[df[time_col] < end])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({time_col: pd.Series(dtype="datetime64[ns]")})


# Fetches all raw rows (all columns) in [since, until).
def fetch_raw(client: SensbeeClient, sensor_id: str, api_key: Optional[str],
              since: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    return fetch_windows(client, sensor_id, api_key, since, until, TIME_COL, WINDOW_LIMIT)


# Compares our 15-min aggregation of the raw rows with SensBee's own time_grouping buckets for every numeric column.
# Returns per column: buckets compared, mismatches (a bucket present on only one side counts) and the max abs diff.
def cross_check(client: SensbeeClient, sensor_id: str, api_key: Optional[str], raw: pd.DataFrame,
                since: pd.Timestamp, until: pd.Timestamp) -> dict:
    numeric = [c for c in raw.columns if c != TIME_COL and pd.api.types.is_numeric_dtype(raw[c])]
    if not numeric:
        return {}
    aggregations = {c: COLUMN_AGGREGATIONS.get(c, DEFAULT_AGGREGATION) for c in numeric}
    server = fetch_windows(client, sensor_id, api_key, since, until, "grouped_time", BUCKET_LIMIT,
                           cols=[f"{c}.{aggregations[c]}" for c in numeric], time_grouping_seconds=900)
    server = server.reindex(columns=["grouped_time", *numeric]).set_index("grouped_time")
    result = {}
    for c in numeric:
        joined = pd.concat([bucket_15min(raw, c, aggregations[c]).rename("ours"),
                            pd.to_numeric(server[c], errors="coerce").rename("server")], axis=1)
        same = np.isclose(joined["ours"], joined["server"], rtol=0, atol=MATCH_ATOL) | joined.isna().all(axis=1)
        diff = (joined["ours"] - joined["server"]).abs().max()
        result[c] = {"aggregation": aggregations[c], "buckets_compared": len(joined), "mismatches": int((~same).sum()),
                     "max_abs_diff": float(diff) if pd.notna(diff) else 0.0}
    return result


# Returns the sensbee_nvp commit and whether its working tree has uncommitted changes.
def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], capture_output=True, text=True).stdout.strip()
    return {"repo": "sensbee_nvp", "commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


# Formats the manifest as a short Markdown data-quality report.
def quality_report(manifest: dict) -> str:
    code = manifest["code"]
    lines = [
        f"# SensBee snapshot {manifest['snapshot_id']}",
        "",
        f"Cutoff (exclusive, naive UTC): {manifest['until']} · since: {manifest['since'] or 'first row per sensor'} "
        f"· created {manifest['created_at_utc']} · code {code['commit']}{' (dirty)' if code['dirty'] else ''}",
        "",
        "Raw rows as stored. The 15-min buckets are derived with SensBee's rule (floor to the quarter hour; AVG, "
        "MAX for `visitors_total`). For visitor sensors a missing bucket means no event in that quarter hour "
        "(quiet or closed), not an outage. Zero share and autocorrelation mix in-season and off-season months.",
        "",
        "| Sensor | Rows | First row | Last row | Target | Buckets present/expected | Longest missing run (h) "
        "| Zero share | ACF 1 d | ACF 1 w | Cross-check: columns, buckets, mismatches, max abs diff |",
        "|" + "---|" * 11,
    ]
    for name, s in manifest["sensors"].items():
        q, checks = s["quality"], s["cross_check"].values()
        check = (f"{len(checks)}, {sum(c['buckets_compared'] for c in checks)}, "
                 f"{sum(c['mismatches'] for c in checks)}, {max((c['max_abs_diff'] for c in checks), default=0.0):.2e}")
        lines.append(
            f"| {name} | {s['rows']} | {s['first_row']} | {s['last_row']} | {s['target_column']} "
            f"| {q.get('buckets_present', 0)}/{q.get('buckets_expected', 0)} | {q.get('longest_missing_run_hours', '–')} "
            f"| {q.get('zero_share', '–')} | {q.get('autocorr_1_day', '–')} | {q.get('autocorr_1_week', '–')} | {check} |"
        )
    return "\n".join(lines) + "\n"


# Downloads, cross-checks and writes one snapshot; sensors maps env prefix → (sensor_id, read_key). Nothing is written
# until every sensor has been fetched and checked. Returns the new snapshot folder.
def take_snapshot(sensors: dict[str, tuple[str, Optional[str]]], until: pd.Timestamp, out_root: Path,
                  since: Optional[pd.Timestamp] = None, base_url: str = DEFAULT_BASE_URL) -> Path:
    snapshot_id = f"sensbee-{since:%Y-%m-%d}-{until:%Y-%m-%d}" if since is not None else f"sensbee-{until:%Y-%m-%d}"
    out_dir = out_root / snapshot_id
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} EXISTS; snapshots are never overwritten")

    client = SensbeeClient(base_url)
    raws: dict[str, pd.DataFrame] = {}
    entries: dict[str, dict] = {}
    for name, (sensor_id, api_key) in sensors.items():
        try:
            start = since
            if start is None:
                first = client.get_sensor_data(sensor_id=sensor_id, api_key=api_key, limit=1, ordering="ASC")
                start = pd.Timestamp(first[0][TIME_COL]).floor("D") if first else until
            raw = fetch_raw(client, sensor_id, api_key, start, until)
            checks = cross_check(client, sensor_id, api_key, raw, start, until)
        except httpx.HTTPStatusError as exc:
            # The request URL holds the READ key and the sensor UUID, so only the status code is shown
            raise RuntimeError(f"SENSBEE HTTP {exc.response.status_code} for {name}") from None
        except httpx.HTTPError as exc:
            raise RuntimeError(f"SENSBEE REQUEST FAILED for {name}: {type(exc).__name__}") from None

        target = TARGET_COLUMNS.get(name)
        aggregation = COLUMN_AGGREGATIONS.get(target, DEFAULT_AGGREGATION)
        raws[name] = raw
        entries[name] = {
            "file": f"raw/{name}.csv.gz",
            "rows": len(raw),
            "columns": [c for c in raw.columns if c != TIME_COL],
            "first_row": raw[TIME_COL].min().isoformat() if not raw.empty else None,
            "last_row": raw[TIME_COL].max().isoformat() if not raw.empty else None,
            "target_column": target,
            "target_aggregation": aggregation,
            "quality": (quality_stats(bucket_15min(raw, target, aggregation)) if target in raw.columns
                        else {"buckets_present": 0}),
            "cross_check": checks,
        }
        mismatches = sum(c["mismatches"] for c in checks.values())
        print(f"FETCHED {name}: {len(raw)} rows, cross-check mismatches {mismatches}", file=sys.stderr)

    out_dir.mkdir(parents=True)
    for name, raw in raws.items():
        entries[name]["sha256"] = write_raw_csv(raw, out_dir / entries[name]["file"])
    manifest = {
        "snapshot_id": snapshot_id,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "since": since.isoformat() if since is not None else None,
        "until": until.isoformat(),
        "timestamps": "naive UTC",
        "base_url": base_url,
        "code": git_state(),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "httpx": httpx.__version__},
        "sensors": entries,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    (out_dir / "quality_report.md").write_text(quality_report(manifest))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen snapshot of the thesis sensors (read-only)")
    parser.add_argument("--out", type=Path, required=True, help="parent folder; a new <snapshot id> folder is created")
    parser.add_argument("--until", help="exclusive cutoff date, naive UTC, rounded down to 00:00 (default: today)")
    parser.add_argument("--since", help="start date, naive UTC, rounded down to 00:00 (default: each sensor's "
                                        "first row); for smoke runs")
    parser.add_argument("--sensors", nargs="+", choices=list(TARGET_COLUMNS), default=list(TARGET_COLUMNS))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    sensors: dict[str, tuple[str, Optional[str]]] = {}
    for name in args.sensors:
        sensor_id = (os.getenv(f"{name}_SENSOR_UUID") or "").strip()
        if not sensor_id:
            raise SystemExit(f"MISSING {name}_SENSOR_UUID in the env file")
        sensors[name] = (sensor_id, (os.getenv(f"{name}_MY_API_KEY_READ") or "").strip() or None)

    # Whole days only: month windows then split on 00:00, which is also a bucket edge on the server
    until = (pd.Timestamp(args.until).floor("D") if args.until
             else pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).floor("D"))
    since = pd.Timestamp(args.since).floor("D") if args.since else None
    out_dir = take_snapshot(sensors, until, args.out, since, args.base_url)
    print(f"SNAPSHOT WRITTEN to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
