#!/usr/bin/env python3
# Read-only audit of the SensBee sensors this project can reach: history length, gaps, storage type, timestamp timezone.
#
# Sensors come from .env pairs (<NAME>_SENSOR_UUID + optional <NAME>_MY_API_KEY_READ). With --public, every sensor in
# the public GET /api/sensors/list is audited too. API keys are read from the environment and never printed or saved.
#
# Usage (from sensbee_nvp/):
#   venv/bin/python scripts/audit_sensors.py                          # Markdown report to stdout
#   venv/bin/python scripts/audit_sensors.py --public --json out.json # plus public sensors, details as JSON

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_access.sensbee_client import DEFAULT_BASE_URL  # noqa: E402

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DAY_SECONDS = 86400


# Collects (name, sensor_id, read_key) triples from <NAME>_SENSOR_UUID / <NAME>_MY_API_KEY_READ env pairs.
def sensors_from_env() -> list[dict[str, Optional[str]]]:
    sensors = []
    for var, value in sorted(os.environ.items()):
        if var.endswith("_SENSOR_UUID") and value.strip():
            prefix = var[: -len("_SENSOR_UUID")]
            key = (os.getenv(f"{prefix}_MY_API_KEY_READ") or "").strip() or None
            sensors.append({"name": prefix, "id": value.strip(), "key": key, "source": "env file"})
    return sensors


# Calls a SensBee GET endpoint and returns (status_code, parsed_json_or_None). Never raises on HTTP errors.
def api_get(client: httpx.Client, path: str, params: dict[str, Any]) -> tuple[int, Any]:
    try:
        resp = client.get(path, params=params)
    except httpx.HTTPError as exc:
        return -1, str(exc)
    try:
        body = resp.json()
    except ValueError:
        body = None
    return resp.status_code, body


# Parses a SensBee timestamp (naive, RFC3339 without timezone) into a naive datetime.
def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", ""))


# Audits one sensor: info, first/last row, daily row counts over the whole history.
def audit_sensor(client: httpx.Client, sensor: dict[str, Optional[str]]) -> dict[str, Any]:
    sid, key = sensor["id"], sensor["key"]
    base = {"key": key} if key else {}
    report: dict[str, Any] = {"name": sensor["name"], "id": sid, "source": sensor["source"], "has_key": bool(key)}

    status, info = api_get(client, f"/api/sensors/{sid}/info", base)
    report["info_status"] = status
    if status == 200 and isinstance(info, dict):
        full = info.get("sensor_info", info)
        report["sensor_name"] = full.get("name")
        report["storage_type"] = full.get("storage_type")
        report["storage_params"] = full.get("storage_params")
        report["columns_info"] = [
            {"name": c.get("name"), "type": c.get("val_type"), "unit": c.get("val_unit")} for c in full.get("columns", [])
        ]

    status, first = api_get(client, f"/api/sensors/{sid}/data/load", {**base, "ordering": "ASC", "limit": 1})
    report["data_status"] = status
    if status != 200 or not first:
        report["access"] = "no data" if status == 200 else f"HTTP {status}"
        return report
    _, last = api_get(client, f"/api/sensors/{sid}/data/load", {**base, "ordering": "DESC", "limit": 1})

    first_row, last_row = first[0], last[0]
    report["access"] = "ok"
    report["columns"] = [c for c in first_row if c != "created_at"]
    report["timestamp_sample"] = last_row.get("created_at")
    t_first, t_last = parse_ts(first_row["created_at"]), parse_ts(last_row["created_at"])
    report["first"], report["last"] = t_first.strftime(TIME_FORMAT), t_last.strftime(TIME_FORMAT)
    report["span_days"] = round((t_last - t_first).total_seconds() / DAY_SECONDS, 1)

    # Minutes between "now" and the newest row, if SensBee timestamps were UTC. About -120 would mean local time (CEST).
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    report["newest_row_age_min_if_utc"] = round((now_utc - t_last).total_seconds() / 60, 1)

    # Daily COUNT buckets over the whole history (buckets without rows are not returned by the API).
    count_col = next((c for c in report["columns"] if isinstance(first_row.get(c), (int, float))), None)
    if count_col is None:
        return report
    day_start = t_first.replace(hour=0, minute=0, second=0, microsecond=0)
    n_days = (t_last - day_start).days + 1
    status, daily = api_get(client, f"/api/sensors/{sid}/data/load", {
        **base,
        "from": day_start.strftime(TIME_FORMAT),
        "to": (t_last + timedelta(seconds=1)).strftime(TIME_FORMAT),
        "cols": f"{count_col}.COUNT",
        "time_grouping": DAY_SECONDS,
        "ordering": "ASC",
        "limit": n_days + 10,
    })
    if status != 200 or not isinstance(daily, list):
        report["daily_status"] = status
        return report

    counts = {parse_ts(r["grouped_time"]).date(): int(r.get(count_col) or 0) for r in daily}
    days = [day_start.date() + timedelta(days=i) for i in range(n_days)]
    per_day = [counts.get(d, 0) for d in days]
    longest, run = 0, 0
    for c in per_day:
        run = run + 1 if c == 0 else 0
        longest = max(longest, run)
    nonzero = sorted(c for c in per_day if c > 0)
    report["count_column"] = count_col
    report["rows_total"] = sum(per_day)
    report["days_total"] = n_days
    report["days_with_data"] = len(nonzero)
    report["median_rows_per_active_day"] = nonzero[len(nonzero) // 2] if nonzero else 0
    report["longest_gap_days"] = longest
    report["monthly_rows"] = {}
    for d, c in zip(days, per_day):
        month = d.strftime("%Y-%m")
        report["monthly_rows"][month] = report["monthly_rows"].get(month, 0) + c
    return report


# Formats the per-sensor results as a Markdown table.
def markdown_table(results: list[dict[str, Any]]) -> str:
    head = ("| Sensor | Access | Storage | First | Last | Span (d) | Days with data | Rows | Rows/day (median) "
            "| Longest gap (d) | Newest row age if UTC (min) | Columns |")
    lines = [head, "|" + "---|" * 12]
    for r in results:
        cols = ", ".join(r.get("columns", []))
        lines.append(
            f"| {r['name']} | {r.get('access', '?')} | {r.get('storage_type') or '–'} | {r.get('first', '–')} "
            f"| {r.get('last', '–')} | {r.get('span_days', '–')} | {r.get('days_with_data', '–')}/{r.get('days_total', '–')} "
            f"| {r.get('rows_total', '–')} | {r.get('median_rows_per_active_day', '–')} | {r.get('longest_gap_days', '–')} "
            f"| {r.get('newest_row_age_min_if_utc', '–')} | {cols} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit of SensBee sensor history")
    parser.add_argument("--public", action="store_true", help="also audit every sensor in /api/sensors/list")
    parser.add_argument("--json", type=Path, help="write per-sensor details to this JSON file")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    sensors = sensors_from_env()

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=60.0) as client:
        status, public = api_get(client, "/api/sensors/list", {})
        public = public if status == 200 and isinstance(public, list) else []
        known = {s["id"] for s in sensors}
        names = {p.get("id"): p.get("name") for p in public}
        for s in sensors:
            s["public_name"] = names.get(s["id"])
        if args.public:
            sensors += [{"name": p.get("name", "?"), "id": p["id"], "key": None, "source": "public list"}
                        for p in public if p.get("id") not in known]

        results = []
        for s in sensors:
            result = audit_sensor(client, s)
            result["public_name"] = s.get("public_name")
            results.append(result)
            print(f"audited {s['name']}: {result.get('access')}", file=sys.stderr)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"# SensBee sensor audit ({generated})\n")
    print(f"Base URL: {args.base_url} · sensors from the env file: {len(known)} · public sensors listed: {len(public)}\n")
    print(markdown_table(results))
    if public:
        print("\n## Public sensor list\n\n| Name | ID | Lat | Lon | In env file |\n|---|---|---|---|---|")
        for p in sorted(public, key=lambda p: p.get("name", "")):
            print(f"| {p.get('name')} | {p.get('id')} | {p.get('latitude')} | {p.get('longitude')} | {p.get('id') in known} |")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"generated_at": generated, "base_url": args.base_url, "sensors": results, "public_list": public}
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nDetails written to {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
