# Frozen SensBee snapshots: raw rows as CSV.gz files plus a manifest with a SHA-256 hash per file.
#
# A snapshot stores raw rows (all columns, naive UTC timestamps), never pre-aggregated data. The 15-min series is
# derived here with the same rule as SensBee's time_grouping: bucket label = floor(epoch / 900 s) * 900 s, then AVG
# or MAX per bucket (sensor_mgmt/src/database/data_db.rs). Buckets without rows are absent, as on the server; gap
# filling and masking belong to the harness, not to the snapshot.

import gzip
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

TIME_COL = "created_at"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
MANIFEST_NAME = "manifest.json"
BUCKET_FREQ = "15min"
BUCKET_HOURS = pd.Timedelta(BUCKET_FREQ) / pd.Timedelta("1h")
BUCKETS_PER_DAY = int(pd.Timedelta("1D") / pd.Timedelta(BUCKET_FREQ))
BUCKETS_PER_WEEK = 7 * BUCKETS_PER_DAY

# SensBee aggregation name → pandas aggregation (both skip nulls).
AGGREGATIONS: dict[str, str] = {"AVG": "mean", "MAX": "max"}


# Returns the SHA-256 hex digest of a file.
def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# Writes raw rows as gzip CSV and returns the file's SHA-256. gzip.compress with mtime=0 stores no timestamp and no
# file name in the header, so the same data always gives the same bytes, whatever the file is called.
def write_raw_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_text = df.to_csv(index=False, date_format=TIME_FORMAT)
    data = gzip.compress(csv_text.encode("utf-8"), mtime=0)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


# Reads one raw CSV.gz file back, with the time column parsed to naive datetimes.
def read_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format=TIME_FORMAT)
    return df


# Loads every raw file listed in a snapshot's manifest after checking its SHA-256. Raises ValueError on a mismatch.
def load_snapshot(snapshot_dir: Path) -> dict[str, pd.DataFrame]:
    manifest = json.loads((snapshot_dir / MANIFEST_NAME).read_text())
    frames: dict[str, pd.DataFrame] = {}
    for name, entry in manifest["sensors"].items():
        path = snapshot_dir / entry["file"]
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(f"SHA-256 MISMATCH for {entry['file']}: manifest {entry['sha256']}, file {digest}")
        frames[name] = read_raw_csv(path)
    logger.info(f"LOADED SNAPSHOT {manifest['snapshot_id']}: {len(frames)} sensors, hashes OK")
    return frames


# Aggregates raw rows into 15-min buckets like SensBee's time_grouping (label = bucket start, AVG → mean, MAX → max).
def bucket_15min(raw: pd.DataFrame, column: str, aggregation: str) -> pd.Series:
    labels = raw[TIME_COL].dt.floor(BUCKET_FREQ)
    series = pd.to_numeric(raw[column], errors="coerce").groupby(labels).agg(AGGREGATIONS[aggregation])
    series.index.name = None
    return series.astype(float)


# Converts NaN to None so the manifest stays valid JSON.
def _finite(value: float, digits: int) -> Optional[float]:
    return round(float(value), digits) if math.isfinite(value) else None


# Data-quality numbers for one 15-min bucket series: coverage between its first and last bucket, the longest run of
# missing buckets, the share of zeros, and the autocorrelation at 1 day and 1 week.
def quality_stats(series: pd.Series) -> dict:
    if series.empty:
        return {"buckets_present": 0}
    full = series.reindex(pd.date_range(series.index[0], series.index[-1], freq=BUCKET_FREQ))
    missing = full.isna()
    longest_run = int(missing.groupby((~missing).cumsum()).sum().max())
    present = full.dropna()
    return {
        "first_bucket": full.index[0].isoformat(),
        "last_bucket": full.index[-1].isoformat(),
        "buckets_expected": len(full),
        "buckets_present": len(present),
        "coverage": _finite(len(present) / len(full), 4),
        "longest_missing_run_hours": longest_run * BUCKET_HOURS,
        "zero_share": _finite((present == 0).mean(), 4),
        "autocorr_1_day": _finite(full.autocorr(BUCKETS_PER_DAY), 3),
        "autocorr_1_week": _finite(full.autocorr(BUCKETS_PER_WEEK), 3),
    }
