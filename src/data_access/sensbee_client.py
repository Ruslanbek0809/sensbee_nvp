# HTTP client for fetching sensor data from SensBee API. There are few things to note:
#
# All API fetches use server-side time_grouping for pre-aggregated buckets.
# The server handles aggregation; the client only fills sparse gaps afterward.
#
# limit must always be set when using time_grouping — otherwise the API
# returns only 10 rows regardless of the window size (backend default).
# This module computes limit automatically from the window and interval.

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL    = "https://sensbee.smartcity.ilmenau.de:8443"
DEFAULT_SENSOR_ID   = os.getenv("MANEBACH_WEATHER_STATION_SENSOR_UUID")
DEFAULT_API_KEY     = os.getenv("MANEBACH_WEATHER_STATION_MY_API_KEY_READ")
DEFAULT_COLUMN_NAME = "temperature"

# Server-side aggregation per column.
# visitors_total → MAX: peak occupancy in the bucket (entries minus exits can go down,
#   so MAX captures the busiest point in each 15-min window).
# All other numeric columns → AVG by default.
# NOTE: LAST is not supported by the SensBee backend.
COLUMN_AGGREGATIONS: dict[str, str] = {
    "visitors_total": "MAX",
}
DEFAULT_AGGREGATION = "AVG"

# Gaps this long with no visitor events mean the facility is closed.
# Shorter gaps are forward-filled (occupancy unchanged between events).
CLOSURE_GAP_MINUTES = 120


# HTTP client for fetching sensor data from SensBee API.    
class SensbeeClient:

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")

    # Fetches sensor data from the SensBee API. When time_grouping_seconds is set, the API groups events into fixed-size buckets and returns 'grouped_time' instead of 'created_at'. Each cols entry must include an aggregation: e.g. "visitors_total.MAX".
    def get_sensor_data(
        self,
        sensor_id: str,
        api_key: Optional[str] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        ordering: str = "ASC",
        cols: Optional[list[str]] = None,
        time_grouping_seconds: Optional[int] = None,
    ) -> list[dict]:
        url = f"{self.base_url}/api/sensors/{sensor_id}/data/load"
        params: dict = {"ordering": ordering}

        if api_key:
            params["key"] = api_key
        if from_time:
            params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%S")
        if to_time:
            params["to"] = to_time.strftime("%Y-%m-%dT%H:%M:%S")
        if limit is not None:
            params["limit"] = limit
        if cols:
            params["cols"] = ",".join(cols)
        if time_grouping_seconds is not None:
            params["time_grouping"] = time_grouping_seconds

        # The READ key is left out of the log.
        logger.debug("GET %s  params=%s", url, {k: v for k, v in params.items() if k != "key"})

        with httpx.Client(timeout=60.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

# Fetches a sensor time series from SensBee, pre-aggregated into regular buckets by the server (time_grouping). The client fills any remaining sparse gaps.
def load_sensor_series_from_api(
    sensor_id: Optional[str] = None,
    api_key: Optional[str] = None,
    column_name: Optional[str] = None,
    window_hours: Optional[int] = None,
    from_time: Optional[datetime] = None,
    to_time: Optional[datetime] = None,
    resample_interval_minutes: int = 15,
    base_url: Optional[str] = None,
) -> pd.Series:
    sensor_id   = sensor_id   or DEFAULT_SENSOR_ID
    api_key     = api_key     or DEFAULT_API_KEY
    column_name = column_name or DEFAULT_COLUMN_NAME
    base_url    = base_url    or DEFAULT_BASE_URL

    if from_time is None and to_time is None:
        if window_hours is None:
            raise ValueError("PROVIDE EITHER window_hours or from_time/to_time.")
        # SensBee compares from/to with naive UTC timestamps, so "now" must be UTC, not local time.
        to_time   = datetime.now(timezone.utc).replace(tzinfo=None)
        from_time = to_time - timedelta(hours=window_hours)

    # Auto-calculate limit: one bucket per interval across the full window + buffer.
    window_minutes = (to_time - from_time).total_seconds() / 60
    limit = math.ceil(window_minutes / resample_interval_minutes) + 10

    aggregation = COLUMN_AGGREGATIONS.get(column_name, DEFAULT_AGGREGATION)
    cols_param  = [f"{column_name}.{aggregation}"]

    client  = SensbeeClient(base_url)
    records = client.get_sensor_data(
        sensor_id=sensor_id,
        api_key=api_key,
        from_time=from_time,
        to_time=to_time,
        limit=limit,
        ordering="ASC",
        cols=cols_param,
        time_grouping_seconds=resample_interval_minutes * 60,
    )

    if not records:
        raise ValueError(
            f"NO DATA for sensor {sensor_id} between {from_time} and {to_time}."
        )

    logger.info(
        "FETCHED %d grouped records from sensor %s (%s → %s, %s)",
        len(records), sensor_id, from_time, to_time,
        f"{column_name}.{aggregation}",
    )

    df = pd.DataFrame(records)
    # Time-grouping responses use 'grouped_time'; raw responses use 'created_at'.
    time_col = "grouped_time" if "grouped_time" in df.columns else "created_at"
    df[time_col] = pd.to_datetime(df[time_col])
    series = pd.to_numeric(df.set_index(time_col)[column_name], errors="coerce")

    # Reindex over the full requested window so overnight zeros are included.
    # SensBee labels buckets floor(epoch / interval) * interval (:00/:15/:30/:45), so the grid is floored the same way.
    freq       = f"{resample_interval_minutes}min"
    full_index = pd.date_range(
        start=pd.Timestamp(from_time).floor(freq),
        end=pd.Timestamp(to_time).floor(freq),
        freq=freq,
    )
    series = series.reindex(full_index)

    is_event_based = column_name in {"visitors_total", "visitor_change"}
    if is_event_based:
        ffill_limit = max(1, CLOSURE_GAP_MINUTES // resample_interval_minutes)
        series = series.ffill(limit=ffill_limit).fillna(0)
        series = series.clip(lower=0)  # negative sensor anomalies → 0
    else:
        series = series.ffill().dropna()

    logger.info(
        "FINAL SERIES '%s': %d PTS, RANGE [%.2f, %.2f]",
        column_name, len(series), series.min(), series.max(),
    )
    return series
