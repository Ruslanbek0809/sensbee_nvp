# HTTP client for fetching sensor data from SensBee API.

import logging
import os
from datetime import datetime
from typing import Optional

import httpx
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Default SensBee API parameters
DEFAULT_BASE_URL = "https://sensbee.smartcity.ilmenau.de:8443"
DEFAULT_SENSOR_ID = os.getenv("MANEBACH_WEATHER_STATION_SENSOR_UUID")
DEFAULT_API_KEY = os.getenv("MANEBACH_WEATHER_STATION_MY_API_KEY_READ")
DEFAULT_LIMIT = 672
DEFAULT_ORDERING = "DESC"
DEFAULT_COLUMN_NAME = "humidity"

# Simple HTTP client for SensBee API.
class SensbeeClient:

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")
    
    # # Fetch list of accessible sensors.
    # def list_sensors(self) -> list[dict]:
    #     url = f"{self.base_url}/api/sensors/list"
    #     logger.debug(f"list_sensors => GET {url}")
        
    #     with httpx.Client(timeout=30.0) as client:
    #         resp = client.get(url)
    #         resp.raise_for_status()
    #         return resp.json()
    
    # Fetch sensor data from SensBee API.
    def get_sensor_data(
        self,
        sensor_id: Optional[str] = None,
        api_key: Optional[str] = None,
        limit: Optional[int] = None,
        ordering: Optional[str] = None,
        cols: Optional[list[str]] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
    ) -> list[dict]:
        url = f"{self.base_url}/api/sensors/{sensor_id}/data/load"
        
        params = {
            "limit": limit,
            "ordering": ordering,
        }
        
        # Only include key for private sensors (public sensors don't need it)
        if api_key:
            params["key"] = api_key
        if cols:
            params["cols"] = ",".join(cols)
        if from_time:
            params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%S")
        if to_time:
            params["to"] = to_time.strftime("%Y-%m-%dT%H:%M:%S")
        
        logger.debug(f"GET {url} params={params}")
        
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

# Loads a sensor time series from SensBee API. 
def load_sensor_series_from_api(
    base_url: Optional[str] = None,
    sensor_id: Optional[str] = None,
    api_key: Optional[str] = None,
    limit: Optional[int] = None,
    ordering: Optional[str] = None,
    column_name: Optional[str] = None,
) -> pd.Series:
    # Apply default parameters if not provided
    base_url = base_url or DEFAULT_BASE_URL
    sensor_id = sensor_id or DEFAULT_SENSOR_ID
    api_key = api_key or DEFAULT_API_KEY
    limit = limit or DEFAULT_LIMIT
    ordering = ordering or DEFAULT_ORDERING
    column_name = column_name or DEFAULT_COLUMN_NAME
    
    client = SensbeeClient(base_url)
    
    # Fetch data ordered ASC (oldest first) for forecasting
    data = client.get_sensor_data(
        sensor_id=sensor_id,
        api_key=api_key,
        limit=limit,
        ordering=ordering,
    )
    
    if len(data) == 0:
        raise ValueError(f"NO DATA returned for sensor {sensor_id}")
    
    logger.info(f"FETCHED {len(data)} rows from sensor {sensor_id}")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Parse created_at as datetime
    if "created_at" not in df.columns:
        raise ValueError("MISSING 'created_at' column")
    
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.sort_values("created_at", ascending=True).set_index("created_at")
    
    # Check requested column exists
    if column_name not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise ValueError(f"Column '{column_name}' NOT FOUND. AVAILABLE ONES: {available}")
    
    series = df[column_name].copy()
    logger.debug(f"LOADED series for column '{column_name}' with {len(series)} points")
    
    return series


# # Load multiple columns from SensBee API.
# def load_sensor_dataframe_from_api(
#     base_url: Optional[str] = None,
#     sensor_id: Optional[str] = None,
#     api_key: Optional[str] = None,
#     limit: Optional[int] = None,
#     ordering: Optional[str] = None,
#     columns: Optional[list[str]] = None,
# ) -> pd.DataFrame:
#     base_url = base_url or DEFAULT_BASE_URL
#     sensor_id = sensor_id or DEFAULT_SENSOR_ID
#     api_key = api_key or DEFAULT_API_KEY
#     limit = limit or DEFAULT_LIMIT
#     ordering = ordering or DEFAULT_ORDERING
    
#     client = SensbeeClient(base_url)
    
#     data = client.get_sensor_data(
#         sensor_id=sensor_id,
#         api_key=api_key,
#         limit=limit,
#         ordering=ordering,
#         cols=columns,
#     )
    
#     if len(data) == 0:
#         raise ValueError(f"NO DATA returned for sensor {sensor_id}")
    
#     df = pd.DataFrame(data)
#     df["created_at"] = pd.to_datetime(df["created_at"])
#     df = df.sort_values("created_at", ascending=True).set_index("created_at")
    
#     logger.info(f"LOADED {len(df)} rows with columns: {df.columns.tolist()}")
#     return df