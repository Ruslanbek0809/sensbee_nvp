# SensBee REST API client for fetching sensor data.

import logging
import os
import time
from typing import Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Default base URL
DEFAULT_BASE_URL = "https://sensbee.smartcity.ilmenau.de:8443"

# Client for interacting with the SensBee REST API. Provides methods to fetch sensor data with caching support.
class SensBeeClient:

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("SENSBEE_BASE_URL", DEFAULT_BASE_URL)

        # Remove trailing slash if present
        self.base_url = self.base_url.rstrip("/")
        
        # In-memory cache: keyed by (sensor_id, limit) -> (df, fetched_at)
        self._cache: dict[Tuple[str, int], Tuple[pd.DataFrame, float]] = {}
        
        logger.info(f"Initialized SensBee client with base URL: {self.base_url}")

    def fetch_latest_values(
        self, sensor_id: str, api_key: str, limit: int = 200
    ) -> pd.DataFrame:
        """
        Fetch the latest sensor values from SensBee API.
        
        Args:
            sensor_id: UUID of the sensor
            api_key: API key for authentication
            limit: Maximum number of records to retrieve (default: 200)
            
        Returns:
            DataFrame with columns: timestamp (as pandas datetime) and all numeric value columns
            
        Raises:
            requests.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/api/sensors/{sensor_id}/data/load"
        params = {
            "key": api_key,
            "ordering": "DESC",
            "limit": str(limit),
        }
        
        try:
            logger.debug(f"Fetching data for sensor {sensor_id} with limit {limit}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list):
                logger.warning(f"Unexpected response format: expected list, got {type(data)}")
                return pd.DataFrame()
            
            if len(data) == 0:
                logger.warning(f"No data returned for sensor {sensor_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Parse timestamp column (usually 'created_at')
            timestamp_col = None
            for col in ["created_at", "grouped_time"]:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col:
                try:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                    # Rename to 'timestamp' for consistency
                    df = df.rename(columns={timestamp_col: "timestamp"})
                except Exception as e:
                    logger.warning(f"Failed to parse timestamp column {timestamp_col}: {e}")
                    # Keep original column name if parsing fails
            else:
                logger.warning("No timestamp column found in response")
            
            # Sort by timestamp descending (as returned by API)
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} records for sensor {sensor_id}")
            return df
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching sensor {sensor_id}: {e.response.status_code} - {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching sensor {sensor_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching sensor {sensor_id}: {e}")
            raise

    def get_series(
        self,
        sensor_id: str,
        api_key: str,
        column_name: str,
        limit: int = 200,
        max_age_seconds: int = 30,
    ) -> pd.Series:
        """
        Get a time series for a specific column, using cache if available.
        
        Args:
            sensor_id: UUID of the sensor
            api_key: API key for authentication
            column_name: Name of the column to extract
            limit: Maximum number of records to retrieve (default: 200)
            max_age_seconds: Maximum age of cached data in seconds (default: 30)
            
        Returns:
            pandas.Series sorted ascending by time (oldest first)
            
        Raises:
            KeyError: If the column doesn't exist in the data
        """
        cache_key = (sensor_id, limit)
        current_time = time.time()
        
        # Check cache
        if cache_key in self._cache:
            df, fetched_at = self._cache[cache_key]
            age = current_time - fetched_at
            
            if age < max_age_seconds:
                logger.debug(
                    f"Using cached data for sensor {sensor_id} (age: {age:.1f}s)"
                )
            else:
                logger.debug(
                    f"Cache expired for sensor {sensor_id} (age: {age:.1f}s), fetching new data"
                )
                # Fetch new data
                df = self.fetch_latest_values(sensor_id, api_key, limit)
                self._cache[cache_key] = (df, current_time)
        else:
            # No cache, fetch data
            logger.debug(f"No cache for sensor {sensor_id}, fetching data")
            df = self.fetch_latest_values(sensor_id, api_key, limit)
            self._cache[cache_key] = (df, current_time)
        
        # Check if column exists
        if column_name not in df.columns:
            available_cols = ", ".join(df.columns.tolist())
            error_msg = (
                f"Column '{column_name}' not found in sensor {sensor_id} data. "
                f"Available columns: {available_cols}"
            )
            logger.error(error_msg)
            raise KeyError(error_msg)
        
        # Extract series and sort ascending by time
        if "timestamp" in df.columns:
            series = df.set_index("timestamp")[column_name].sort_index(ascending=True)
        else:
            # If no timestamp, just return the series in reverse order (API returns DESC)
            series = df[column_name].iloc[::-1].reset_index(drop=True)
            logger.warning(
                f"No timestamp column found, returning series without time index"
            )
        
        return series

    # Clear the in-memory cache
    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("Cache cleared")

