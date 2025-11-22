"""
SensBee client for fetching sensor data.

NOTE: This module loads data from local JSON file instead of making real API calls.
The live SensBee API integration is disabled for local development.

All methods use data/real_sensbee_json_data.json regardless of sensor_id or api_key parameters.

For a more advanced data loader with resampling and processing capabilities,
see data_loader.py which provides load_sensor_series_from_json().
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Path to the local data file
DATA_FILE = Path(__file__).parent.parent.parent / "data" / "real_sensbee_json_data.json"


# Client for interacting with SensBee sensor data.
class SensBeeClient:
    # Initialize the SensBee client.
    def __init__(self, base_url: Optional[str] = None):
        # Cache is kept for API compatibility but not actively used in fixture mode
        self._cache: dict[Tuple[str, int], Tuple[pd.DataFrame, float]] = {}
        
        logger.info("Initialized SensBee client (fixture-only mode)")

    # Fetch the latest sensor values from local fixture.
    def fetch_latest_values(
        self, sensor_id: str, api_key: str, limit: int = 200
    ) -> pd.DataFrame:
        """
        Args:
            sensor_id: Ignored in fixture mode (kept for API compatibility)
            api_key: Ignored in fixture mode (kept for API compatibility)
            limit: Maximum number of records to retrieve (default: 200)
            
        Returns:
            DataFrame with columns: timestamp (as pandas datetime) and all numeric value columns.
            Data is sorted by timestamp ascending, returns the last 'limit' rows.
        """
        logger.debug(f"Loading local data (sensor_id={sensor_id} ignored, limit={limit})")
        
        # Load data from local JSON file
        if not DATA_FILE.exists():
            raise FileNotFoundError(
                f"Data file not found: {DATA_FILE}\n"
                f"Make sure the file exists at sensbee_nvp/data/real_sensbee_json_data.json"
            )
        
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in data file: {e}")
        
        if not isinstance(data, list):
            raise ValueError("Data file must contain a JSON array")
        
        if len(data) == 0:
            logger.warning("Data file is empty")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Parse created_at as datetime and rename to timestamp
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
            df = df.rename(columns={"created_at": "timestamp"})
            df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
        else:
            logger.warning("No 'created_at' or 'timestamp' column found in data")
        
        # Get the last 'limit' rows (most recent data)
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} records from local data file (limit={limit})")
        return df

    def get_series(
        self,
        sensor_id: str,
        api_key: str,
        column_name: str,
        limit: int = 200,
        max_age_seconds: int = 30,
    ) -> pd.Series:
        """
        Get a time series for a specific column from local fixture.
        
        Args:
            sensor_id: Ignored in fixture mode (kept for API compatibility)
            api_key: Ignored in fixture mode (kept for API compatibility)
            column_name: Name of the column to extract
            limit: Maximum number of records to retrieve (default: 200)
            max_age_seconds: Ignored in fixture mode (kept for API compatibility)
            
        Returns:
            pandas.Series sorted ascending by time (oldest first), truncated to last 'limit' values
            
        Raises:
            ValueError: If the column doesn't exist in the data
        """
        # Fetch data from fixture
        df = self.fetch_latest_values(sensor_id, api_key, limit)
        
        if len(df) == 0:
            raise ValueError("No data available from fixture")
        
        # Check if column exists
        if column_name not in df.columns:
            available_cols = ", ".join(df.columns.tolist())
            error_msg = (
                f"Column '{column_name}' not found in fixture data. "
                f"Available columns: {available_cols}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Extract series and ensure it's sorted ascending by time
        if "timestamp" in df.columns:
            # Set timestamp as index and extract column
            series = df.set_index("timestamp")[column_name].sort_index(ascending=True)
        else:
            # If no timestamp, just return the series
            series = df[column_name].reset_index(drop=True)
            logger.warning("No timestamp column found, returning series without time index")
        
        # Ensure we only return the last 'limit' values
        if len(series) > limit:
            series = series.tail(limit)
        
        return series

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        logger.info("Cache cleared")
