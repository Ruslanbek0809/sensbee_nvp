
# Data loader module for loading sensor series from local JSON files.

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default path to the local JSON data file
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "real_sensbee_json_data.json"
DEFAULT_COLUMN_NAME = "temperature"

# Load a sensor time series from a local JSON file and process it.
def load_sensor_series_from_json(
    path: Optional[str] = None,
    column_name: Optional[str] = None,
) -> pd.Series:
    # Apply default parameters if not provided
    data_path = Path(path) if path else DEFAULT_DATA_PATH
    column_name = column_name or DEFAULT_COLUMN_NAME
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file NOT FOUND: {data_path}\n")
    
    logger.info(f"LOADING sensor data from: {data_path}")
    
    # Load JSON file
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {data_path}: {e}")
    
    if len(data) == 0:
        raise ValueError("JSON array is empty")
    
    logger.debug(f"LOADED {len(data)} records from JSON file")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Parse created_at as datetime
    if "created_at" not in df.columns:
        raise ValueError(f"Column 'created_at' NOT FOUND in JSON data")
    
    try:
        df["created_at"] = pd.to_datetime(df["created_at"])
    except Exception as e:
        raise ValueError(f"FAILED to parse 'created_at' as datetime: {e}")
    
    # Sort ascending by time and set as index
    df = df.sort_values("created_at", ascending=True).reset_index(drop=True)
    df = df.set_index("created_at")
    
    # Check if the requested column exists
    if column_name not in df.columns:
        available_cols = ", ".join(df.columns.tolist())
        raise ValueError(f"Column '{column_name}' NOT FOUND in JSON data")
    
    # Select the requested column
    series = df[column_name].copy()
    
    logger.debug(f"LOADED series for column '{column_name}', length: {len(series)}")
    
    return series

