
# Data loader module for loading sensor series from local JSON files.

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default path to the local JSON data file
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "real_sensbee_json_data.json"

# Load a sensor time series from a local JSON file and process it.
def load_sensor_series_from_json(
    path: Optional[str] = None,
    column_name: str = "temperature", # For now, default to temperature
    resample_rule: Optional[str] = None,
    history_hours: int = 24 * 7,
) -> pd.Series:
    # Use default path if not provided
    if path is None:
        data_path = DEFAULT_DATA_PATH
    else:
        data_path = Path(path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}\n")
    
    logger.info(f"Loading sensor data from: {data_path}")
    
    # Load JSON file
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {data_path}: {e}")
    
    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain a JSON array, got {type(data).__name__}")
    
    if len(data) == 0:
        raise ValueError("JSON array is empty")
    
    logger.debug(f"Loaded {len(data)} records from JSON file")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Parse created_at as datetime
    if "created_at" not in df.columns:
        raise ValueError(f"Column 'created_at' not found in JSON data. Available columns: {', '.join(df.columns.tolist())}")
    
    try:
        df["created_at"] = pd.to_datetime(df["created_at"])
    except Exception as e:
        raise ValueError(f"Failed to parse 'created_at' as datetime: {e}")
    
    # Sort ascending by time and set as index
    df = df.sort_values("created_at", ascending=True).reset_index(drop=True)
    df = df.set_index("created_at")
    
    # Check if the requested column exists
    if column_name not in df.columns:
        available_cols = ", ".join(df.columns.tolist())
        raise ValueError(f"Column '{column_name}' not found in JSON data. Available columns: {available_cols}")
    
    # Select the requested column
    series = df[column_name].copy()
    
    logger.debug(f"Extracted series for column '{column_name}', length: {len(series)}")
    
    return series

