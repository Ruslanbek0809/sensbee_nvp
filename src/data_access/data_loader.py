
# Data loader module for loading sensor series from local JSON files.
# Handles both regular (15-min interval) and irregular (event-based) sensor data.

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default path to the local JSON data file
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "real_sensbee_json_data.json"
DEFAULT_COLUMN_NAME = "temperature"

# Event-based sensors fire one record per visitor entry/exit.
# They need a different fill strategy: short gaps → ffill (count unchanged),
# long gaps → fill with 0 (facility is closed overnight).
EVENT_BASED_COLUMNS = {"visitors_total", "visitor_change"}

# After this many minutes with no events, the facility is considered closed.
# Gaps shorter than this are forward-filled (count didn't change).
# Gaps at least this long are filled with 0 (closed, count resets to 0).
CLOSURE_GAP_MINUTES = 120  # 2 hours

# LLMTime (Gruver et al. 2023) finds ~500 values near-optimal for LLM input.
# At 15-min intervals: 7-day = 672 pts, 14-day = 1344 pts.
# Exceeding this budget is intentional in the benchmark to test whether
# over-long input hurts or improves forecast quality.
TOKEN_BUDGET_MAX_POINTS = 700


# Loads a sensor time series from a local JSON file and returns a clean pd.Series.
def load_sensor_series_from_json(
    path: Optional[str] = None,
    column_name: Optional[str] = None,
    resample_interval_minutes: Optional[int] = 15,
) -> pd.Series:
    # Apply default parameters if not provided
    data_path = Path(path) if path else DEFAULT_DATA_PATH
    column_name = column_name or DEFAULT_COLUMN_NAME

    if not data_path.exists():
        raise FileNotFoundError(f"Data file NOT FOUND: {data_path}")

    logger.info(f"LOADING sensor data from: {data_path}")

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {data_path}: {e}")

    if not data:
        raise ValueError("JSON array is empty")

    df = pd.DataFrame(data)

    # Support both raw exports ('created_at') and pre-aggregated exports
    # downloaded with time_grouping ('grouped_time').
    if "grouped_time" in df.columns:
        time_col = "grouped_time"
    elif "created_at" in df.columns:
        time_col = "created_at"
    else:
        raise ValueError("NEITHER 'grouped_time' NOR 'created_at' found in JSON data")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col, ascending=True).set_index(time_col)

    if column_name not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise ValueError(
            f"Column '{column_name}' NOT FOUND. Available columns: {available}"
        )

    series = pd.to_numeric(df[column_name], errors="coerce").dropna()

    is_event_based = column_name in EVENT_BASED_COLUMNS

    if resample_interval_minutes is not None:
        freq = f"{resample_interval_minutes}min"

        if is_event_based:
            # Short gaps within a session: forward-fill (count unchanged).
            # Long gaps overnight: fill with 0 (facility closed).
            ffill_limit = max(1, CLOSURE_GAP_MINUTES // resample_interval_minutes)
            resampled = series.resample(freq).last()
            resampled = resampled.ffill(limit=ffill_limit)
            resampled = resampled.fillna(0)
            series = resampled
            logger.info(
                f"EVENT-BASED resampled '{column_name}' to {resample_interval_minutes}-min: "
                f"{len(series)} PTS  "
                f"(ffill limit={ffill_limit} buckets = {CLOSURE_GAP_MINUTES} min, "
                f"OVERNIGHT GAPS FILL WITH 0)"
            )
        else:
            series = series.resample(freq).last().ffill()
            logger.info(
                f"RESAMPLED '{column_name}' to {resample_interval_minutes}-min: "
                f"{len(series)} POINTS"
            )

    logger.info(
        f"LOADED '{column_name}': {len(series)} POINTS, "
        f"RANGE [{series.min():.2f}, {series.max():.2f}]"
    )

    return series

# Returns the median gap between consecutive timestamps in minutes.
def detect_interval_minutes(series: pd.Series) -> float:
    if len(series) < 2:
        return 15.0
    gaps = series.index.to_series().diff().dropna().dt.total_seconds() / 60
    return float(gaps.median())

# Splits a series into (input_series, ground_truth) for forecast evaluation.
def split_series_for_evaluation(
    series: pd.Series,
    horizon_steps: int,
    input_steps: Optional[int] = None,
) -> tuple[pd.Series, pd.Series]:
    if len(series) <= horizon_steps:
        raise ValueError(
            f"SERIES TOO SHORT ({len(series)} POINTS) FOR HORIZON {horizon_steps}. "
            "USE A LONGER WINDOW DATASET OR REDUCE THE FORECAST HORIZON."
        )

    ground_truth = series.iloc[-horizon_steps:]
    available = series.iloc[:-horizon_steps]

    if input_steps is not None:
        if input_steps > len(available):
            logger.warning(
                f"REQUESTED {input_steps} INPUT STEPS BUT ONLY {len(available)} AVAILABLE. "
                "USING ALL AVAILABLE."
            )
            input_series = available
        else:
            input_series = available.iloc[-input_steps:]
    else:
        input_series = available

    return input_series, ground_truth
