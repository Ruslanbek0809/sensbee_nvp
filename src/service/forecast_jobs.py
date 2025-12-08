import logging
import os
from typing import Optional

import pandas as pd
import yaml

from ..data_access.sensbee_client import load_sensor_series_from_api
from ..models.llmtime_wrapper import llmtime_forecast

logger = logging.getLogger(__name__)

# Loads config
CONFIG_PATH = os.getenv("FORECAST_CONFIG", "config/forecast_config.yaml")


# Loads forecast configuration.
def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# Generates a forecast for a single column.
def forecast_single_column(
    sensor_id: str,
    api_key: str,
    column_name: str,
    horizon_hours: int,
    history_days: int,
    provider: Optional[str] = None,
) -> dict:
    # Calculates limit (history_days * 4 points per hour * 24 hours)
    limit = history_days * 24 * 4
    
    logger.info(
        f"FORECASTING {column_name} for sensor {sensor_id}, "
        f"history={history_days}d ({limit} points), horizon={horizon_hours}h"
    )
    
    # Loads latest data (DESC to get most recent points, then sorted ASC for time series)
    series = load_sensor_series_from_api(
        sensor_id=sensor_id,
        api_key=api_key,
        column_name=column_name,
        limit=limit,
    )
    
    if len(series) == 0:
        raise ValueError(f"NO DATA returned for sensor {sensor_id}")
    
    sampling_minutes = (series.index[1] - series.index[0]).total_seconds() / 60
    steps_per_hour = 60 / sampling_minutes
    horizon_steps = int(horizon_hours * steps_per_hour)
    
    # Generates forecast
    forecast = llmtime_forecast(
        series=series,
        horizon=horizon_steps,
        provider=provider,
        column_name=column_name,
    )
    
    # Generates timestamps
    last_ts = series.index[-1]
    forecast_timestamps = pd.date_range(
        start=last_ts + pd.Timedelta(minutes=sampling_minutes),
        periods=len(forecast),
        freq=f"{sampling_minutes}min"
    )
    
    # TODO: Storing forecast to SensBee can be implemented here
    
    return {
        "column": column_name,
        "sensor_id": sensor_id,
        "input_points": len(series),
        "forecast_points": len(forecast),
        "horizon_hours": horizon_hours,
        "last_data_timestamp": series.index[-1].isoformat(),
        "forecast_start": forecast_timestamps[0].isoformat(),
        "forecast_end": forecast_timestamps[-1].isoformat(),
        "forecast_values": forecast.tolist(),
        "forecast_timestamps": [ts.isoformat() for ts in forecast_timestamps],
    }


# Forecasts all configured columns for a given sensor
def forecast_sensor_all_columns(sensor_config: dict) -> list[dict]:
    results = []
    
    sensor_id = sensor_config["source_sensor_id"]
    api_key = sensor_config["source_api_key"]
    
    for col_config in sensor_config["columns"]:
        column_name = col_config["name"]
        horizon_hours = col_config["horizon_hours"]
        history_days = col_config["history_days"]
        
        try:
            result = forecast_single_column(
                sensor_id=sensor_id,
                api_key=api_key,
                column_name=column_name,
                horizon_hours=horizon_hours,
                history_days=history_days,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"FAILED to forecast {column_name}: {e}")
            results.append({
                "column": column_name,
                "error": str(e),
            })
    
    return results


# Main job function to run all scheduled forecasts.
def run_scheduled_forecasts() -> dict:
    config = load_config()
    all_results = {}
    
    logger.info(f"STARTING scheduled forecast run for {len(config['sensors'])} sensors")
    
    for sensor_config in config["sensors"]:
        sensor_name = sensor_config.get("name", sensor_config["source_sensor_id"])
        logger.info(f"PROCESSING sensor: {sensor_name}")
        
        try:
            results = forecast_sensor_all_columns(sensor_config)
            all_results[sensor_name] = results
        except Exception as e:
            logger.error(f"FAILED to process sensor {sensor_name}: {e}")
            all_results[sensor_name] = {"error": str(e)}
    
    logger.info("COMPLETED scheduled forecast run")
    return all_results

