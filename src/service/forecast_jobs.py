import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from ..data_access.sensbee_client import load_sensor_series_from_api
from ..models.nvp_llms import nvp_llms_forecast

logger = logging.getLogger(__name__)
# # Loads config (Commented out for now)
# CONFIG_PATH = os.getenv("FORECAST_CONFIG", "config/forecast_config.yaml")


# # Loads forecast configuration. (Commented out for now)
# def load_config() -> dict:
#     with open(CONFIG_PATH, "r") as f:
#         return yaml.safe_load(f)

# Remote inference service URL (GPU node).
# Set via environment variable or defaults to localhost.
# INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8001")


# Calls remote inference service on GPU node. 
# def call_remote_inference(
#     values: list,
#     horizon: int,
#     column_name: str,
#     model: str = "mistral-7b",
#     num_forecasts: int = 3,
#     temperature: float = 0.7,
# ) -> np.ndarray:
#     import httpx
    
#     url = f"{INFERENCE_SERVICE_URL}/predict"
#     payload = {
#         "values": values,
#         "horizon": horizon,
#         "column_name": column_name,
#         "model": model,
#         "num_forecasts": num_forecasts,
#         "temperature": temperature,
#     }
    
#     logger.info(f"CALLING REMOTE INFERENCE: {url}")
    
#     # Long timeout for LLM inference (up to 5 minutes)
#     response = httpx.post(url, json=payload, timeout=300.0)
#     response.raise_for_status()
    
#     result = response.json()
#     logger.info(
#         f"REMOTE INFERENCE COMPLETE: {result['forecast_points']} points "
#         f"in {result['inference_time_seconds']:.2f}s"
#     )
    
#     return np.array(result["forecast"])


# Generates a forecast for a single column.
def forecast_single_column(
    sensor_id: str,
    api_key: Optional[str],
    column_name: str,
    horizon_hours: int,
    history_days: int,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    num_forecasts: int = 5,
    temperature: float = 0.9,
    use_normalization: bool = True,
    include_context: bool = False,
) -> dict:
    # History window in hours; the SensBee client derives the API limit from it
    window_hours = history_days * 24

    logger.info(
        f"FORECASTING {column_name} for sensor {sensor_id}, "
        f"history={history_days}d ({window_hours}h), horizon={horizon_hours}h, "
        f"provider={provider}, norm={use_normalization}, ctx={include_context}"
    )

    # Loads latest data from SensBee API
    series = load_sensor_series_from_api(
        sensor_id=sensor_id,
        api_key=api_key,
        column_name=column_name,
        window_hours=window_hours,
    )
    
    if len(series) == 0:
        raise ValueError(f"NO DATA returned for sensor {sensor_id}")
    
    # Calculates horizon in steps
    sampling_minutes = (series.index[1] - series.index[0]).total_seconds() / 60
    steps_per_hour = 60 / sampling_minutes
    horizon_steps = int(horizon_hours * steps_per_hour)
    
    # Generates forecast using LLM
    forecast = nvp_llms_forecast(
        series=series,
        horizon=horizon_steps,
        provider=provider,
        model=model,
        column_name=column_name,
        num_forecasts=num_forecasts,
        temperature=temperature,
        use_normalization=use_normalization,
        include_context=include_context,
    )
    
    # Generates forecast timestamps
    last_ts = series.index[-1]
    forecast_timestamps = pd.date_range(
        start=last_ts + pd.Timedelta(minutes=sampling_minutes),
        periods=len(forecast),
        freq=f"{sampling_minutes}min"
    )
    
    # TODO for the future: Storing forecast to SensBee can be implemented here
    
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


# # Forecasts all configured columns for a given sensor (Commented out for now)
# def forecast_sensor_all_columns(sensor_config: dict) -> list[dict]:
#     results = []
    
#     sensor_id = sensor_config["source_sensor_id"]
#     api_key = sensor_config["source_api_key"]
    
#     for col_config in sensor_config["columns"]:
#         column_name = col_config["name"]
#         horizon_hours = col_config["horizon_hours"]
#         history_days = col_config["history_days"]
        
#         try:
#             result = forecast_single_column(
#                 sensor_id=sensor_id,
#                 api_key=api_key,
#                 column_name=column_name,
#                 horizon_hours=horizon_hours,
#                 history_days=history_days,
#             )
#             results.append(result)
#         except Exception as e:
#             logger.error(f"FAILED to forecast {column_name}: {e}")
#             results.append({
#                 "column": column_name,
#                 "error": str(e),
#             })
    
#     return results


# # Main job function to run all scheduled forecasts. (Commented out for now)
# def run_scheduled_forecasts() -> dict:
#     config = load_config()
#     all_results = {}
    
#     logger.info(f"STARTING scheduled forecast run for {len(config['sensors'])} sensors")
    
#     for sensor_config in config["sensors"]:
#         sensor_name = sensor_config.get("name", sensor_config["source_sensor_id"])
#         logger.info(f"PROCESSING sensor: {sensor_name}")
        
#         try:
#             results = forecast_sensor_all_columns(sensor_config)
#             all_results[sensor_name] = results
#         except Exception as e:
#             logger.error(f"FAILED to process sensor {sensor_name}: {e}")
#             all_results[sensor_name] = {"error": str(e)}
    
#     logger.info("COMPLETED scheduled forecast run")
#     return all_results

