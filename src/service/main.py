
# FastAPI service for on-demand next-value prediction of smart city sensor data. 

import asyncio
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .forecast_jobs import forecast_single_column

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Here, we define the request body for the forecast endpoint.
class ForecastRequest(BaseModel):
    sensor_id: str = Field(
        ..., 
        description="UUID of the SensBee sensor"
    )
    api_key: Optional[str] = Field(
        default=None, 
        description="READ API key for the sensor"
    )
    column: str = Field(
        ..., 
        description="Column name to forecast (e.g., 'temperature', 'humidity')"
    )
    horizon_hours: int = Field(
        default=24, 
        ge=6, # min 6 hours
        le=168, # max 1 week
        description="Forecast horizon in hours (6-168)"
    )
    history_days: int = Field(
        default=7, 
        ge=1, # min 1 day
        le=14, # max 14 days
        description="Days of history to use (1-14)"
    )
    provider: str = Field(
        default="mistral", 
        description="LLM provider: 'mistral', 'groq', 'openai', or 'local' (GPU required)"
    )
    num_samples: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of LLM samples (1-10). Higher = better quality but slower & more expensive"
    )
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=1.5,
        description="LLM sampling temperature (0.1-1.5). Higher = more diversity"
    )

# Here, we define the response body for the forecast endpoint.
class ForecastResponse(BaseModel):
    sensor_id: str
    column: str
    forecast_values: list[float]
    forecast_timestamps: list[str]
    input_points: int
    forecast_points: int
    horizon_hours: int
    last_data_timestamp: str
    generated_at: str
    provider: str
    cached: bool = False


# Here, we define the in-memory cache for the forecast responses.
forecast_cache: dict[str, ForecastResponse] = {}

# Here, we build the cache key for the forecast responses.
def build_cache_key(sensor_id: str, column: str) -> str:
    return f"{sensor_id}_{column}"


app = FastAPI(
    title="SensBee NVP Service",
    description=("On-demand next-value prediction service for smart city sensor data. It uses LLM-based forecasting to predict future sensor values."),
    version="0.2.0",
)

### API Endpoints

# Fast way to verify the service is up without triggering the logic. Also, it shows the cache size.
@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "cache_size": len(forecast_cache),
    }

# Here, we generate on-demand forecast response based on user-provided parameters.
@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest) -> ForecastResponse:
    logger.info(f"/forecast Forecast POST request: sensor={request.sensor_id}, column={request.column}, horizon={request.horizon_hours}h, history={request.history_days}d, provider={request.provider}")
    
    try:
        # Run forecast in thread pool to avoid blocking the event loop. It runs the forecast_single_column function in a separate thread to avoid blocking the main thread
        # because forecast_single_column can take a while to complete based on the history and horizon parameters' values entered by the user.
        result = await asyncio.to_thread(
            forecast_single_column,
            sensor_id=request.sensor_id,
            api_key=request.api_key,
            column_name=request.column,
            horizon_hours=request.horizon_hours,
            history_days=request.history_days,
            provider=request.provider,
            num_samples=request.num_samples,
            temperature=request.temperature,
        )
        
        # Build response
        response = ForecastResponse(
            sensor_id=request.sensor_id,
            column=request.column,
            forecast_values=result["forecast_values"],
            forecast_timestamps=result["forecast_timestamps"],
            input_points=result["input_points"],
            forecast_points=result["forecast_points"],
            horizon_hours=result["horizon_hours"],
            last_data_timestamp=result["last_data_timestamp"],
            generated_at=datetime.now().isoformat(),
            provider=request.provider,
            cached=False,
        )
        
        # Store in cache
        cache_key = build_cache_key(request.sensor_id, request.column)
        forecast_cache[cache_key] = response
        logger.info(f"/forecast CACHED forecast: {cache_key}")
        
        return response
        
    except ValueError as e:
        logger.error(f"/forecast INVALID VALUE error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"/forecast Forecast FAILED: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Here, you can retrieve a cached forecast for the given sensor and column. 404 if no cached forecast exists.
@app.get("/forecast/cache/{sensor_id}/{column}", response_model=ForecastResponse)
async def get_cached_forecast(sensor_id: str, column: str) -> ForecastResponse:
    cache_key = build_cache_key(sensor_id, column)
    
    if cache_key not in forecast_cache:
        raise HTTPException(
            status_code=404, 
            detail=f"NO CACHED forecast for sensor={sensor_id}, column={column}"
        )
    
    response = forecast_cache[cache_key]
    response.cached = True
    return response


# Here, you can list all cached forecasts with their metadata.
@app.get("/forecast/cache")
async def list_cached_forecasts() -> dict:
    return {
        "count": len(forecast_cache),
        "forecasts": [
            {
                "sensor_id": item.sensor_id,
                "column": item.column,
                "generated_at": item.generated_at,
                "forecast_points": item.forecast_points,
            }
            for item in forecast_cache.values()
        ],
    }


# Here, you can clear all cached forecasts manually with a request.
@app.delete("/forecast/cache")
async def clear_cache() -> dict:
    count = len(forecast_cache)
    forecast_cache.clear()
    logger.info(f"Cache cleared: {count} forecasts removed")
    return {
        "cleared": count, 
        "message": f"Cleared {count} cached forecasts",
    }


# Scheduled Forecasting (Commented out for now)
# To enable scheduled forecasting, uncomment the code below and add these imports:
#   from contextlib import asynccontextmanager
#   from apscheduler.schedulers.asyncio import AsyncIOScheduler
#   from apscheduler.triggers.cron import CronTrigger
#   from apscheduler.triggers.interval import IntervalTrigger
#   from .forecast_jobs import load_config, run_scheduled_forecasts
# 
# scheduler: Optional[AsyncIOScheduler] = None
# latest_scheduled_result: Optional[dict] = None
# latest_scheduled_time: Optional[datetime] = None
# 
# # Runs forecast job asynchronously.
# async def run_scheduled_job_async():
#     global latest_scheduled_result, latest_scheduled_time
#     
#     try:
#         logger.info("STARTING scheduled forecast job")
#         result = await asyncio.to_thread(run_scheduled_forecasts)
#         latest_scheduled_result = result
#         latest_scheduled_time = datetime.now()
#         logger.info("COMPLETED scheduled forecast job")
#     except Exception as e:
#         logger.error(f"FAILED scheduled forecast job: {e}")
#         latest_scheduled_result = {"error": str(e)}
#         latest_scheduled_time = datetime.now()
# 
# # Starts scheduler on startup, stops on shutdown.
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global scheduler
    
#     # Starts APScheduler
#     scheduler = AsyncIOScheduler()
#     scheduler.start()
#     logger.info("STARTED APScheduler")
    
#     # Schedules jobs from config
#     config = load_config()
    
#     # Checks for schedule_minutes (for testing) or schedule_hours (for production)
#     schedule_minutes = config.get("schedule_minutes")
#     schedule_hours = config.get("schedule_hours")
    
#     if schedule_minutes:
#         # Adds job to scheduler
#         scheduler.add_job(
#             run_forecast_job_async,
#             trigger=IntervalTrigger(minutes=schedule_minutes),
#             id="forecast_job",
#             replace_existing=True,
#         )
#         logger.info(f"SCHEDULED forecast job every {schedule_minutes} minutes")
#     elif schedule_hours:
#         # Support both single value (interval) and list (specific hours)
#         if isinstance(schedule_hours, int):
#             scheduler.add_job(
#                 run_forecast_job_async,
#                 trigger=IntervalTrigger(hours=schedule_hours),
#                 id="forecast_job",
#                 replace_existing=True,
#             )
#             logger.info(f"SCHEDULED forecast job every {schedule_hours} hours")
#         # If we want to schedule at specific hours daily, we can use CronTrigger
#         # elif isinstance(schedule_hours, list):
#         #     for hour in schedule_hours:
#         #         scheduler.add_job(
#         #             run_forecast_job_async,
#         #             trigger=CronTrigger(hour=hour, minute=0),
#         #             id=f"forecast_job_{hour}",
#         #             replace_existing=True,
#         #         )
#         #         logger.info(f"Scheduled forecast job for {hour}:00 daily")
    
#     yield
    
#     # Shuts down scheduler
#     if scheduler and scheduler.running:
#         scheduler.shutdown()
#         logger.info("STOPPED APScheduler if it was running")
# 
# 
# # Add lifespan to FastAPI app:
# # app = FastAPI(..., lifespan=lifespan)
#  
# # Gets the latest forecast results from last scheduled run.
# @app.get("/scheduled/latest")
# async def get_latest_scheduled() -> dict:
#     if latest_scheduled_result is None:
#         raise HTTPException(
#             status_code=404, 
#             detail="No scheduled forecast results available"
#         )
#     
#     return {
#         "result": latest_scheduled_result,
#         "generated_at": latest_scheduled_time.isoformat() if latest_scheduled_time else None,
#     }

# # Manually triggers a forecast job (for testing purposes only)
# @app.post("/forecasts/trigger")
# async def trigger_forecast_now() -> dict:
#     try:
#         await run_forecast_job_async()
#         return {
#             "status": "completed",
#             "message": "Forecast job completed",
#             "result_available": latest_forecast_result is not None,
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Gets current forecast configuration
# @app.get("/forecasts/config")
# async def get_forecast_config() -> dict:
#     try:
#         return load_config()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))