
# FastAPI service for on-demand next-value prediction of smart city sensor data. 

import asyncio
import hashlib
import hmac
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from importlib.util import find_spec
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ..models.local_llm import LOCAL_MODELS
from ..models.nvp_llms import DEFAULT_MODELS, configured_model
from ..models.tsfm_forecaster import preload
from .forecast_jobs import CHRONOS2_HISTORY_DAYS, CHRONOS2_PROVIDER, forecast_single_column

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request body of the forecast endpoint.
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
        description="Column name to forecast (e.g., 'temperature', 'visitors_total')"
    )
    horizon_hours: int = Field(
        default=24, 
        ge=6,
        le=168,
        description="Forecast horizon in hours (6-168)"
    )
    history_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=CHRONOS2_HISTORY_DAYS,
        description="Days of history to use. LLM providers: 1-14, default 7. 'chronos-2': 1-28, default 28"
    )
    provider: str = Field(
        default="groq",
        description="LLM provider: 'groq', 'openai', 'mistral' (only if the server enables them), or 'local' "
                    "(GPU required). 'chronos-2' uses the Chronos-2 time-series model and also returns quantiles"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model name (optional). For local: one of its presets, e.g. 'mistral-7b' or 'llama2-7b'. "
                    "Hosted providers accept only the server's configured model; 'chronos-2' takes none"
    )
    num_forecasts: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of independent forecasts to generate (1-20). Median is returned."
    )
    temperature: float = Field(
        default=0.9,
        ge=0.1,
        le=1.5,
        description="LLM sampling temperature (0.1-1.5)."
    )
    use_normalization: bool = Field(
        default=True,
        description="Applies LLMTime-style quantile scaling (α=0.95, β=0.3)."
    )
    include_context: bool = Field(
        default=False,
        description="Prepends statistical context (min, max, median, trend) to the prompt. Can improve accuracy with raw values"
    )

# Response body of the forecast endpoint.
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
    use_normalization: bool = True
    include_context: bool = False
    cached: bool = False
    # Only for provider "chronos-2" (None otherwise): the 80 % interval (q0.1 and q0.9), every decile by its level
    # ("0.1" to "0.9"), and the model, recipe and inputs behind the forecast.
    forecast_lower: Optional[list[float]] = None
    forecast_upper: Optional[list[float]] = None
    forecast_quantiles: Optional[dict[str, list[float]]] = None
    model_info: Optional[dict] = None


# In-memory cache of forecast responses by sensor and column. Each entry also holds the digest of the READ key the
# forecast was made with, and the cache endpoints serve it only to a caller who sends the same key (X-API-Key header).
forecast_cache: dict[str, tuple[str, ForecastResponse]] = {}


def build_cache_key(sensor_id: str, column: str) -> str:
    return f"{sensor_id}_{column}"


# SHA-256 digest of a READ key, "" standing for no key. The cache keeps only the digest, never the key.
def key_digest(api_key: Optional[str]) -> str:
    return hashlib.sha256((api_key or "").strip().encode()).hexdigest()


# Whether a cache entry was made with the caller's key (compared in constant time).
def same_key(entry: tuple[str, ForecastResponse], api_key: Optional[str]) -> bool:
    return hmac.compare_digest(entry[0], key_digest(api_key))


HOSTED_PROVIDERS = ("groq", "openai", "mistral")  # called with the server's paid keys


# The hosted providers this server may call, from the comma-separated HOSTED_PROVIDERS variable. Unset: none.
def enabled_hosted_providers() -> set[str]:
    return {name.strip().lower() for name in os.getenv("HOSTED_PROVIDERS", "").split(",") if name.strip()}


# Checks the provider and model before any data is loaded or any model is called, and answers 400 otherwise.
# - A hosted provider (groq, openai, mistral) must be enabled in HOSTED_PROVIDERS, and takes only its configured
#   model, so a caller can't pick what the server's paid keys are spent on.
# - "local" takes its presets or its configured model. Any other name would be downloaded from the Hugging Face Hub
#   and loaded with trust_remote_code.
# - "chronos-2" takes no model name.
def check_provider_and_model(provider: str, model: Optional[str]) -> None:
    if provider == CHRONOS2_PROVIDER:
        allowed = set()
    elif provider == "local":
        allowed = set(LOCAL_MODELS) | {configured_model("local")}
    elif provider in HOSTED_PROVIDERS:
        if provider not in enabled_hosted_providers():
            raise HTTPException(status_code=400, detail=f"PROVIDER NOT ENABLED on this server: {provider}. "
                                                        f"The operator lists allowed hosted providers in HOSTED_PROVIDERS")
        allowed = {configured_model(provider)}
    else:
        raise HTTPException(status_code=400, detail=f"UNKNOWN PROVIDER: {provider}. "
                                                    f"SUPPORTED: {', '.join([*DEFAULT_MODELS, CHRONOS2_PROVIDER])}")
    if model is not None and model not in allowed:
        raise HTTPException(status_code=400, detail=f"MODEL NOT ALLOWED for provider {provider}: {model}. "
                                                    f"ALLOWED: {', '.join(sorted(allowed)) or 'none (leave model empty)'}")


# Loads Chronos-2 before the service accepts requests when chronos-forecasting is installed (the WITH_TSFM build), so
# the first chronos-2 request doesn't wait for it. Without it the service starts as before.
@asynccontextmanager
async def lifespan(app: FastAPI):
    if find_spec("chronos") is not None:
        await asyncio.to_thread(preload)
        logger.info("PRELOADED Chronos-2 at startup")
    yield


app = FastAPI(
    title="SensBee NVP Service",
    description=("On-demand next-value prediction service for smart city sensor data. It uses LLM-based forecasting to predict future sensor values."),
    version="0.2.0",
    lifespan=lifespan,
)

### API Endpoints

# Checks that the service is up without running a forecast. Also returns the cache size.
@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "cache_size": len(forecast_cache),
    }

# Runs a forecast on demand for the sensor, column and horizon in the request.
@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest) -> ForecastResponse:
    logger.info(f"/forecast Forecast POST request: sensor={request.sensor_id}, column={request.column}, horizon={request.horizon_hours}h, history={request.history_days}d, provider={request.provider}")
    provider = request.provider.lower()
    check_provider_and_model(provider, request.model)

    try:
        # Run the forecast in a worker thread so it does not block the event loop.
        result = await asyncio.to_thread(
            forecast_single_column,
            sensor_id=request.sensor_id,
            api_key=request.api_key,
            column_name=request.column,
            horizon_hours=request.horizon_hours,
            history_days=request.history_days,
            provider=provider,
            model=request.model,
            num_forecasts=request.num_forecasts,
            temperature=request.temperature,
            use_normalization=request.use_normalization,
            include_context=request.include_context,
        )
        
        # Build the response
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
            use_normalization=request.use_normalization,
            include_context=request.include_context,
            cached=False,
            forecast_lower=result.get("forecast_lower"),
            forecast_upper=result.get("forecast_upper"),
            forecast_quantiles=result.get("forecast_quantiles"),
            model_info=result.get("model_info"),
        )
        
        # Cache the response
        cache_key = build_cache_key(request.sensor_id, request.column)
        forecast_cache[cache_key] = (key_digest(request.api_key), response)
        logger.info(f"/forecast CACHED forecast: {cache_key}")
        
        return response
        
    except ValueError as e:
        logger.error(f"/forecast INVALID VALUE error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    # SensBee errors carry the request URL, and with it the READ key: only the status or error type goes out.
    except httpx.HTTPStatusError as e:
        logger.error(f"/forecast UPSTREAM FAILED: HTTP {e.response.status_code}")
        raise HTTPException(status_code=502, detail=f"UPSTREAM REQUEST FAILED: HTTP {e.response.status_code}")
    except httpx.HTTPError as e:
        logger.error(f"/forecast UPSTREAM FAILED: {type(e).__name__}")
        raise HTTPException(status_code=502, detail=f"UPSTREAM REQUEST FAILED: {type(e).__name__}")
    except Exception as e:
        logger.error(f"/forecast Forecast FAILED: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"FORECAST FAILED ({type(e).__name__})")


# Returns the cached forecast for the sensor and column. Answers 404 if there is none or it was made with another key,
# so a caller without the key can't tell the two apart.
@app.get("/forecast/cache/{sensor_id}/{column}", response_model=ForecastResponse)
async def get_cached_forecast(sensor_id: str, column: str, x_api_key: Optional[str] = Header(default=None)) -> ForecastResponse:
    entry = forecast_cache.get(build_cache_key(sensor_id, column))

    if entry is None or not same_key(entry, x_api_key):
        raise HTTPException(
            status_code=404, 
            detail=f"NO CACHED forecast for sensor={sensor_id}, column={column}"
        )
    
    response = entry[1]
    response.cached = True
    return response


# Lists the cached forecasts made with the caller's key (without a key: those made without one), with their metadata.
@app.get("/forecast/cache")
async def list_cached_forecasts(x_api_key: Optional[str] = Header(default=None)) -> dict:
    items = [entry[1] for entry in forecast_cache.values() if same_key(entry, x_api_key)]
    return {
        "count": len(items),
        "forecasts": [
            {
                "sensor_id": item.sensor_id,
                "column": item.column,
                "generated_at": item.generated_at,
                "forecast_points": item.forecast_points,
            }
            for item in items
        ],
    }


# Clears the cached forecasts made with the caller's key (without a key: those made without one).
@app.delete("/forecast/cache")
async def clear_cache(x_api_key: Optional[str] = Header(default=None)) -> dict:
    mine = [cache_key for cache_key, entry in forecast_cache.items() if same_key(entry, x_api_key)]
    for cache_key in mine:
        del forecast_cache[cache_key]
    count = len(mine)
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