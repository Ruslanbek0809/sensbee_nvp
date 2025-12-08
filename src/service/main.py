# FastAPI service for next-value prediction of smart city sensor data with scheduled forecasting.

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.asyncio import AsyncIOScheduler
# from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .forecast_jobs import load_config, run_scheduled_forecasts

logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler: Optional[AsyncIOScheduler] = None

# Store latest forecast results in memory
latest_forecast_result: Optional[dict] = None
latest_forecast_time: Optional[datetime] = None


# Runs forecast job asynchronously.
async def run_forecast_job_async():
    global latest_forecast_result, latest_forecast_time
    
    try:
        logger.info("STARTING scheduled forecast job")
        result = await asyncio.to_thread(run_scheduled_forecasts)
        latest_forecast_result = result
        latest_forecast_time = datetime.now()
        logger.info("COMPLETED scheduled forecast job")
    except Exception as e:
        logger.error(f"FAILED scheduled forecast job: {e}")
        latest_forecast_result = {"error": str(e)}
        latest_forecast_time = datetime.now()


# Starts scheduler on startup, and stops on shutdown.
@asynccontextmanager
async def lifespan():
    global scheduler
    
    # Starts APScheduler
    scheduler = AsyncIOScheduler()
    scheduler.start()
    logger.info("STARTED APScheduler")
    
    # Schedules jobs from config
    config = load_config()
    
    # Checks for schedule_minutes (for testing) or schedule_hours (for production)
    schedule_minutes = config.get("schedule_minutes")
    schedule_hours = config.get("schedule_hours")
    
    if schedule_minutes:
        # Adds job to scheduler
        scheduler.add_job(
            run_forecast_job_async,
            trigger=IntervalTrigger(minutes=schedule_minutes),
            id="forecast_job",
            replace_existing=True,
        )
        logger.info(f"SCHEDULED forecast job every {schedule_minutes} minutes")
    elif schedule_hours:
        # Support both single value (interval) and list (specific hours)
        if isinstance(schedule_hours, int):
            scheduler.add_job(
                run_forecast_job_async,
                trigger=IntervalTrigger(hours=schedule_hours),
                id="forecast_job",
                replace_existing=True,
            )
            logger.info(f"SCHEDULED forecast job every {schedule_hours} hours")
        # If we want to schedule at specific hours daily, we can use CronTrigger
        # elif isinstance(schedule_hours, list):
        #     for hour in schedule_hours:
        #         scheduler.add_job(
        #             run_forecast_job_async,
        #             trigger=CronTrigger(hour=hour, minute=0),
        #             id=f"forecast_job_{hour}",
        #             replace_existing=True,
        #         )
        #         logger.info(f"Scheduled forecast job for {hour}:00 daily")
    
    yield
    
    # Shuts down scheduler
    if scheduler and scheduler.running:
        scheduler.shutdown()
        logger.info("STOPPED APScheduler if it was running")


app = FastAPI(
    title="SensBee NVP Service",
    description="Next-value prediction service for smart city sensor data using LLM forecasting with scheduled forecasting",
    version="0.2.0",
    lifespan=lifespan,
)

# Fast way to verify the service is up without triggering the logic
@app.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "scheduler": "running" if scheduler and scheduler.running else "stopped"
    }


# Gets the latest forecast results from last scheduled run.
@app.get("/forecasts/latest")
async def get_latest_forecasts() -> dict:
    if latest_forecast_result is None:
        raise HTTPException(status_code=404, detail="NO forecast results available yet")
    
    return {
        "result": latest_forecast_result,
        "generated_at": latest_forecast_time.isoformat() if latest_forecast_time else None,
    }


# Manually triggers a forecast job (for testing purposes only)
@app.post("/forecasts/trigger")
async def trigger_forecast_now() -> dict:
    try:
        await run_forecast_job_async()
        return {
            "status": "completed",
            "message": "Forecast job completed",
            "result_available": latest_forecast_result is not None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Gets current forecast configuration
@app.get("/forecasts/config")
async def get_forecast_config() -> dict:
    try:
        return load_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
