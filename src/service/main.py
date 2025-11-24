# FastAPI service for next-value prediction of smart city sensor data.

from fastapi import FastAPI

app = FastAPI(
    title="SensBee NVP Service",
    description="Next-value prediction service for smart city sensor data using LLM forecasting",
    version="0.1.0",
)


# Fast way to verify the service is up without triggering business logic
@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}

