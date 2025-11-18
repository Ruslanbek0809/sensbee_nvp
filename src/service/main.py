# FastAPI service for next-value prediction of smart city sensor data.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..data_access.sensbee_client import SensBeeClient
from ..models.baseline import naive_last_value_forecast

app = FastAPI(
    title="SensBee NVP Service",
    description="Next-value prediction service for smart city sensor data",
    version="0.1.0",
)

# Initialize SensBee client (singleton)
sensbee_client = SensBeeClient()


# Request model for baseline forecasting endpoint.
class BaselineForecastRequest(BaseModel):

    sensor_id: str = Field(..., description="UUID of the sensor")
    api_key: str = Field(..., description="API key for authentication")
    column: str = Field(..., description="Column name to forecast")
    horizon: int = Field(..., ge=1, description="Number of steps to forecast ahead")

# Response model for baseline forecasting endpoint.
class BaselineForecastResponse(BaseModel):

    sensor_id: str
    column: str
    history: list[float]
    forecast: list[float]
    horizon: int

# Fast way to verify the service is up without triggering business logic
@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/baseline", response_model=BaselineForecastResponse)
async def baseline_forecast(request: BaselineForecastRequest):
    """
    Generate a baseline forecast using naive last-value method.
    
    Retrieves the last 168 data points from the sensor and applies
    naive_last_value_forecast to generate predictions.
    
    Args:
        request: Forecast request containing sensor_id, api_key, column, and horizon
        
    Returns:
        BaselineForecastResponse with history and forecast values
        
    Raises:
        HTTPException: 400 if input validation fails or data cannot be retrieved
    """
    try:
        # Retrieve time series data (last 168 points)
        series = sensbee_client.get_series(
            sensor_id=request.sensor_id,
            api_key=request.api_key,
            column_name=request.column,
            limit=168,
        )

        # Validate series is not empty
        if len(series) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No data available for sensor {request.sensor_id}",
            )

        # Generate forecast using naive last value method
        forecast = naive_last_value_forecast(series, request.horizon)

        # Convert to lists for JSON serialization
        history = series.tolist()
        forecast_list = forecast.tolist()

        return BaselineForecastResponse(
            sensor_id=request.sensor_id,
            column=request.column,
            history=history,
            forecast=forecast_list,
            horizon=request.horizon,
        )

    except KeyError as e:
        # Column not found
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except ValueError as e:
        # Empty series or other validation error
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except Exception as e:
        # Other errors (e.g., API connection issues)
        raise HTTPException(
            status_code=400,
            detail=f"Error retrieving data: {str(e)}",
        )

