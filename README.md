# SensBee NVP - Next-Value Prediction Service

On-demand LLM-based time series forecasting service for SensBee sensor data.

This service provides next-value predictions for sensor data using LLMs. Users can request forecasts via REST API, specifying the sensor, column, and forecast horizon.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add LLM API key to `.env`:
```bash
# Mistral (free, recommended)
echo "MISTRAL_API_KEY=your-key" >> .env

# Or Groq (free, fastest)
echo "GROQ_API_KEY=your-key" >> .env

# Or OpenAI (paid)
echo "OPENAI_API_KEY=your-key" >> .env
```

## Quick Start

### Start the Service

```bash
cd sensbee_nvp
source venv/bin/activate
uvicorn src.service.main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Test with Swagger UI (Recommended)

1. Open browser: http://localhost:8000/docs
2. Click **POST /forecast** → **Try it out**
3. Enter request body and click **Execute**

### Test with curl

```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "8d790e21-f948-4e75-9c47-ea8b1aa75e9d",
    "api_key": "6d5ecc8d-1e5a-4c66-be1d-c6fd34958777",
    "column": "temperature",
    "horizon_hours": 6,
    "history_days": 3,
    "provider": "mistral"
  }'
```

## API Endpoints


`/forecast` => POST method that Generates on-demand forecast
`/forecast/cache/{sensor_id}/{column}` => GET method that fetches cached forecast
`/forecast/cache` =>  GET method that lists all cached forecast
`/forecast/cache` => DELETE method that clears cache
`/health` => GET method that checks health of the server and displays cache size

## Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sensor_id` => SensBee sensor UUID (required. Type: string)
| `api_key` => READ API key for private sensors (optional. Type: string)
| `column` => Column to forecast (e.g., "temperature") (required. Type: string)
| `horizon_hours` => Forecast horizon (6-168 hours) (required. Type: int)
| `history_days` => Days of history to use (1-14 days) (required. Type: int)
| `provider` => LLM provider: mistral, groq, openai (required. Type: string)

## Project Structure

```
sensbee_nvp/
├── src/
│   ├── data_access/    # SensBee API client
│   ├── models/         # LLMTime forecasting wrapper
│   └── service/        # FastAPI service (main.py)
├── config/             # Configuration files
├── data/               # Local test data
├── requirements.txt
└── README.md
```

## Available Weather Columns

For Ilmenau weather stations:
- `temperature` - Temperature in °C
- `humidity` - Relative humidity in %
- `wind_speed` - Wind speed in m/s
- `precipitation` - Precipitation in Liter/m²
