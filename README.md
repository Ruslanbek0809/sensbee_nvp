# SensBee NVP - Next-Value Prediction Service

On-demand LLM-based time series forecasting service for SensBee sensor data.

This service provides next-value predictions for sensor data using LLMs. Users can request forecasts via REST API, specifying the sensor, column, and forecast horizon.

## Setup

### Option A: Virtual Environment (Used in Development right now)

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add LLM API key to .env
echo "MISTRAL_API_KEY=your-key" >> .env
# Or: echo "GROQ_API_KEY=your-key" >> .env
# Or: echo "OPENAI_API_KEY=your-key" >> .env
```

### Option B: Docker (Development & Production)

```bash
# Stop any existing container first
docker compose down

# Build and start
docker compose up -d --build

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Option C: Local LLM (GPU Required)

For best forecast quality, use local LLMs with token control:

#### For NVIDIA GPUs (CUDA):
```bash
# Install dependencies (includes bitsandbytes for quantization)
pip install torch transformers accelerate bitsandbytes

# Set LOCAL_MODEL in .env
echo "LOCAL_MODEL=llama2-7b" >> .env
```

#### For Apple Silicon (Mac M1/M2/M3):
```bash
# Install dependencies (no bitsandbytes needed - not supported on Mac)
pip install torch transformers accelerate

# Set LOCAL_MODEL in .env (Mistral works best on Mac)
echo "LOCAL_MODEL=mistral-7b" >> .env
# Or: echo "LOCAL_MODEL=llama2-7b" >> .env
```


## Quick Start

### Start with Virtual Environment

```bash
cd sensbee_nvp
source venv/bin/activate
uvicorn src.service.main:app --reload --port 8000
```

### Start with Docker

```bash
docker compose up -d
```

Service runs at: http://localhost:8000

## Testing

### Test with Swagger UI

1. Open browser: http://localhost:8000/docs
2. Click **POST /forecast** → **Try it out**
3. Enter request body and click **Execute**

### Test with curl (API-based)

```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "8d790e21-f948-4e75-9c47-ea8b1aa75e9d",
    "api_key": "6d5ecc8d-1e5a-4c66-be1d-c6fd34958777",
    "column": "temperature",
    "horizon_hours": 6,
    "history_days": 3,
    "provider": "groq",
    "num_samples": 5,
    "temperature": 0.9
  }'
```


## API Endpoints

`/forecast` => POST method that Generates on-demand forecast
`/forecast/cache/{sensor_id}/{column}` => GET method that fetches cached forecast
`/forecast/cache` =>  GET method that lists all cached forecast
`/forecast/cache` => DELETE method that clears cache
`/health` => GET method that checks health of the server and displays cache size

## Request Parameters

`sensor_id` => SensBee sensor UUID (required. Type: string)
`api_key` => READ API key for private sensors (optional. Type: string)
`column` => Column to forecast (e.g., "temperature") (required. Type: string)
`horizon_hours` => Forecast horizon (6-168 hours) (required. Type: int)
`history_days` => Days of history to use (1-14 days) (required. Type: int)
`provider` => LLM provider: mistral, groq, openai (required. Type: string)
`num_samples` => Number of samples (1-10). Higher = better (required. Type: int)
`temperature` => LLM temperature (0.1-1.5). Higher = diverse (required. Type: float)


## Project Structure

```
sensbee_nvp/
├── src/
│   ├── data_access/    # SensBee API client
│   ├── models/         # NVP LLMs + local LLM support
│   │   ├── nvp_llms.py  # Main forecasting logic
│   │   └── local_llm.py        # Local LLM with token control
│   └── service/        # FastAPI service (main.py)
├── config/             # Configuration files
├── data/               # Local test data
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```
