# Open-sourced recently
# SensBee NVP (next-value prediction) service

On-demand LLM-based time series forecasting service for SensBee sensor data.

## Overview

This service provides zero-shot time series forecasting using LLMs. It serializes sensor readings into comma-separated strings and prompt LLMs to continue the sequence.

### Key Features

- **Zero-shot forecasting**: No training required, meaning it works on any numeric sensor
- **Multiple LLM providers**: Groq (Llama-3.3-70B), OpenAI (GPT-3.5-turbo-instruct), Mistral API, and local models (Llama-2-7B, Mistral-7B)
- **REST API**: FastAPI microservice with Swagger UI documentation
- **Configurable parameters**: Normalization, statistical context, temperature, number of forecasts
- **Smart filtering**: Four-rule filter rejects invalid forecasts before aggregation

## Forecasting Pipeline

The pipeline follows six stages:

1. **Fetch history.** It retrieves sensor data from SensBee API with time grouping.
2. **Resample.** It resamples to 15-minute grid using forward-fill technique for regular sensors, and dual-fill technique for event-based sensors.
3. **Normalize.** It uses LLMTime-style quantile scaler (α=0.95, β=0.3) shifts minimum and scales so 95th percentile equals to 1.0.
4. **Serialize.** It rounds to 3 decimal place and joins with commas. Only exception is GPT models which needs digit spacing.
5. **Prompt LLM.** It generate N independent forecasts (default N=5) at temperature 0.9.
6. **Parse, filter &amp; aggregate.** It parses the generated data, applies four filter rules, computes median, and inverses transform.

### Filter Rules

Forecasts are discarded if they show following trends:
- (a) Out-of-bounds values
- (b) Constant or near-constant output (≤2 unique values)
- (c) Flat output (std < 0.001)
- (d) Perfectly linear trends (second derivative < 0.005)

If all forecasts are filtered, the system falls back to the last observed value (naive baseline).

## Setup

### Option A: Virtual Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Add LLM API key to .env
echo "GROQ_API_KEY=your-key" >> .env
# Or: echo "OPENAI_API_KEY=your-key" >> .env
# Or: echo "MISTRAL_API_KEY=your-key" >> .env
```

### Option B: Docker

```bash
# Build and start
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

For local inference with token control:

#### NVIDIA GPUs (CUDA):
```bash
pip install torch transformers accelerate bitsandbytes
echo "LOCAL_MODEL=llama2-7b" >> .env
# Or: echo "LOCAL_MODEL=mistral-7b" >> .env
```

#### Apple Silicon (MPS):
```bash
pip install torch transformers accelerate
echo "LOCAL_MODEL=mistral-7b" >> .env
```

### Option D: HPC Cluster (TU Ilmenau)

See `hpc_setup/README_CLUSTER.md` for deployment instructions on the TU Ilmenau HPC cluster with A100 GPUs.

## Quick Start

```bash
source venv/bin/activate
uvicorn src.service.main:app --reload --port 8000
```

Service runs at: http://localhost:8000

## API Endpoints

`/forecast` => POST method that Generates on-demand forecast
`/forecast/cache/{sensor_id}/{column}` => GET method that fetches cached forecast
`/forecast/cache` =>  GET method that lists all cached forecast
`/forecast/cache` => DELETE method that clears cache
`/health` => GET method that checks health of the server and displays cache size

### Request Parameters

`sensor_id` => SensBee sensor UUID (required. Type: string)
`api_key` => READ API key for private sensors (required. Type: string)
`column` => Column to forecast (e.g., "temperature") (required. Type: string)
`horizon_hours` => Forecast horizon (6-168 hours) (required. Type: int. Default is 24)
`history_days` => Days of history to use (1-14 days) (required. Type: int. Default 7)
`provider` => LLM provider: mistral, groq, openai (required. Type: string. Default is groq)
`num_samples` => Number of samples (1-20). Higher = better (required. Type: int. Default is 5)
`temperature` => LLM temperature (0.1-1.5). Higher = diverse (required. Type: float. Default is 0.9)
`use_normalization` => Applies LLMTime-style quantile scaling (optional. Type: bool. Default is true)
`include_context` => Prepends statistical context such as, min, max, trend (optional. Type: bool. Default is false)

### Example Request

```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "YOUR_SENSOR_UUID",
    "api_key": "YOUR_SENSBEE_READ_API_KEY",
    "column": "temperature",
    "horizon_hours": 24,
    "history_days": 7,
    "provider": "groq",
    "num_forecasts": 5,
    "temperature": 0.9
  }'
```

## Models Tested

| Model | Type | Access | Context Length |
|-------|------|--------|----------------|
| Groq Llama-3.3-70B | Instruct | API (Groq) | 128K tokens |
| GPT-3.5-turbo-instruct | Instruct | API (OpenAI) | 4K tokens |
| Llama-2-7B | Base | Local (HPC) | 4K tokens |
| Mistral-7B-v0.1 | Base | Local (HPC) | 8K tokens |

## Benchmark Results

Based on the ablation study with 192 tests (8 configurations × 3 windows × 2 sensors × 4 models):

### Temperature Forecasting (24-hour horizon)

| Model | Best MAE | Window | vs. Naive |
|-------|----------|--------|-----------|
| Llama-2-7B | **0.81°C** | 7-day | +27.3% |
| Mistral-7B | 0.87°C | 14-day | +21.7% |
| Groq 70B | 1.08°C | 1-day | +3.1% |
| GPT-3.5-instruct | 3.32°C | 1-day | −198% |

Naive baseline: MAE = 1.115°C

### Key Findings

1. **Local 7B base models achieved the lowest single-test MAE.** However, it displayed high variance across configurations
2. **Aggressive instruction tuning destroys numeric continuation.** GPT-3.5-instruct failed all 24 tests (WBA-10 = 0%)
3. **Zero-shot visitor count forecasting is unsolved.** The reason is binary day/night pattern not getting captured.

### Design Rules from ablation study

- **Raw values + statistical context** produced the lowest mean MAE (1.07°C)
- **Also, use ≥5 independent forecasts** — 3 forecasts caused frequent fallback to naive baseline

## Project Structure

```
sensbee_nvp/
├── src/
│   ├── data_access/         # SensBee API client and data loader
│   │   ├── __init__.py
│   │   ├── sensbee_client.py
│   │   └── data_loader.py
│   ├── models/              # Forecasting models
│   │   ├── __init__.py
│   │   ├── nvp_llms.py      # Main forecasting logic
│   │   ├── serialize.py     # LLMTime-style serialization
│   │   └── local_llm.py     # Local LLM with token control
│   └── service/             # FastAPI service
│       ├── main.py          # REST API endpoints
│       └── forecast_jobs.py # Forecast execution
├── hpc_setup/               # HPC cluster deployment
│   ├── run_benchmark.py     # Comprehensive benchmark script
│   ├── setup_cluster.sh     # Cluster setup script
│   └── README_CLUSTER.md    # HPC deployment guide
├── data/                    # Local test data (JSON)
├── results/                 # Benchmark outputs (figures, tables)
├── test_llmtime.py          # Quick test script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## References

- Gruver et al., "Large Language Models are Zero-Shot Time Series Forecasters," NeurIPS 2023
- Jin et al., "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models," ICLR 2024
- Li et al., "UrbanGPT: Spatio-Temporal Large Language Models," KDD 2024

## License

Research project — TU Ilmenau, DBIS Chair
