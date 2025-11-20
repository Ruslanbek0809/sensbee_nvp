# Testing LLMTime Wrapper

This guide explains how to test the LLMTime wrapper in `sensbee_nvp` with multiple LLM providers.

## 🚀 Quick Start

### 1. Test Individual Components (No API Key Required)

```bash
# Activate virtual environment
source venv/bin/activate

# Test encoding function
python test_llmtime.py --test-encode

# Test parsing function
python test_llmtime.py --test-parse

# Test full pipeline with mock (no API call)
python test_llmtime.py --test-mock

# Run all non-API tests
python test_llmtime.py
```

### 2. Test with Real LLM APIs

The wrapper supports multiple providers.

#### Option A: Mistral API (Free Tier)

```bash
# 1. Get free API key from https://console.mistral.ai/
# 2. Add to .env file
echo "MISTRAL_API_KEY=your-key-here" >> .env

# 3. Test with Mistral
python test_llmtime.py --test-full --provider mistral
```

#### Option B: Groq (Fastest + Free Tier)

```bash
# 1. Get free API key from https://console.groq.com/
# 2. Add to .env file
echo "GROQ_API_KEY=your-key-here" >> .env

# 3. Test with Groq
python test_llmtime.py --test-full --provider groq
```

#### Option C: OpenAI (Paid)

```bash
# 1. Get API key from https://platform.openai.com/
# 2. Add to .env file
echo "OPENAI_API_KEY=your-key-here" >> .env

# 3. Test with OpenAI
python test_llmtime.py --test-full --provider openai
```


## 🔧 Auto-Detection

The wrapper automatically detects available providers:
- If `MISTRAL_API_KEY` is set → uses Mistral (default)
- If `GROQ_API_KEY` is set → can use Groq
- If `OPENAI_API_KEY` is set → can use OpenAI

Set `LLM_PROVIDER` in `.env` to override:
```bash
LLM_PROVIDER=mistral  # or 'groq' or 'openai'
```

## 📝 Environment Variables

Add to `.env` file:
```bash
# Choose your provider (mistral, groq, or openai)
LLM_PROVIDER=mistral

# Mistral (recommended - free tier)
MISTRAL_API_KEY=your-mistral-key

# Groq (fastest - free tier)
GROQ_API_KEY=your-groq-key

# OpenAI (paid)
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4o-mini
```

## Manual Testing in Python

### Test Encoding

```python
from src.models.llmtime_wrapper import encode_series_to_prompt
import pandas as pd

series = pd.Series([20.1, 20.3, 20.7, 21.0, 21.2])
prompt = encode_series_to_prompt(series, horizon=3)
print(prompt)
```

### Test Parsing

```python
from src.models.llmtime_wrapper import parse_forecast_from_text

# Test various response formats
responses = [
    "21.5, 21.8, 22.0",
    "The next values are: 21.5, 21.8, 22.0",
    "21.5\n21.8\n22.0"
]

for resp in responses:
    forecast = parse_forecast_from_text(resp, horizon=3)
    print(f"Response: {resp}")
    print(f"Parsed: {forecast}\n")
```

### Test Full Pipeline (Mock)

```python
from src.models.llmtime_wrapper import encode_series_to_prompt, parse_forecast_from_text
import pandas as pd

# Create series
series = pd.Series([20.1, 20.3, 20.7, 21.0, 21.2, 21.5])
horizon = 3

# Step 1: Encode
prompt = encode_series_to_prompt(series, horizon)
print(f"Prompt: {prompt}\n")

# Step 2: Simulate API response
mock_response = "21.8, 22.0, 22.2"

# Step 3: Parse
forecast = parse_forecast_from_text(mock_response, horizon)
print(f"Forecast: {forecast}")
```

### Test Full Pipeline (Real API - Mistral)

```python
from src.models.llmtime_wrapper import llmtime_forecast
import pandas as pd

# Create test series
series = pd.Series([20.1, 20.3, 20.7, 21.0, 21.2, 21.5, 21.7])
horizon = 3

# Generate forecast with Mistral (default)
forecast = llmtime_forecast(series, horizon, provider='mistral')
print(f"Forecast: {forecast}")

# Or with Groq
forecast = llmtime_forecast(series, horizon, provider='groq')
print(f"Forecast: {forecast}")
```

## Testing with SensBee Data

### Test with Real Sensor Data

```python
from src.data_access.sensbee_client import SensBeeClient
from src.models.llmtime_wrapper import llmtime_forecast

# Initialize client
client = SensBeeClient()

# Get sensor data (use example from supervisor)
sensor_id = "8d790e21-f948-4e75-9c47-ea8b1aa75e9d"
api_key = "f9dc9952-bc02-4f76-841a-a20035e6f574"
column = "temperature"  # or whatever column exists

# Fetch series
series = client.get_series(sensor_id, api_key, column, limit=50)

# Generate forecast with Mistral
forecast = llmtime_forecast(series, horizon=24, provider='mistral')
print(f"Forecast for next 24 steps: {forecast}")
```

## Testing via FastAPI Endpoint

You can also test via the FastAPI service (after adding the endpoint):

```bash
# Start the server
uvicorn src.service.main:app --reload

# In another terminal, test with curl
curl -X POST "http://localhost:8000/predict/llmtime" \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "8d790e21-f948-4e75-9c47-ea8b1aa75e9d",
    "api_key": "f9dc9952-bc02-4f76-841a-a20035e6f574",
    "column": "temperature",
    "horizon": 24
  }'
```
