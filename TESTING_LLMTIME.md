# Testing LLMTime Wrapper

Note: Testing part doesn't include scheduled forecasting part yet. 

Test forecasting with SensBee API or local JSON data. It tests on default sensor which is Ilmenau's Manebach 1 weather station. Forecast temperature for next 24 hours (96 steps at 15-min intervals)

## Quick Test

```bash
source venv/bin/activate

# Test API source
python test_llmtime.py --source api

# Test local JSON
python test_llmtime.py --source local

# Test both
python test_llmtime.py --source both
```

## Full CLI Usage

```bash
# From SensBee API (uses defaults from sensbee_client.py)
python -m src.models.llmtime_wrapper --source api

# From local JSON (uses defaults from data_loader.py)
python -m src.models.llmtime_wrapper --source local

# Custom options
python -m src.models.llmtime_wrapper --source api \
  --sensor-id <UUID> \
  --api-key <KEY> \
  --column temperature \
  --horizon 24 \
  --provider mistral
```

## LLM Providers

Set API key in `.env`:

```bash
# Groq (default, free, fastest)
echo "GROQ_API_KEY=your-key" >> .env

# Mistral (free)
echo "MISTRAL_API_KEY=your-key" >> .env

# OpenAI (paid)
echo "OPENAI_API_KEY=your-key" >> .env
```

Then use `--provider groq|mistral|openai`.

## Default Parameters

Edit these files to change defaults:
- `src/data_access/sensbee_client.py` - API defaults (sensor, url, key)
- `src/data_access/data_loader.py` - Local JSON defaults (path, column)
