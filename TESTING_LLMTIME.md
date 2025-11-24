# Testing LLMTime Wrapper

This guide explains how to test the LLMTime wrapper in `sensbee_nvp` with multiple LLM providers using local JSON sensor data.

### Main Entry Point: Forecast from Local JSON

The primary way to generate forecasts is using the CLI with your local JSON data:

```bash
# Activate virtual environment
source venv/bin/activate

# Forecast temperature for next 24 hours (96 steps at 15-min intervals)
python -m src.models.llmtime_wrapper --horizon 24

# Use different provider
python -m src.models.llmtime_wrapper --horizon 24 --provider groq

# Use specific model
python -m src.models.llmtime_wrapper --horizon 24 --provider openai --model gpt-4o-mini

# Enable verbose logging
python -m src.models.llmtime_wrapper --horizon 24 --verbose
```

### Test with Real LLM APIs

The wrapper supports multiple providers. First, set up your API keys:

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

## FastAPI Service

The FastAPI service currently only has a health check endpoint:

```bash
# Start the server
uvicorn src.service.main:app --reload

# Check health
curl http://localhost:8000/health
```
