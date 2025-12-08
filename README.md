# SensBee NVP - Next-Value Prediction Service

LLM-based time series forecasting for SensBee sensor data.

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
   
## Quick Start

```bash
# From SensBee API (testing)
python -m src.models.llmtime_wrapper --source api --horizon 24

# From local JSON file (testing)
python -m src.models.llmtime_wrapper --source local --horizon 24
```

## Configuration

Add API keys to `.env`:
```bash
MISTRAL_API_KEY=your-key   # Free: https://console.mistral.ai/
# or
GROQ_API_KEY=your-key      # Free: https://console.groq.com/
# or 
OPENAI_API_KEY=your-key    # Paid: https://platform.openai.com/
```

## CLI Options

```
--source {local,api}   Data source (default: api)
--sensor-id UUID       Sensor UUID for API source
--api-key KEY          READ API key for API source
--column NAME          Column to forecast (default: temperature)
--horizon HOURS        Forecast horizon (default: 24)
--provider NAME        LLM provider: mistral, groq, openai
--verbose              Enable debug logging
```

## Project Structure

```
sensbee_nvp/
├── src/
│   ├── config/         # Forecast configurations
│   ├── data_access/    # SensBee API client + local JSON loader
│   ├── models/         # LLMTime forecasting wrapper
│   └── service/        # FastAPI service (WIP)
├── data/               # Local test data
└── requirements.txt
```
