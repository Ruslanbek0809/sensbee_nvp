# SensBee NVP - Next-Value Prediction Service

Next-value prediction service for smart city sensor data.

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

## Running the Service

Activate the virtual environment first, then run:

```bash
uvicorn src.service.main:app --reload
```

The service will be available at:
- API: http://localhost:8000
- Health endpoint: http://localhost:8000/health
- API documentation: http://localhost:8000/docs

## Project Structure

```
sensbee_nvp/
├── src/
│   ├── data_access/    # SensBee data loading (local JSON)
│   ├── models/         # LLMTime forecasting wrapper
│   └── service/        # FastAPI REST service
└── requirements.txt
```

