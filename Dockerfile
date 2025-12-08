# Dockerfile for SensBee NVP service

FROM python:3.11-slim

WORKDIR /app

# Installs system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copies requirements first and installs Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies application code
COPY . .

# Creates non-privileged user to run the application
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Exposes FastAPI port
EXPOSE 8000

# Starts FastAPI service
CMD ["uvicorn", "src.service.main:app", "--host", "0.0.0.0", "--port", "8000"]


