# Dockerfile for SensBee NVP service
#
# The default build has the LLM providers only. With --build-arg WITH_TSFM=1 it also installs CPU torch and
# chronos-forecasting (requirements-tsfm.txt) and downloads the Chronos-2 weights at the pinned revision, so the
# container runs offline (HF_HUB_OFFLINE=1). The default build is as before.
# CHRONOS2_REVISION must equal CHRONOS2 in benchmark/models/__init__.py (tests/service/test_packaging.py checks it).
# The weights are downloaded before the code is copied, so a code change doesn't download them again. Each build arg is
# declared just before the first layer that uses it, so changing it (or requirements-tsfm.txt) rebuilds only the layers
# from there on: a new revision downloads only the weights again.

FROM python:3.13-slim

WORKDIR /app

# Installs system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Creates non-privileged user to run the application. It owns the model cache.
RUN useradd -m -u 1000 appuser && mkdir -p /app/hf-cache && chown appuser:appuser /app /app/hf-cache
ENV HF_HOME=/app/hf-cache

# Copies requirements first and installs Python packages, with the versions in constraints.txt
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt
ARG WITH_TSFM=0
COPY requirements-tsfm.txt ./
RUN if [ "$WITH_TSFM" = "1" ]; then pip install --no-cache-dir -r requirements-tsfm.txt; fi

# Downloads the Chronos-2 weights (TSFM build only), then turns the Hugging Face Hub offline (TSFM build only:
# HF_HUB_OFFLINE=0 is off)
USER appuser
ARG CHRONOS2_MODEL=amazon/chronos-2
ARG CHRONOS2_REVISION=29ec3766d36d6f73f0696f85560a422f50e8498c
RUN if [ "$WITH_TSFM" = "1" ]; then \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('$CHRONOS2_MODEL', revision='$CHRONOS2_REVISION')"; \
    fi
ENV HF_HUB_OFFLINE=$WITH_TSFM

# Copies application code
COPY --chown=appuser:appuser . .

# Exposes FastAPI port
EXPOSE 8000

# Starts FastAPI service
CMD ["uvicorn", "src.service.main:app", "--host", "0.0.0.0", "--port", "8000"]
