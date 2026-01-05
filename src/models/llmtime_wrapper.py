
# LLMTime wrapper for zero-shot time series forecasting using LLM APIs.
# 
# This implementation integrates key ideas from:
#   - LLMTime paper: Normalization/scaling of input data
#   - TIME-LLM paper: Prompt-as-Prefix with input statistics
#
# Key features:
#   - Normalization: Scales data to a normalized range for better LLM understanding
#   - Input Statistics: Provides min, max, median, trend to guide LLM reasoning
#   - Context-aware prompts: Sensor-specific metadata for realistic predictions

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ..data_access.data_loader import load_sensor_series_from_json
from ..data_access.sensbee_client import load_sensor_series_from_api

load_dotenv()
logger = logging.getLogger(__name__)


# =============================================================================
# Scaler (from LLMTime)
# =============================================================================
# Normalization is CRITICAL for LLM forecasting:
#   - LLMs work better with normalized values (0-1 range)
#   - Raw values like 23.5°C are harder for LLMs to reason about
#   - We denormalize the output back to original scale

@dataclass
class Scaler:
    """Data scaler with transform and inverse transform functions."""
    transform: callable
    inverse_transform: callable


def create_scaler(history: np.ndarray, alpha: float = 0.95, beta: float = 0.3) -> Scaler:
    """
    Create a scaler based on history data.
    
    The scaler normalizes data to roughly [0, 1] range using quantile-based scaling.
    This helps LLMs reason about the data patterns rather than absolute values.
    
    Args:
        history: Historical values to derive scaling from
        alpha: Quantile for scaling (default 0.95)
        beta: Shift parameter to avoid negative values (default 0.3)
    
    Returns:
        Scaler object with transform and inverse_transform methods
    """
    history = history[~np.isnan(history)]
    
    if len(history) == 0:
        return Scaler(transform=lambda x: x, inverse_transform=lambda x: x)
    
    min_val = np.min(history)
    max_val = np.max(history)
    
    # Shift to make all values positive, with some margin
    shift = min_val - beta * (max_val - min_val)
    
    # Scale factor based on quantile
    scale = np.quantile(history - shift, alpha)
    if scale == 0:
        scale = 1.0
    
    def transform(x: np.ndarray) -> np.ndarray:
        return (x - shift) / scale
    
    def inverse_transform(x: np.ndarray) -> np.ndarray:
        return x * scale + shift
    
    return Scaler(transform=transform, inverse_transform=inverse_transform)


# =============================================================================
# Input Statistics (from TIME-LLM Prompt-as-Prefix)
# =============================================================================
# TIME-LLM uses input statistics to help the LLM understand the data:
#   - Min, max, median values
#   - Trend direction (upward/downward)
#   - This replaces the "top-k lags" which requires FFT

def compute_input_statistics(values: np.ndarray) -> dict:
    """
    Compute statistics for Prompt-as-Prefix approach.
    
    These statistics help the LLM understand the data context and generate
    more realistic forecasts that follow the observed patterns.
    """
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "trend": "upward" if np.sum(np.diff(values)) > 0 else "downward",
        "last_value": float(values[-1]),
        "range": float(np.max(values) - np.min(values)),
    }


# =============================================================================
# Sensor Context (Domain Knowledge)
# =============================================================================

def get_sensor_context(column_name: str) -> dict:
    """Get domain-specific context for sensor types."""
    col = column_name.lower()
    
    if "temperature" in col or "temp" in col:
        return {
            "type": "temperature",
            "unit": "°C",
            "description": "Weather temperature measurement from urban weather station",
            "pattern": "daily cycles with morning lows and afternoon highs, influenced by weather fronts",
            "typical_range": "-15 to 35",
        }
    elif "humidity" in col:
        return {
            "type": "relative humidity",
            "unit": "%",
            "description": "Relative humidity measurement from urban weather station",
            "pattern": "inverse correlation with temperature, spikes during rain, higher at night",
            "typical_range": "30 to 100",
        }
    elif "wind" in col:
        return {
            "type": "wind speed",
            "unit": "m/s",
            "description": "Wind speed measurement from urban weather station",
            "pattern": "variable with gusts, typically higher during day, weather-dependent",
            "typical_range": "0 to 15",
        }
    elif "precip" in col or "rain" in col:
        return {
            "type": "precipitation",
            "unit": "mm",
            "description": "Precipitation/rainfall measurement",
            "pattern": "intermittent with long zero periods, event-based spikes",
            "typical_range": "0 to 10",
        }
    else:
        return {
            "type": "sensor measurement",
            "unit": "",
            "description": "Smart city sensor measurement",
            "pattern": "time-varying signal with potential daily patterns",
            "typical_range": "varies",
        }


# =============================================================================
# Prompt Construction (Combining LLMTime + TIME-LLM approaches)
# =============================================================================

def serialize_values(values: np.ndarray, precision: int = 2) -> str:
    """
    Serialize numeric values to a compact string format.
    
    Uses simple formatting with configurable precision.
    LLMTime paper notes that proper number formatting is crucial for tokenization.
    """
    formatted = []
    for v in values:
        if np.isnan(v):
            formatted.append("NaN")
        elif isinstance(v, (int, np.integer)) or v == int(v):
            formatted.append(str(int(v)))
        else:
            # Format with precision, remove trailing zeros
            s = f"{v:.{precision}f}".rstrip("0").rstrip(".")
            formatted.append(s)
    
    return ", ".join(formatted)


def build_forecast_prompt(
    normalized_values: np.ndarray,
    original_stats: dict,
    horizon: int,
    column_name: str,
    sampling_minutes: int,
) -> str:
    """
    Build the forecast prompt using Prompt-as-Prefix approach from TIME-LLM.
    
    The prompt has 3 components:
        1. Dataset context: Description of the sensor and data source
        2. Task instruction: What the model should do
        3. Input statistics: Min, max, median, trend to guide reasoning
    
    Args:
        normalized_values: Scaled values (roughly 0-1 range)
        original_stats: Statistics computed on ORIGINAL (non-normalized) values
        horizon: Number of steps to predict
        column_name: Name of the variable being forecast
        sampling_minutes: Time interval between data points
    """
    context = get_sensor_context(column_name)
    
    # Limit input length to avoid token limits (last ~400 points = ~4 days for 15-min data)
    max_points = 400
    if len(normalized_values) > max_points:
        normalized_values = normalized_values[-max_points:]
    
    # Serialize normalized values
    values_str = serialize_values(normalized_values, precision=3)
    
    # Build prompt with 3 components (TIME-LLM Prompt-as-Prefix)
    prompt = f"""Dataset description: {context['description']}. Data is {context['type']} measured in {context['unit']} at {sampling_minutes}-minute intervals from Ilmenau, Germany smart city sensors.

Task: Forecast the next {horizon} values given the historical data below. The values are normalized (scaled).

Input statistics (original scale):
- Min: {original_stats['min']:.2f} {context['unit']}
- Max: {original_stats['max']:.2f} {context['unit']}
- Median: {original_stats['median']:.2f} {context['unit']}
- Recent trend: {original_stats['trend']}
- Last value: {original_stats['last_value']:.2f} {context['unit']}

Pattern guidance: {context['pattern']}. Typical range: {context['typical_range']} {context['unit']}.

Historical data (normalized): {values_str}

Predict the next {horizon} normalized values. Output ONLY comma-separated numbers, maintaining realistic patterns and avoiding linear trends."""
    
    return prompt


# =============================================================================
# LLM API Calls
# =============================================================================

def call_mistral_completion(prompt: str, model: Optional[str] = None) -> str:
    """Call Mistral API for text completion."""
    try:
        from mistralai import Mistral
        
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found")
        
        model_name = model or os.getenv("MISTRAL_MODEL", "mistral-medium-2505")
        client = Mistral(api_key=api_key)
        
        logger.debug(f"Calling Mistral API: {model_name}")
        
        response = client.chat.complete(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a time series forecasting expert. Output only comma-separated numbers.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=600,
        )
        
        return response.choices[0].message.content.strip()
        
    except ImportError:
        raise ImportError("mistralai package not installed. Run: pip install mistralai")


def call_groq_completion(prompt: str, model: Optional[str] = None) -> str:
    """Call Groq API for text completion."""
    try:
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found")
        
        model_name = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        client = Groq(api_key=api_key)
        
        logger.debug(f"Calling Groq API: {model_name}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a time series forecasting expert. Output only comma-separated numbers.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=600,
        )
        
        return response.choices[0].message.content.strip()
        
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")


def call_openai_completion(prompt: str, model: Optional[str] = None) -> str:
    """Call OpenAI API for text completion."""
    import openai
    
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    logger.debug(f"Calling OpenAI API: {model_name}")
    
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a time series forecasting expert. Output only comma-separated numbers.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=600,
    )
    
    return response.choices[0].message.content.strip()


def call_llm_completion(prompt: str, provider: Optional[str] = None, model: Optional[str] = None) -> str:
    """Unified LLM API call."""
    provider = (provider or os.getenv("LLM_PROVIDER", "mistral")).lower()
    
    if provider == "mistral":
        return call_mistral_completion(prompt, model)
    elif provider == "groq":
        return call_groq_completion(prompt, model)
    elif provider == "openai":
        return call_openai_completion(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: mistral, groq, openai")


# =============================================================================
# Response Parsing
# =============================================================================

def parse_forecast_response(response: str, horizon: int) -> np.ndarray:
    """
    Parse LLM response to extract numeric values.
    
    Handles various response formats and pads if too few values returned.
    """
    # Extract all numbers (including negative and decimals)
    pattern = r"-?\d+\.?\d*"
    matches = re.findall(pattern, response)
    
    if len(matches) == 0:
        raise ValueError(f"No numbers found in response: {response[:200]}")
    
    values = [float(m) for m in matches]
    
    # Handle length mismatch
    if len(values) < horizon:
        logger.warning(f"Got {len(values)} values, expected {horizon}. Padding with last value.")
        values.extend([values[-1]] * (horizon - len(values)))
    
    return np.array(values[:horizon])


# =============================================================================
# Main Forecast Function
# =============================================================================

def llmtime_forecast(
    series: pd.Series,
    horizon: int,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    column_name: str = "value",
) -> np.ndarray:
    """
    Generate forecast using LLM with normalization and Prompt-as-Prefix.
    
    Pipeline:
        1. Create scaler from history (LLMTime)
        2. Normalize input data
        3. Compute statistics on original data (TIME-LLM)
        4. Build prompt with statistics and normalized data
        5. Call LLM API
        6. Parse response
        7. Denormalize output back to original scale
    
    Args:
        series: Historical time series data
        horizon: Number of steps to forecast
        provider: LLM provider (mistral, groq, openai)
        model: Specific model name
        column_name: Variable name for context
    
    Returns:
        Forecast values in original scale
    """
    if len(series) == 0:
        raise ValueError("Empty series")
    
    # Detect sampling interval
    if len(series) >= 2:
        time_diff = (series.index[1] - series.index[0]).total_seconds() / 60
        sampling_minutes = int(round(time_diff))
    else:
        sampling_minutes = 15
    
    values = series.astype(float).values
    
    # Step 1: Create scaler from history (LLMTime approach)
    scaler = create_scaler(values)
    
    # Step 2: Normalize input data
    normalized_values = scaler.transform(values)
    
    # Step 3: Compute statistics on ORIGINAL data (TIME-LLM approach)
    original_stats = compute_input_statistics(values)
    
    logger.info(
        f"Forecasting {column_name}: {len(values)} points, "
        f"range [{original_stats['min']:.2f}, {original_stats['max']:.2f}], "
        f"trend: {original_stats['trend']}"
    )
    
    # Step 4: Build prompt with statistics and normalized data
    prompt = build_forecast_prompt(
        normalized_values=normalized_values,
        original_stats=original_stats,
        horizon=horizon,
        column_name=column_name,
        sampling_minutes=sampling_minutes,
    )
    
    logger.debug(f"Prompt length: {len(prompt)} chars")
    
    # Step 5: Call LLM API
    response = call_llm_completion(prompt, provider=provider, model=model)
    logger.debug(f"Response: {response[:200]}...")
    
    # Step 6: Parse response (normalized values)
    normalized_forecast = parse_forecast_response(response, horizon)
    
    # Step 7: Denormalize back to original scale
    forecast = scaler.inverse_transform(normalized_forecast)
    
    logger.info(f"Generated forecast: {len(forecast)} points, range [{forecast.min():.2f}, {forecast.max():.2f}]")
    
    return forecast


# =============================================================================
# High-Level API Functions
# =============================================================================

def forecast_sensor_from_api(
    sensor_id: Optional[str] = None,
    api_key: Optional[str] = None,
    column_name: Optional[str] = None,
    horizon_hours: int = 24,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    limit: Optional[int] = None,
) -> tuple[np.ndarray, pd.Series]:
    """Fetch sensor data from SensBee API and generate forecast."""
    series = load_sensor_series_from_api(
        sensor_id=sensor_id,
        api_key=api_key,
        column_name=column_name,
        base_url=base_url,
        limit=limit,
    )
    
    if len(series) < 2:
        raise ValueError(f"Not enough data points: {len(series)}")
    
    # Calculate horizon in steps
    sampling_minutes = (series.index[1] - series.index[0]).total_seconds() / 60
    horizon_steps = int(horizon_hours * (60 / sampling_minutes))
    
    from ..data_access.sensbee_client import DEFAULT_COLUMN_NAME
    actual_column = column_name or DEFAULT_COLUMN_NAME
    
    forecast = llmtime_forecast(
        series=series,
        horizon=horizon_steps,
        provider=provider,
        model=model,
        column_name=actual_column,
    )
    
    return forecast, series


def forecast_sensor_from_local_json(
    column_name: Optional[str] = None,
    horizon_hours: int = 24,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[np.ndarray, pd.Series]:
    """Load sensor data from local JSON and generate forecast."""
    series = load_sensor_series_from_json(column_name=column_name)
    
    if len(series) < 2:
        raise ValueError(f"Not enough data points: {len(series)}")
    
    sampling_minutes = (series.index[1] - series.index[0]).total_seconds() / 60
    horizon_steps = int(horizon_hours * (60 / sampling_minutes))
    
    from ..data_access.data_loader import DEFAULT_COLUMN_NAME
    actual_column = column_name or DEFAULT_COLUMN_NAME
    
    forecast = llmtime_forecast(
        series=series,
        horizon=horizon_steps,
        provider=provider,
        model=model,
        column_name=actual_column,
    )
    
    return forecast, series
