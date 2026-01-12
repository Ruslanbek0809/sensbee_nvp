# On-demand, input history, and few-shot learning based time series forecasting using LLM APIs.

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# Data scaler with transform and inverse transform functions.
@dataclass
class Scaler:
    transform: callable
    inverse_transform: callable


# Create a scaler based on history data. Here, we normalize data to roughly [0, 1] range using quantile-based scaling.
def create_scaler(history: np.ndarray, alpha: float = 0.95, beta: float = 0.3) -> Scaler:
    history = history[~np.isnan(history)]
    
    if len(history) == 0:
        return Scaler(transform=lambda x: x, inverse_transform=lambda x: x)
    
    min_val = np.min(history)
    max_val = np.max(history)
    
    # Shift to make all values positive, with margin (beta)
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


# Settings for serialization of numbers.
@dataclass
class SerializerSettings:
    base: int = 10
    prec: int = 3  # Precision after decimal point
    signed: bool = True
    time_sep: str = ", "  # Separator between time steps
    bit_sep: str = ""  # Separator between digits
    minus_sign: str = "-"


# Serialize array to string format suitable for LLM input.
def serialize_array(arr: np.ndarray, settings: SerializerSettings) -> str:
    formatted = []
    for val in arr:
        if np.isnan(val):
            formatted.append("NaN")
        else:
            # Format with precision mentioned in settings.
            if val >= 0:
                s = f"{val:.{settings.prec}f}"
            else:
                s = f"{settings.minus_sign}{abs(val):.{settings.prec}f}"
            formatted.append(s)
    
    return settings.time_sep.join(formatted)


# System message. Simpler approach.
SYSTEM_MESSAGE = ( 
    "You are a time series pattern continuation engine. "
    "You MUST output ONLY comma-separated decimal numbers. "
    "NO text, NO explanations, NO words - ONLY numbers separated by commas. "
    "Time series have cycles and fluctuations - they do NOT just go up or down linearly."
)

# Few-shot examples showing NON-LINEAR patterns. Used to break linear bias.
FEW_SHOT_EXAMPLES = """Example 1 (daily temperature cycle):
Input: 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.85, 0.7, 0.5, 0.3, 0.2, 0.25, 0.4, 0.6, 0.75, 0.85
Output: 0.9, 0.8, 0.65, 0.45, 0.3, 0.2, 0.25, 0.4

Example 2 (humidity fluctuation):
Input: 0.95, 0.92, 0.88, 0.85, 0.82, 0.85, 0.9, 0.93, 0.95, 0.94, 0.9, 0.86
Output: 0.83, 0.85, 0.88, 0.92, 0.95, 0.94, 0.91, 0.87

"""

# User prefix with few-shot learning
USER_PREFIX = (
    FEW_SHOT_EXAMPLES +
    "Now continue THIS sequence. Output ONLY the predicted numbers, nothing else:\n"
    "Input: "
)



# Calls Mistral API for text completion with multi-sample support.
def call_mistral_completion(
    prompt: str,
    model: Optional[str] = None,
    num_samples: int = 3,
    temperature: float = 1.0,
) -> List[str]:
    try:
        from mistralai import Mistral
        
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found")
        
        model_name = model or os.getenv("MISTRAL_MODEL", "mistral-medium-2505")
        client = Mistral(api_key=api_key)
        
        logger.debug(f"CALLING Mistral API: {model_name}, samples={num_samples}, temp={temperature}")
        
        # Generate multiple samples
        completions = []
        for i in range(num_samples):
            response = client.chat.complete(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=800,
            )
            completions.append(response.choices[0].message.content.strip())
            logger.debug(f"SAMPLE {i+1}/{num_samples}: {completions[-1][:100]}...")
        
        return completions
        
    except ImportError:
        raise ImportError("mistralai package NOT installed. Run: pip install mistralai")


# Calls Groq API for text completion with multi-sample support.
def call_groq_completion(
    prompt: str,
    model: Optional[str] = None,
    num_samples: int = 3,
    temperature: float = 1.0,
) -> List[str]:
    try:
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY NOT FOUND")
        
        model_name = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = Groq(api_key=api_key)
        
        logger.debug(f"CALLING Groq API: {model_name}, samples={num_samples}, temp={temperature}")
        
        completions = []
        for i in range(num_samples):
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=800,
            )
            completions.append(response.choices[0].message.content.strip())
            logger.debug(f"SAMPLE {i+1}/{num_samples}: {completions[-1][:100]}...")
        
        return completions
        
    except ImportError:
        raise ImportError("groq package NOT installed. Run: pip install groq")


# Calls OpenAI API for text completion with multi-sample support.
def call_openai_completion(
    prompt: str,
    model: Optional[str] = None,
    num_samples: int = 3,
    temperature: float = 1.0,
) -> List[str]:
    import openai
    
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    logger.debug(f"CALLING OpenAI API: {model_name}, samples={num_samples}, temp={temperature}")
    
    # OpenAI supports n parameter for multiple samples in single call
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=800,
        n=num_samples,
    )
    
    return [choice.message.content.strip() for choice in response.choices]


# Calls local LLM for text completion with TOKEN CONTROL.
def call_local_llm_completion(
    input_str: str,
    model: Optional[str] = None,
    num_samples: int = 3,
    temperature: float = 1.0,
    steps: int = 96,
) -> List[str]:
    try:
        from .local_llm import local_llm_completion, is_local_llm_available
        
        if not is_local_llm_available():
            raise ValueError("LOCAL LLM NOT AVAILABLE. LOCAL LLM REQUIRES GPU (CUDA FOR NVIDIA, MPS FOR APPLE SILICON). USE API-BASED PROVIDERS (mistral, groq, openai) INSTEAD.")
        
        model_name = model or os.getenv("LOCAL_MODEL", "llama2-7b")
        
        logger.info(f"USING LOCAL LLM: {model_name} WITH TOKEN CONTROL")
        
        return local_llm_completion(
            input_str=input_str,
            model_name=model_name,
            steps=steps,
            num_samples=num_samples,
            temperature=temperature,
        )
        
    except ImportError as e:
        raise ImportError(
            f"LOCAL LLM REQUIRES: pip install torch transformers accelerate\n"
            f"ERROR: {e}"
        )


# Unified LLM completion call with multi-sample support.
def call_llm_completion(
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    num_samples: int = 3,
    temperature: float = 1.0,
    steps: int = 96,
) -> List[str]:
    provider = (provider or os.getenv("LLM_PROVIDER", "mistral")).lower()
    
    if provider == "local":
        return call_local_llm_completion(prompt, model, num_samples, temperature, steps)
    elif provider == "mistral":
        return call_mistral_completion(prompt, model, num_samples, temperature)
    elif provider == "groq":
        return call_groq_completion(prompt, model, num_samples, temperature)
    elif provider == "openai":
        return call_openai_completion(prompt, model, num_samples, temperature)
    else:
        raise ValueError(
            f"UNKNOWN PROVIDER: {provider}. "
            f"SUPPORTED: local, mistral, groq, openai"
        )


# Parses LLM response to extract numeric values.
def parse_forecast_response(response: str, horizon: int) -> Optional[np.ndarray]:
    # Extracts all numbers (including negative and decimals)
    pattern = r"-?\d+\.?\d*"
    matches = re.findall(pattern, response)
    
    if len(matches) == 0:
        logger.warning(f"NO NUMBERS FOUND in response: {response[:100]}")
        return None
    
    try:
        values = [float(m) for m in matches]
    except ValueError:
        logger.warning(f"FAILED TO PARSE NUMBERS FROM RESPONSE")
        return None
    
    # Handle length mismatch
    if len(values) < horizon:
        # Pads with last value
        values.extend([values[-1]] * (horizon - len(values)))
    
    return np.array(values[:horizon])


# Aggregates multiple forecast samples.
def aggregate_samples(
    samples: List[np.ndarray],
    method: str = "median"
) -> np.ndarray:
    if len(samples) == 0:
        raise ValueError("NO VALID SAMPLES TO AGGREGATE")
    
    # Stack samples into matrix
    sample_matrix = np.stack(samples, axis=0)
    
    if method == "median":
        return np.median(sample_matrix, axis=0)
    elif method == "mean":
        return np.mean(sample_matrix, axis=0)
    else:
        raise ValueError(f"UNKNOWN AGGREGATION METHOD: {method}")


# Generates forecast using NVP LLMs. Main function.
def nvp_llms_forecast(
    series: pd.Series,
    horizon: int,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    column_name: str = "value",
    num_samples: int = 3,
    temperature: float = 1.0,
) -> np.ndarray:
    if len(series) == 0:
        raise ValueError("EMPTY SERIES")
    
    # Converts to numeric, coercing errors to NaN, then drops NaN values
    numeric_series = pd.to_numeric(series, errors='coerce')
    numeric_series = numeric_series.dropna()
    
    if len(numeric_series) == 0:
        raise ValueError("ALL VALUES ARE NaN after conversion")
    
    values = numeric_series.values.astype(float)
    
    # Logs original statistics before normalization
    original_min = float(np.min(values))
    original_max = float(np.max(values))
    logger.info(f"INPUT DATA: {len(values)} VALID POINTS, RANGE [{original_min:.2f}, {original_max:.2f}]")
    
    # Step 1: Creates scaler from history
    scaler = create_scaler(values)
    
    # Step 2: Normalizes input data
    normalized_values = scaler.transform(values)
    
    # Logs statistics
    logger.info(
        f"FORECASTING {column_name}: NORMALIZED RANGE [{normalized_values.min():.3f}, {normalized_values.max():.3f}]"
    )
    
    # Step 3: Serializes to string
    # Limit input length to avoid token limits (last ~200 points)
    # max_input_points = 200
    # if len(normalized_values) > max_input_points:
    #     normalized_values = normalized_values[-max_input_points:]
    #     logger.info(f"Truncated input to last {max_input_points} points")
    
    settings = SerializerSettings(prec=3, time_sep=", ")
    input_str = serialize_array(normalized_values, settings)
    
    # Build prompt
    prompt = USER_PREFIX + input_str + settings.time_sep
    
    logger.debug(f"PROMPT LENGTH: {len(prompt)} CHARS, INPUT POINTS: {len(normalized_values)}")
    
    # Step 4: Generates multiple samples from LLM
    completions = call_llm_completion(
        prompt=prompt,
        provider=provider,
        model=model,
        num_samples=num_samples,
        temperature=temperature,
        steps=horizon,  # Pass horizon for local LLM token calculation
    )
    
    # Step 5: Parses each response
    valid_samples = []
    for i, completion in enumerate(completions):
        # Logs first 200 chars of each completion for debugging
        logger.info(f"SAMPLE {i+1} RAW RESPONSE: {completion[:200]}...")
        
        parsed = parse_forecast_response(completion, horizon)
        if parsed is not None:
            valid_samples.append(parsed)
            # Checks if sample is linear (for debugging)
            diffs = np.diff(parsed)
            is_linear = np.std(diffs) < 0.01 * np.abs(np.mean(diffs)) if np.mean(diffs) != 0 else True
            logger.info(f"SAMPLE {i+1}: RANGE [{parsed.min():.3f}, {parsed.max():.3f}], LINEAR={is_linear}")
        else:
            logger.warning(f"SAMPLE {i+1}: INVALID (PARSING FAILED)")
    
    if len(valid_samples) == 0:
        raise ValueError("ALL LLM SAMPLES FAILED TO PARSE")
    
    logger.info(f"VALID SAMPLES: {len(valid_samples)}/{num_samples}")
    
    # Step 6: Takes MEDIAN of valid samples
    normalized_forecast = aggregate_samples(valid_samples, method="median")
    
    # Step 7: Denormalizes back to original scale
    forecast = scaler.inverse_transform(normalized_forecast)
    
    logger.info(
        f"GENERATED FORECAST: {len(forecast)} POINTS, "
        f"RANGE [{forecast.min():.2f}, {forecast.max():.2f}]"
    )
    
    return forecast



# # Fetches sensor data from SensBee API and generates forecast.
# def forecast_sensor_from_api(
#     sensor_id: Optional[str] = None,
#     api_key: Optional[str] = None,
#     column_name: Optional[str] = None,
#     horizon_hours: int = 24,
#     provider: Optional[str] = None,
#     model: Optional[str] = None,
#     base_url: Optional[str] = None,
#     limit: Optional[int] = None,
#     num_samples: int = 5,
#     temperature: float = 0.9,
# ) -> tuple[np.ndarray, pd.Series]:
#     series = load_sensor_series_from_api(
#         sensor_id=sensor_id,
#         api_key=api_key,
#         column_name=column_name,
#         base_url=base_url,
#         limit=limit,
#     )
    
#     if len(series) < 2:
#         raise ValueError(f"NOT ENOUGH DATA POINTS: {len(series)}")
    
#     # Calculates horizon in steps
#     sampling_minutes = (series.index[1] - series.index[0]).total_seconds() / 60
#     horizon_steps = int(horizon_hours * (60 / sampling_minutes))
    
#     from ..data_access.sensbee_client import DEFAULT_COLUMN_NAME
#     actual_column = column_name or DEFAULT_COLUMN_NAME
    
#     forecast = nvp_llms_forecast(
#         series=series,
#         horizon=horizon_steps,
#         provider=provider,
#         model=model,
#         column_name=actual_column,
#         num_samples=num_samples,
#         temperature=temperature,
#     )
    
#     return forecast, series


# # Loads sensor data from local JSON and generates forecast.
# def forecast_sensor_from_local_json(
#     column_name: Optional[str] = None,
#     horizon_hours: int = 24,
#     provider: Optional[str] = None,
#     model: Optional[str] = None,
#     num_samples: int = 5,
#     temperature: float = 0.9,
# ) -> tuple[np.ndarray, pd.Series]:
#     series = load_sensor_series_from_json(column_name=column_name)
    
#     if len(series) < 2:
#         raise ValueError(f"NOT ENOUGH DATA POINTS: {len(series)}")
    
#     sampling_minutes = (series.index[1] - series.index[0]).total_seconds() / 60
#     horizon_steps = int(horizon_hours * (60 / sampling_minutes))
    
#     from ..data_access.data_loader import DEFAULT_COLUMN_NAME
#     actual_column = column_name or DEFAULT_COLUMN_NAME
    
#     forecast = nvp_llms_forecast(
#         series=series,
#         horizon=horizon_steps,
#         provider=provider,
#         model=model,
#         column_name=actual_column,
#         num_samples=num_samples,
#         temperature=temperature,
#     )
    
#     return forecast, series
