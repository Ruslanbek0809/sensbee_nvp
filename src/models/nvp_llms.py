# On-demand LLM-based time series forecasting using API and local models. Implements zero-shot forecasting with normalization and optional semantic context.
#
# Pipeline:
#   1. Normalize input using quantile scaler (α=0.95, β=0.3)
#   2. Serialize to comma-separated string
#   3. Prompt LLM to continue the sequence
#   4. Parse, filter, and aggregate N independent forecasts
#   5. Inverse transform to original scale
#
# Four filter rules discard invalid forecasts:
#   (a) Out-of-bounds values
#   (b) Constant or near-constant output (≤2 unique values)
#   (c) Flat output (std < 0.001)
#   (d) Perfectly linear trends (second derivative < 0.005)
#
# If all forecasts are filtered, the system falls back to the last observed value (naive baseline).

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .serialize import (
    Scaler,
    SerializerSettings,
    ModelFamily,
    create_scaler,
    get_serializer_settings,
    serialize_array,
    deserialize_string,
    build_statistical_context,
)

load_dotenv()
logger = logging.getLogger(__name__)


# PROMPT TEMPLATES

# System message for API-based models (kept minimal per LLMTime findings)
# LLMTime paper: "Unlike PromptCast, we show that LLMs can be used directly as 
# forecasters without any added text or prompt engineering"
SYSTEM_MESSAGE_API = (
    "You continue numerical sequences. Output only numbers separated by commas. "
    "No explanations, no text, just the numbers."
)

# Enhanced system message for raw/semantic values (70B models)
SYSTEM_MESSAGE_API_SEMANTIC = (
    "You forecast sensor data. Output only the predicted numbers separated by commas. "
    "No explanations or text."
)


# LLM Provider Functions
# Calls Mistral the API for text completion.
def call_mistral_api(
    prompt: str,
    model: Optional[str] = None,
    num_forecasts: int = 3,
    temperature: float = 0.9,
    max_tokens: int = 500,
) -> List[str]:
    try:
        from mistralai import Mistral
        
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found")
        
        model_name = model or os.getenv("MISTRAL_MODEL", "mistral-medium-2505")
        client = Mistral(api_key=api_key)
        
        logger.debug(f"CALLING MISTRAL API: MODEL={model_name}, PATHS={num_forecasts}")
        
        completions = []
        for i in range(num_forecasts):
            response = client.chat.complete(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE_API},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content.strip()
            completions.append(content)
            logger.debug(f"FORECAST {i+1}: {content[:80]}...")
        
        return completions
        
    except ImportError:
        raise ImportError("mistralai package NOT installed. Run: pip install mistralai")


# Calls the Groq API for text completion.
def call_groq_api(
    prompt: str,
    model: Optional[str] = None,
    num_forecasts: int = 3,
    temperature: float = 0.9,
    max_tokens: int = 500,
    use_semantic_prompt: bool = False,
) -> List[str]:
    try:
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY NOT FOUND")
        
        model_name = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = Groq(api_key=api_key)
        
        # Select system message based on use case
        system_message = SYSTEM_MESSAGE_API_SEMANTIC if use_semantic_prompt else SYSTEM_MESSAGE_API
        
        logger.debug(f"CALLING GROQ API: MODEL={model_name}, PATHS={num_forecasts}")
        
        completions = []
        for i in range(num_forecasts):
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content.strip()
            completions.append(content)
            logger.debug(f"FORECAST {i+1}: {content[:80]}...")
        
        return completions
        
    except ImportError:
        raise ImportError("groq package NOT installed. Run: pip install groq")


# Calls the OpenAI API for text completion.
# Supports both completion models (gpt-3.5-turbo-instruct) and chat models.
def call_openai_api(
    prompt: str,
    model: Optional[str] = None,
    num_forecasts: int = 3,
    temperature: float = 0.9,
    max_tokens: int = 500,
) -> List[str]:
    import openai
    
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-instruct") # os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    logger.debug(f"CALLING OPENAI API: MODEL={model_name}, PATHS={num_forecasts}")
    
    # Use COMPLETION API for instruct models (LLMTime approach - much better!)
    # Completion models do raw sequence continuation without chat formatting.
    if "instruct" in model_name.lower():
        response = openai.completions.create(
            model=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            n=num_forecasts,
        )
        return [choice.text.strip() for choice in response.choices]
    
    # Fallback to chat API for chat models
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE_API},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_forecasts,
    )
    
    return [choice.message.content.strip() for choice in response.choices]


# Calls local LLM with token control. This is the preferred method as it provides token-level control, ensuring the model outputs only valid numeric sequences.
def call_local_llm(
    input_str: str,
    settings: SerializerSettings,
    model: Optional[str] = None,
    num_forecasts: int = 3,
    temperature: float = 0.9,
    steps: int = 96,
) -> List[str]:
    try:
        from .local_llm import (
            local_llm_completion,
            is_local_llm_available,
        )
        
        if not is_local_llm_available():
            raise ValueError("LOCAL LLM NOT AVAILABLE. LOCAL LLM REQUIRES GPU (CUDA FOR NVIDIA, MPS FOR APPLE SILICON). USE API-BASED PROVIDERS (mistral, groq, openai) INSTEAD.")
        
        model_name = model or os.getenv("LOCAL_MODEL", "llama2-7b")
        
        logger.info(f"USING LOCAL LLM: {model_name} WITH TOKEN CONTROL")
        
        return local_llm_completion(
            input_str=input_str,
            settings=settings,
            model_name=model_name,
            steps=steps,
            num_forecasts=num_forecasts,
            temperature=temperature,
        )
        
    except ImportError as e:
        raise ImportError(
            f"Local LLM requires: pip install torch transformers accelerate\n"
            f"Error: {e}"
        )


# Unified LLM completion interface.
def call_llm_completion(
    prompt: str,
    settings: SerializerSettings,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    num_forecasts: int = 3,
    temperature: float = 0.9,
    steps: int = 96,
    use_semantic_prompt: bool = False,
) -> List[str]:
    provider = (provider or os.getenv("LLM_PROVIDER", "mistral")).lower()
    
    # With prec=3 and integer mode, each value is ~3-4 digits + comma ≈ 5 tokens.
    avg_tokens_per_step = 5 if settings.use_integers else 7
    max_tokens = int(avg_tokens_per_step * steps * 1.3)
    
    if provider == "local":
        return call_local_llm(
            input_str=prompt,
            settings=settings,
            model=model,
            num_forecasts=num_forecasts,
            temperature=temperature,
            steps=steps,
        )
    elif provider == "mistral":
        return call_mistral_api(prompt, model, num_forecasts, temperature, max_tokens)
    elif provider == "groq":
        return call_groq_api(prompt, model, num_forecasts, temperature, max_tokens, use_semantic_prompt)
    elif provider == "openai":
        return call_openai_api(prompt, model, num_forecasts, temperature, max_tokens)
    else:
        raise ValueError(
            f"UNKNOWN PROVIDER: {provider}. "
            f"SUPPORTED: local, mistral, groq, openai"
        )


# Main Forecasting Functions

# Builds the complete prompt for forecasting.
# LLMTime approach: for base/local models, send ONLY the serialized numbers
# (the trailing comma signals the model to continue generating).
# For API chat models, the system message already constrains the output;
# the user message should contain minimal instruction plus the sequence.
def build_forecast_prompt(
    serialized_input: str,
    settings: SerializerSettings,
    context: Optional[str] = None,
    is_local: bool = False,
    horizon: Optional[int] = None,
) -> str:
    if is_local:
        # Local base models: raw sequence only (LLMTime default).
        # The trailing comma after the last value signals continuation.
        return serialized_input

    # API chat models: minimal framing to keep the model in numeric mode.
    parts = []

    if context:
        parts.append(context)
        parts.append("")

    parts.append(serialized_input)

    return "\n".join(parts)


# Aggregates multiple forecast samples into a single prediction.
def aggregate_samples(
    samples: List[np.ndarray],
    method: str = "median",
) -> np.ndarray:
    if not samples:
        raise ValueError("NO VALID SAMPLES TO AGGREGATE")
    
    # Stacks samples into matrix
    sample_matrix = np.stack(samples, axis=0)
    
    if method == "median":
        return np.median(sample_matrix, axis=0)
    elif method == "mean":
        return np.mean(sample_matrix, axis=0)
    else:
        raise ValueError(f"UNKNOWN AGGREGATION METHOD: {method}")


# Generates forecast using LLM with hybrid approach.
def nvp_llms_forecast(
    series: pd.Series,
    horizon: int,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    column_name: str = "value",
    num_forecasts: int = 5,
    temperature: float = 0.9,
    use_normalization: bool = True,
    include_context: bool = False,
    alpha: float = 0.95,
    beta: float = 0.3,
) -> np.ndarray:
    if len(series) == 0:
        raise ValueError("EMPTY SERIES")
    
    # Converts to numeric and cleans values
    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_series = numeric_series.dropna()
    
    if len(numeric_series) == 0:
        raise ValueError("ALL VALUES ARE NaN AFTER CONVERSION")
    
    values = numeric_series.values.astype(float)
    original_min = float(np.min(values)) 
    original_max = float(np.max(values))
    
    logger.info(
        f"INPUT: {len(values)} POINTS, RANGE [{original_min:.2f}, {original_max:.2f}]"
    )
    
    # Determines provider and gets appropriate settings
    provider = (provider or os.getenv("LLM_PROVIDER", "mistral")).lower()
    is_local = provider == "local"
    
    settings = get_serializer_settings(
        model_name=model,
        provider=provider,
        is_local=is_local,
    )
    
    # Step 1: Creates scaler and normalizes values
    if use_normalization:
        scaler = create_scaler(values, alpha=alpha, beta=beta)
        normalized_values = scaler.transform(values)
        logger.info(
            f"NORMALIZED RANGE: [{normalized_values.min():.3f}, {normalized_values.max():.3f}]"
        )
    else:
        scaler = Scaler(
            original_min=original_min,
            original_max=original_max,
        )
        normalized_values = values
    
    # Step 2: Serializes input
    serialized_input = serialize_array(normalized_values, settings)
    
    logger.debug(f"SERIALIZED INPUT (FIRST 100 CHARS): {serialized_input[:100]}...")
    
    # Step 3: Builds context and prompt
    context = None
    if include_context:
        context = build_statistical_context(
            values=normalized_values,
            column_name=column_name,
            horizon=horizon,
            scaler=scaler if use_normalization else None,
            include_raw_stats=use_normalization,
        )
    
    prompt = build_forecast_prompt(
        serialized_input=serialized_input,
        settings=settings,
        context=context,
        is_local=is_local,
        horizon=horizon,
    )
    
    logger.debug(f"PROMPT LENGTH: {len(prompt)} CHARS")
    
    # Step 4: Generates completions
    # Use semantic prompt for API models when using raw values (semantic info preserved)
    use_semantic_prompt = (not is_local) and (not use_normalization)
    
    completions = call_llm_completion(
        prompt=prompt,
        settings=settings,
        provider=provider,
        model=model,
        num_forecasts=num_forecasts,
        temperature=temperature,
        steps=horizon,
        use_semantic_prompt=use_semantic_prompt,
    )
    
    # Step 5: Parses completions with outlier filtering
    valid_samples = []
    
    # Determines bounds based on normalization usage
    if use_normalization:
        # For normalized values: should be roughly in [0, 2] range after normalization.
        # LLMTime uses quantile scaling where α-percentile = 1.0, so values can exceed 1.
        # Widen bounds to avoid filtering valid extrapolations.
        min_bound = -2.0
        max_bound = 5.0
        bound_description = "normalized"
    else:
        # For raw values: use input data range with reasonable buffer
        # Allow 2x the input range to account for extrapolation
        input_range = original_max - original_min
        buffer = max(input_range * 2.0, 10.0)  # At least 10 units buffer
        min_bound = original_min - buffer
        max_bound = original_max + buffer
        bound_description = f"raw (input range: [{original_min:.2f}, {original_max:.2f}])"
    
    logger.debug(
        f"OUTLIER FILTERING: BOUNDS=[{min_bound:.2f}, {max_bound:.2f}] ({bound_description})"
    )
    
    for i, completion in enumerate(completions):
        logger.info(f"FORECAST {i+1} RAW: {completion[:250]}...")

        parsed = deserialize_string(
            text=completion,
            settings=settings,
            expected_length=horizon,
        )

        if parsed is None or len(parsed) == 0:
            logger.warning(f"FORECAST {i+1}: PARSING FAILED")
            continue

        # --- Filter (a): Out-of-bounds values ---
        if parsed.min() < min_bound or parsed.max() > max_bound:
            logger.warning(
                f"FORECAST {i+1}: FILTERED (a) out-of-bounds "
                f"[{parsed.min():.3f}, {parsed.max():.3f}] outside [{min_bound:.2f}, {max_bound:.2f}]"
            )
            continue

        # --- Filter (b): Constant or near-constant output (≤2 unique values) ---
        if len(np.unique(np.round(parsed, 3))) <= 2:
            logger.warning(f"FORECAST {i+1}: FILTERED (b) constant or near-constant (≤2 unique values)")
            continue

        # --- Filter (c): Flat output (std < 0.001) ---
        if float(np.std(parsed)) < 0.001:
            logger.warning(f"FORECAST {i+1}: FILTERED (c) flat output (std < 0.001)")
            continue

        # --- Filter (d): Perfectly linear trends (second derivative < 0.005) ---
        if len(parsed) > 4:
            second_diff = np.diff(parsed, 2)
            if np.all(np.abs(second_diff) < 0.005):
                logger.warning(f"FORECAST {i+1}: FILTERED (d) perfectly linear (2nd deriv < 0.005)")
                continue

        valid_samples.append(parsed)
        logger.info(
            f"FORECAST {i+1}: PARSED {len(parsed)} VALUES, "
            f"RANGE [{parsed.min():.3f}, {parsed.max():.3f}] ✓"
        )
    
    if not valid_samples:
        # Fallback: if all samples filtered, use last value extrapolation
        logger.warning("ALL SAMPLES FILTERED. USING LAST VALUE FALLBACK.")
        return np.full(horizon, values[-1])
    
    logger.info(f"VALID FORECASTS AFTER FILTERING: {len(valid_samples)}/{num_forecasts}")
    
    # Step 6: Aggregates samples
    normalized_forecast = aggregate_samples(valid_samples, method="median")
    
    # Step 7: Inverses transform
    if use_normalization:
        forecast = scaler.inverse_transform(normalized_forecast)
    else:
        forecast = normalized_forecast
    
    # Post-processing: clip visitor-type columns to non-negative values.
    # Visitor counts cannot be negative; the scaler can produce sub-zero values
    # when the model generates values below the normalized minimum.
    _NON_NEGATIVE_COLUMNS = {"visitors_total", "visitor_change", "visitors", "count"}
    if column_name.lower() in _NON_NEGATIVE_COLUMNS:
        forecast = np.clip(forecast, 0, None)

    logger.info(
        f"FORECAST: {len(forecast)} POINTS, RANGE [{forecast.min():.2f}, {forecast.max():.2f}]"
    )

    return forecast


# Function to forecast from local JSON file.
def forecast_from_json(
    json_path: str,
    column_name: str = "temperature",
    horizon_hours: int = 24,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    num_forecasts: int = 5,
    temperature: float = 0.9,
) -> Tuple[np.ndarray, pd.Series]:
    from ..data_access.data_loader import load_sensor_series_from_json
    
    series = load_sensor_series_from_json(
        path=json_path,
        column_name=column_name,
    )
    
    if len(series) < 2:
        raise ValueError(f"NOT ENOUGH DATA POINTS: {len(series)}")
    
    # Calculate horizon in steps
    if isinstance(series.index[0], pd.Timestamp) and len(series) > 1:
        sampling_minutes = (series.index[1] - series.index[0]).total_seconds() / 60
        horizon_steps = int(horizon_hours * (60 / sampling_minutes))
    else:
        horizon_steps = horizon_hours
    
    forecast = nvp_llms_forecast(
        series=series,
        horizon=horizon_steps,
        provider=provider,
        model=model,
        column_name=column_name,
        num_forecasts=num_forecasts,
        temperature=temperature,
    )
    
    return forecast, series


# Function to forecast from SensBee API.
def forecast_from_api(
    sensor_id: str,
    column_name: str = "temperature",
    horizon_hours: int = 24,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    history_days: int = 7,
    num_forecasts: int = 5,
    temperature: float = 0.9,
) -> Tuple[np.ndarray, pd.Series]:
    from ..data_access.sensbee_client import load_sensor_series_from_api

    series = load_sensor_series_from_api(
        sensor_id=sensor_id,
        api_key=api_key,
        column_name=column_name,
        window_hours=history_days * 24,
        base_url=base_url,
    )
    
    if len(series) < 2:
        raise ValueError(f"Not enough data points: {len(series)}")
    
    # Calculate horizon in steps
    sampling_minutes = (series.index[1] - series.index[0]).total_seconds() / 60
    horizon_steps = int(horizon_hours * (60 / sampling_minutes))
    
    forecast = nvp_llms_forecast(
        series=series,
        horizon=horizon_steps,
        provider=provider,
        model=model,
        column_name=column_name,
        num_forecasts=num_forecasts,
        temperature=temperature,
    )
    
    return forecast, series
