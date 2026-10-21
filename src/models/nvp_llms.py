# Zero-shot LLM forecasting of time series on demand, with API and local models.
# It normalizes the series and can add an optional text context to the prompt.
#
# Pipeline:
#   1. Normalize the input with a quantile scaler (α=0.95, β=0.3).
#   2. Serialize it to a comma-separated string.
#   3. Prompt the LLM to continue the sequence.
#   4. Parse and filter N independent forecasts, then aggregate them.
#   5. Inverse transform the result to the original scale.
#
# Four filter rules discard invalid forecasts:
#   (a) Out-of-bounds values
#   (b) Constant or near-constant output (≤2 unique values)
#   (c) Flat output (std < 0.001)
#   (d) Perfectly linear trends (second derivative < 0.005)
#
# If all forecasts are filtered out, the result is the last observed value (naive baseline).

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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

# System message for API models. It is kept minimal, following the LLMTime findings.
# LLMTime paper: "Unlike PromptCast, we show that LLMs can be used directly as
# forecasters without any added text or prompt engineering"
SYSTEM_MESSAGE_API = (
    "You continue numerical sequences. Output only numbers separated by commas. "
    "No explanations, no text, just the numbers."
)

# System message for raw (not normalized) values with their meaning kept, used with 70B models
SYSTEM_MESSAGE_API_SEMANTIC = (
    "You forecast sensor data. Output only the predicted numbers separated by commas. "
    "No explanations or text."
)


# LLM Provider Functions
# Calls the Mistral API for text completion.
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
        
        # Choose the system message for this use case
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
# Works with completion models (gpt-3.5-turbo-instruct) and with chat models.
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
    
    # Instruct models use the completion API (the LLMTime approach, which works much better).
    # Completion models continue the raw sequence, without chat formatting.
    if "instruct" in model_name.lower():
        response = openai.completions.create(
            model=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            n=num_forecasts,
        )
        return [choice.text.strip() for choice in response.choices]
    
    # Chat models use the chat API
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


# Calls a local LLM with token control. This is the preferred method.
# It controls which tokens the model may produce, so the output is always a valid numeric sequence.
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


# One entry point for completions from any provider.
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
    
    # With prec=3 and integer mode, a value is about 3-4 digits plus a comma, roughly 5 tokens.
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

# Builds the full prompt for forecasting.
# LLMTime approach for base and local models: send only the serialized numbers.
# The trailing comma tells the model to keep generating.
# API chat models: the system message already limits the output.
# The user message is the sequence plus, at most, a short instruction.
def build_forecast_prompt(
    serialized_input: str,
    settings: SerializerSettings,
    context: Optional[str] = None,
    is_local: bool = False,
    horizon: Optional[int] = None,
) -> str:
    if is_local:
        # Local base models get the raw sequence only (LLMTime default).
        # The trailing comma after the last value tells the model to continue.
        return serialized_input

    # API chat models: as little text as possible, to keep the model writing numbers.
    parts = []

    if context:
        parts.append(context)
        parts.append("")

    parts.append(serialized_input)

    return "\n".join(parts)


# Combines several forecast samples into one prediction.
def aggregate_samples(
    samples: List[np.ndarray],
    method: str = "median",
) -> np.ndarray:
    if not samples:
        raise ValueError("NO VALID SAMPLES TO AGGREGATE")
    
    # Stack the samples into a matrix
    sample_matrix = np.stack(samples, axis=0)
    
    if method == "median":
        return np.median(sample_matrix, axis=0)
    elif method == "mean":
        return np.mean(sample_matrix, axis=0)
    else:
        raise ValueError(f"UNKNOWN AGGREGATION METHOD: {method}")


# Result of the LLMTime sampling steps. It holds:
#   - the valid samples on the model's scale (normalized when normalization is used)
#   - the scaler that maps them back
#   - the last observed value
#   - counts of completions, parse failures, samples removed by each filter rule, and valid samples
@dataclass
class LLMTimeSamples:
    samples: List[np.ndarray]
    scaler: Scaler
    last_value: float
    counts: Dict[str, int]

    # The median of the valid samples on the original scale (Steps 6-7).
    # use_normalization says whether the samples are normalized.
    def point(self, use_normalization: bool) -> np.ndarray:
        median = aggregate_samples(self.samples, method="median")
        return self.scaler.inverse_transform(median) if use_normalization else median


# Runs LLMTime steps 1-5 (scale, serialize, prompt, complete, parse and filter) and returns the valid samples.
# complete is the completion function. It is called like call_llm_completion, which is the default.
def llmtime_samples(
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
    complete: Optional[Callable[..., List[str]]] = None,
) -> LLMTimeSamples:
    if len(series) == 0:
        raise ValueError("EMPTY SERIES")
    
    # Convert to numbers and drop what cannot be converted
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
    
    # Find the provider and its serializer settings
    provider = (provider or os.getenv("LLM_PROVIDER", "mistral")).lower()
    is_local = provider == "local"
    
    settings = get_serializer_settings(
        model_name=model,
        provider=provider,
        is_local=is_local,
    )
    
    # Step 1: Create the scaler and normalize the values
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
    
    # Step 2: Serialize the input
    serialized_input = serialize_array(normalized_values, settings)
    
    logger.debug(f"SERIALIZED INPUT (FIRST 100 CHARS): {serialized_input[:100]}...")
    
    # Step 3: Build the context and the prompt
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
    
    # Step 4: Generate the completions
    # API models with raw values get the semantic system message, because the raw values keep their meaning
    use_semantic_prompt = (not is_local) and (not use_normalization)
    
    completions = (complete or call_llm_completion)(
        prompt=prompt,
        settings=settings,
        provider=provider,
        model=model,
        num_forecasts=num_forecasts,
        temperature=temperature,
        steps=horizon,
        use_semantic_prompt=use_semantic_prompt,
    )
    
    # Step 5: Parse the completions and filter out the invalid ones
    valid_samples = []
    counts = {"completions": len(completions), "parse_failed": 0, "filtered_a": 0, "filtered_b": 0,
              "filtered_c": 0, "filtered_d": 0}
    
    # The bounds depend on whether the values are normalized
    if use_normalization:
        # Normalized values should be roughly in [0, 2].
        # LLMTime scales so the α-percentile is 1.0, so values can exceed 1.
        # The bounds are wider than that, so valid extrapolations are not filtered.
        min_bound = -2.0
        max_bound = 5.0
        bound_description = "normalized"
    else:
        # Raw values: the input range plus a buffer.
        # The buffer is 2x the input range, to allow for extrapolation.
        input_range = original_max - original_min
        buffer = max(input_range * 2.0, 10.0)  # but at least 10 units
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
            counts["parse_failed"] += 1
            continue

        # Filter (a): out-of-bounds values
        if parsed.min() < min_bound or parsed.max() > max_bound:
            logger.warning(
                f"FORECAST {i+1}: FILTERED (a) out-of-bounds "
                f"[{parsed.min():.3f}, {parsed.max():.3f}] outside [{min_bound:.2f}, {max_bound:.2f}]"
            )
            counts["filtered_a"] += 1
            continue

        # Filter (b): constant or near-constant output (≤2 unique values)
        if len(np.unique(np.round(parsed, 3))) <= 2:
            logger.warning(f"FORECAST {i+1}: FILTERED (b) constant or near-constant (≤2 unique values)")
            counts["filtered_b"] += 1
            continue

        # Filter (c): flat output (std < 0.001)
        if float(np.std(parsed)) < 0.001:
            logger.warning(f"FORECAST {i+1}: FILTERED (c) flat output (std < 0.001)")
            counts["filtered_c"] += 1
            continue

        # Filter (d): perfectly linear trend (second derivative < 0.005)
        if len(parsed) > 4:
            second_diff = np.diff(parsed, 2)
            if np.all(np.abs(second_diff) < 0.005):
                logger.warning(f"FORECAST {i+1}: FILTERED (d) perfectly linear (2nd deriv < 0.005)")
                counts["filtered_d"] += 1
                continue

        valid_samples.append(parsed)
        logger.info(
            f"FORECAST {i+1}: PARSED {len(parsed)} VALUES, "
            f"RANGE [{parsed.min():.3f}, {parsed.max():.3f}] ✓"
        )
    
    counts["valid"] = len(valid_samples)
    return LLMTimeSamples(samples=valid_samples, scaler=scaler, last_value=float(values[-1]), counts=counts)


# Forecasts a series with an LLM and returns the point forecast: the median of the valid samples, or the last value
# when every sample is filtered out.
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
    result = llmtime_samples(
        series=series,
        horizon=horizon,
        provider=provider,
        model=model,
        column_name=column_name,
        num_forecasts=num_forecasts,
        temperature=temperature,
        use_normalization=use_normalization,
        include_context=include_context,
        alpha=alpha,
        beta=beta,
    )
    valid_samples = result.samples
    
    if not valid_samples:
        # Fallback: if every sample was filtered out, repeat the last value
        logger.warning("ALL SAMPLES FILTERED. USING LAST VALUE FALLBACK.")
        return np.full(horizon, result.last_value)
    
    logger.info(f"VALID FORECASTS AFTER FILTERING: {len(valid_samples)}/{num_forecasts}")
    
    # Steps 6-7: Aggregate the samples and invert the transform
    forecast = result.point(use_normalization)

    # Post-processing: clip visitor-type columns to non-negative values.
    # Visitor counts cannot be negative, but the scaler can give values below zero
    # when the model writes values under the normalized minimum.
    _NON_NEGATIVE_COLUMNS = {"visitors_total", "visitor_change", "visitors", "count"}
    if column_name.lower() in _NON_NEGATIVE_COLUMNS:
        forecast = np.clip(forecast, 0, None)

    logger.info(
        f"FORECAST: {len(forecast)} POINTS, RANGE [{forecast.min():.2f}, {forecast.max():.2f}]"
    )

    return forecast


# Forecasts from a local JSON file.
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


# Forecasts from the SensBee API.
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
