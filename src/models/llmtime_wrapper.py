"""
Minimal LLMTime wrapper for zero-shot time series forecasting using various LLM APIs.

This is a simplified implementation that follows the LLMTime idea of converting
numbers to text and using zero-shot completion, without the full serialization pipeline.
"""

import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Default model name
DEFAULT_MODEL = "mistral-small-latest" # DEFAULT: Mistral Small (has free tier)

# Convert a time series to a simple English prompt for forecasting.
def encode_series_to_prompt(series: pd.Series, horizon: int) -> str:
    if len(series) == 0:
        raise ValueError("Series cannot be empty")
    
    # Convert series to list of values, format to reasonable precision
    values = series.tolist()
    
    # Format values: use 2 decimal places for floats, integers as-is
    formatted_values = []
    for val in values:
        if isinstance(val, (int, np.integer)):
            formatted_values.append(str(val))
        elif isinstance(val, (float, np.floating)):
            # Round to 2 decimal places, remove trailing zeros
            formatted_val = f"{val:.2f}".rstrip("0").rstrip(".")
            formatted_values.append(formatted_val)
        else:
            # Fallback: convert to string
            formatted_values.append(str(val))
    
    # Create comma-separated list
    values_str = ", ".join(formatted_values)
    
    # Build prompt
    prompt = (
        f"Given the following time series: {values_str}, "
        f"predict the next {horizon} values. "
        f"Return only the {horizon} numbers separated by commas, without any explanation."
    )
    
    return prompt


# Calls Mistral API to get a text completion.
def call_mistral_completion(prompt: str, model: Optional[str] = None) -> str:
    try:
        from mistralai import Mistral
        
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY not found. Get a free key at https://console.mistral.ai/"
            )
        
        model_name = model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")
        client = Mistral(api_key=api_key)
        
        logger.debug(f"Calling Mistral API with model: {model_name}")
        
        response = client.chat.complete(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that predicts time series values. Return only numbers separated by commas.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        
        return response.choices[0].message.content.strip()
        
    except ImportError:
        raise ImportError(
            "mistralai package not installed. Install with: pip install mistralai"
        )
    except Exception as e:
        logger.error(f"Mistral API call failed: {e}")
        raise


# Calls Groq API to get a text completion.
def call_groq_completion(prompt: str, model: Optional[str] = None) -> str:
    try:
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Get a free key at https://console.groq.com/"
            )
        
        model_name = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        client = Groq(api_key=api_key)
        
        logger.debug(f"Calling Groq API with model: {model_name}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that predicts time series values. Return only numbers separated by commas.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        
        return response.choices[0].message.content.strip()
        
    except ImportError:
        raise ImportError(
            "groq package not installed. Install with: pip install groq"
        )
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        raise


# Call OpenAI API to get a text completion for the prompt.
def call_openai_completion(prompt: str, model: Optional[str] = None) -> str:
    import openai
    
    # Get model name from env var or use default
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Set temperature and max_tokens
    temperature = 0.3
    max_tokens = 200
    
    try:
        logger.debug(f"Calling OpenAI API with model: {model_name}")
        
        # Use ChatCompletion for newer models
        if model_name.startswith("gpt-4") or model_name.startswith("gpt-3.5"):
            response = openai.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that predicts time series values. Return only numbers separated by commas.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        else:
            # Use Completion API for older models
            response = openai.completions.create(
                model=model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].text.strip()
            
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise


# Unified function to call different LLM providers.
def call_llm_completion(prompt: str, provider: Optional[str] = None, model: Optional[str] = None) -> str:
    # Auto-detect provider from model name or env var
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "mistral").lower()
    
    provider = provider.lower()
    
    if provider == "openai":
        return call_openai_completion(prompt, model)
    elif provider == "mistral":
        return call_mistral_completion(prompt, model)
    elif provider == "groq":
        return call_groq_completion(prompt, model)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: 'openai', 'mistral', 'groq'"
        )


# Extract numeric values from the model's text response. Handles various response formats.
def parse_forecast_from_text(response: str, horizon: int) -> np.ndarray:
    if not response:
        raise ValueError("Empty response from model")
    
    # Extract all numbers from the response using regex
    # Matches integers and floats (including negative numbers)
    number_pattern = r"-?\d+\.?\d*"
    matches = re.findall(number_pattern, response)
    
    if len(matches) == 0:
        raise ValueError(f"No numbers found in response: {response}")
    
    # Convert to floats
    try:
        values = [float(match) for match in matches]
    except ValueError as e:
        raise ValueError(f"Failed to parse numbers from response: {e}")
    
    # Take the first 'horizon' values
    if len(values) < horizon:
        logger.warning(
            f"Only found {len(values)} values in response, expected {horizon}. "
            f"Response: {response}"
        )
        # Pad with the last value if we have at least one
        if len(values) > 0:
            values.extend([values[-1]] * (horizon - len(values)))
        else:
            raise ValueError(f"Insufficient values found: {len(values)} < {horizon}")
    else:
        # Take exactly horizon values
        values = values[:horizon]
    
    return np.array(values)


# Generate a forecast using the LLMTime approach: convert series to text, get zero-shot completion from an LLM API, and parse the result.
def llmtime_forecast(
    series: pd.Series,
    horizon: int,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> np.ndarray:
    if len(series) == 0:
        raise ValueError("Series cannot be empty for forecasting")
    
    if horizon <= 0:
        raise ValueError("Horizon must be positive")
    
    # Step 1: Encode series to prompt
    prompt = encode_series_to_prompt(series, horizon)
    logger.debug(f"Generated prompt (first 200 chars): {prompt[:200]}...")
    
    # Step 2: Call LLM API
    response = call_llm_completion(prompt, provider=provider, model=model)
    logger.debug(f"Received response (first 200 chars): {response[:200]}...")
    
    # Step 3: Parse forecast from response
    forecast = parse_forecast_from_text(response, horizon)
    logger.info(f"Generated forecast of length {len(forecast)}")
    
    return forecast
