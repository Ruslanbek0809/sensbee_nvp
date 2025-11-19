"""
Minimal LLMTime wrapper for zero-shot time series forecasting using OpenAI API.

This is a simplified implementation that follows the LLMTime idea of converting
numbers to text and using zero-shot completion, without the full serialization pipeline.
"""

import logging
import os
import re
from typing import Optional

import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Default model name
DEFAULT_MODEL = "gpt-4o-mini"


def encode_series_to_prompt(series: pd.Series, horizon: int) -> str:
    """
    Convert a time series to a simple English prompt for forecasting.
    
    Args:
        series: Time series data (pandas Series)
        horizon: Number of future values to predict
        
    Returns:
        String prompt asking the model to predict the next values
    """
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


def call_openai_completion(prompt: str, model: Optional[str] = None) -> str:
    """
    Call OpenAI API to get a text completion for the prompt.
    
    Args:
        prompt: The prompt string
        model: Model name (defaults to env var OPENAI_MODEL or gpt-4o-mini)
        
    Returns:
        Response text from the model
        
    Raises:
        Exception: If the API call fails
    """
    # Get model name from env var or use default
    model_name = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    
    # Set temperature and max_tokens based on model
    # Lower temperature for more deterministic forecasts
    temperature = 0.3
    # Estimate max_tokens: roughly 10 tokens per number (conservative)
    max_tokens = 200  # Should be enough for most horizons
    
    try:
        logger.debug(f"Calling OpenAI API with model: {model_name}")
        
        # Use ChatCompletion for newer models (gpt-4, gpt-3.5-turbo, etc.)
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


def parse_forecast_from_text(response: str, horizon: int) -> np.ndarray:
    """
    Extract numeric values from the model's text response.
    
    This function is robust and handles various response formats:
    - "1.5, 2.3, 4.7"
    - "The next values are: 1.5, 2.3, 4.7"
    - "1.5\n2.3\n4.7"
    - etc.
    
    Args:
        response: Text response from the model
        horizon: Expected number of values
        
    Returns:
        numpy array of forecast values
        
    Raises:
        ValueError: If insufficient valid numbers are found
    """
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


def llmtime_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """
    Generate a forecast using the LLMTime approach: convert series to text,
    get zero-shot completion from OpenAI, and parse the result.
    
    Args:
        series: Time series data (pandas Series)
        horizon: Number of steps to forecast ahead
        
    Returns:
        numpy array of forecast values
        
    Raises:
        ValueError: If series is empty or other validation fails
        Exception: If API call or parsing fails
    """
    if len(series) == 0:
        raise ValueError("Series cannot be empty for forecasting")
    
    if horizon <= 0:
        raise ValueError("Horizon must be positive")
    
    # Step 1: Encode series to prompt
    prompt = encode_series_to_prompt(series, horizon)
    logger.debug(f"Generated prompt (first 200 chars): {prompt[:200]}...")
    
    # Step 2: Call OpenAI API
    response = call_openai_completion(prompt)
    logger.debug(f"Received response (first 200 chars): {response[:200]}...")
    
    # Step 3: Parse forecast from response
    forecast = parse_forecast_from_text(response, horizon)
    logger.info(f"Generated forecast of length {len(forecast)}")
    
    return forecast

