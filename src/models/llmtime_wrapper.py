# Minimal LLMTime wrapper for zero-shot time series forecasting using various LLM APIs.
# This is a simple implementation that follows the LLMTime idea of converting numbers to text.

import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ..data_access.data_loader import load_sensor_series_from_json

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Get context for urban sensor types (based on SensBee smart city sensors)
def get_sensor_context(column_name: str) -> dict:
    col = column_name.lower()
    
    if "temperature" in col or "temp" in col:
        return {
            "type": "temperature",
            "unit": "°C",
            "pattern": "daily cycles and weather fronts",
            "range": "-15 to 30"
        }
    elif "humidity" in col:
        return {
            "type": "relative humidity",
            "unit": "%",
            "pattern": "0–100% with daily cycles and spikes during rain",
            "range": "0 to 100"
        }
    else:
        return {
            "type": "sensor value",
            "unit": "",
            "pattern": "specific variable's time-series patterns",
            "range": "possible values for this sensor"
        }


# Convert a time series to a context-aware prompt for forecasting.
def encode_series_to_prompt(
    series: pd.Series,
    horizon: int,
    column_name: str = "value",
    sampling_minutes: int = 15,
) -> str:
    if len(series) == 0:
        raise ValueError("Series cannot be empty")
    
    # Use only the last 7 days (for 15-min data this is 7*24*4 = 672 points)
    max_points = 7 * 24 * (60 // sampling_minutes)
    if len(series) > max_points:
        series = series.iloc[-max_points:]
    
    # Convert to numeric values
    values = series.astype(float).tolist()
    
    # Format numbers compactly
    formatted_values = []
    for v in values:
        if isinstance(v, (int, np.integer)):
            formatted_values.append(str(int(v)))
        else:
            formatted_values.append(("{:.2f}".format(float(v))).rstrip("0").rstrip("."))
    
    values_str = ", ".join(formatted_values)

    # COMMENTED for now: Only show the tail to keep the prompt short (~ last 200 points = ~2 days at 15min)
    # tail_len = 200
    # shown = formatted_values[-tail_len:]
    # values_str = ", ".join(shown)
    # if len(formatted_values) > tail_len:
    #     values_str = "... " + values_str
    
    # Get sensor context
    context = get_sensor_context(column_name)
    
    # Build compact prompt (output format handled by system message, not duplicated here)
    prompt = (
        f"Forecast {context['type']} ({context['unit']}) for Ilmenau urban sensors. Interval: {sampling_minutes}min. "
        f"Pattern: {context['pattern']}. Range: {context.get('range', 'realistic')}.\n\n. Recent: {values_str}\n\n"
        f"Predict next {horizon} values. Requirements: realistic fluctuations, daily cycles, avoid linear trends, stay in range."
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
        
        model_name = model or os.getenv("MISTRAL_MODEL", "mistral-medium-2508") # mistral-small-2506 or mistral-medium-2508
        client = Mistral(api_key=api_key)
        
        logger.debug(f"CALLING Mistral API with model: {model_name}")
        
        response = client.chat.complete(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Forecast urban sensor time series. Output only numbers separated by commas."
                    ), # Content here is general instructions for the model like role definition. 
                },
                {"role": "user", "content": prompt}, # Prompt is task-specific instructions
            ],
            temperature=0.5,
            max_tokens=500,  # Enough for 96-step forecasts output
        )
        
        return response.choices[0].message.content.strip()
        
    except ImportError:
        raise ImportError(
            "mistralai package NOT INSTALLED. Install with: pip install mistralai"
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
        
        logger.debug(f"CALLING Groq API with model: {model_name}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Forecast urban sensor time series. Output only numbers separated by commas.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=500,
        )
        
        return response.choices[0].message.content.strip()
        
    except ImportError:
        raise ImportError(
            "groq package NOT INSTALLED. Install with: pip install groq"
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
    temperature = 0.5
    max_tokens = 500
    
    try:
        logger.debug(f"CALLING OpenAI API with model: {model_name}")
        
        # Use ChatCompletion for newer models
        if model_name.startswith("gpt-4") or model_name.startswith("gpt-3.5"):
            response = openai.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Forecast urban sensor time series. Output only numbers separated by commas.",
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
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
    
    provider = provider.lower()
    
    if provider == "mistral":
        return call_mistral_completion(prompt, model)
    elif provider == "groq":
        return call_groq_completion(prompt, model)
    elif provider == "openai":
        return call_openai_completion(prompt, model)
    else:
        raise ValueError(f"UNKNOWN PROVIDER: {provider}. Supported: 'openai', 'mistral', 'groq'")


# Extract numeric values from the model's text response. Handles various response formats.
def parse_forecast_from_text(response: str, horizon: int) -> np.ndarray:
    
    # Extract all numbers from the response using regex
    # Matches integers and floats (including negative numbers)
    number_pattern = r"-?\d+\.?\d*"
    matches = re.findall(number_pattern, response)
    
    if len(matches) == 0:
        raise ValueError(f"NO NUMBERS FOUND in response: {response}")
    
    # Convert to floats
    try:
        values = [float(match) for match in matches]
    except ValueError as e:
        raise ValueError(f"FAILED to parse numbers from response: {e}")
    
    # Take the first 'horizon' values
    if len(values) < horizon:
        logger.warning(f"ONLY FOUND {len(values)} values in response, expected {horizon}. Response: {response}")
        # 1 solution is to pad with the last value if we have at least one
        if len(values) > 0:
            values.extend([values[-1]] * (horizon - len(values)))
    else:
        # Take exactly horizon values
        values = values[:horizon]
    
    return np.array(values)

# Generate forecast with context-aware prompts that use history effectively.
def llmtime_forecast(
    series: pd.Series,
    horizon: int,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    column_name: str = "value",
) -> np.ndarray:
    if len(series) == 0:
        raise ValueError("EMPTY SERIES")
    
    # Step 1: Encode series to context-aware prompt
    # Detect sampling interval from series index (assume regular intervals)
    if len(series) >= 2:
        time_diff = (series.index[1] - series.index[0]).total_seconds() / 60
        sampling_minutes = int(round(time_diff))
    else:
        sampling_minutes = 15  # Default for SensBee 15-minute data
    
    prompt = encode_series_to_prompt(
        series, horizon, column_name=column_name, sampling_minutes=sampling_minutes
    )
    logger.debug(f"GENERATED INPUT / PROMPT (first 300 chars): {prompt[:300]}...")
    
    # Step 2: Call LLM API
    response = call_llm_completion(prompt, provider=provider, model=model)
    logger.debug(f"RESPONSE from LLM (first 200 chars): {response[:200]}...")
    
    # Step 3: Parse forecast from response
    forecast = parse_forecast_from_text(response, horizon)
    logger.info(f"PARSED FORECAST of length {len(forecast)}")
    
    return forecast


# Load any urban sensor series from local JSON file and generate LLM forecast.
def forecast_sensor_from_fixture(
    column_name: str = "temperature",
    horizon_hours: int = 24,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> np.ndarray:
    # Path to the data file
    data_path = Path(__file__).parent.parent.parent / "data" / "real_sensbee_json_data.json"
    
    # Load sensor series: use original 15-minute intervals
    series = load_sensor_series_from_json(
        path=str(data_path),
        column_name=column_name,
    )
    
    # Calculate horizon in steps based on sampling interval (15 minutes = 4 steps per hour)
    if len(series) >= 2:
        sampling_interval_minutes = (series.index[1] - series.index[0]).total_seconds() / 60
        steps_per_hour = 60 / sampling_interval_minutes
        horizon_steps = int(horizon_hours * steps_per_hour)
    else:
        horizon_steps = horizon_hours * 4  # Fallback: assume 15-minute intervals (4 per hour)
    
    logger.info(
        f"forecast_sensor_from_fixture => FORECASTING {horizon_hours} hours = {horizon_steps} steps at {sampling_interval_minutes:.0f}-minute intervals"
    )
    
    # Generate forecast using LLM (with column name for context-aware prompts)
    forecast = llmtime_forecast(
        series=series,
        horizon=horizon_steps,
        provider=provider,
        model=model,
        column_name=column_name,  # Context-aware prompts
    )
    
    return forecast


# Convenience function: Load temperature data from JSON and generate LLM forecast.
def forecast_temperature_from_fixture(
    horizon_hours: int = 24,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> np.ndarray:
    return forecast_sensor_from_fixture(
        column_name="temperature",
        horizon_hours=horizon_hours,
        provider=provider,
        model=model,
    )


# CLI entry point for running forecasts from command line
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Generate LLM forecast for temperature from local JSON data (15-minute intervals)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Number of hours to forecast ahead (default: 24). At 15-min intervals, this equals 4*horizon steps.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["mistral", "groq", "openai"],
        default="groq",
        help="LLM provider to use (default: groq)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name (uses provider default if not specified)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    try:
        print("=" * 60)
        print("LLM Temperature Forecast (15-minute intervals)")
        print("=" * 60)
        print(f"FORECAST HORIZON: {args.horizon} hours ({args.horizon * 4} steps at 15-min intervals). Provider: {args.provider}")
        if args.model:
            print(f"MODEL: {args.model}")
        print()
        print("LOADING data and GENERATING forecast...")
        print()
        
        # Generate forecast
        forecast = forecast_temperature_from_fixture(
            horizon_hours=args.horizon,
            provider=args.provider,
            model=args.model,
        )
        
        # Get last timestamp from data (to note timestamp of last data point, might beneeded for Grafana format)
        data_path = Path(__file__).parent.parent.parent / "data" / "real_sensbee_json_data.json"
        series = load_sensor_series_from_json(
            path=str(data_path),
            column_name="temperature",
        )
        last_timestamp = series.index[-1]
        
        # Print results
        print("=" * 60)
        print("Forecast results")
        print("=" * 60)
        steps_per_hour = 4
        total_steps = len(forecast)
        total_hours = total_steps / steps_per_hour
        
        print(f"\nFORECAST: {total_steps} points at 15-min intervals ({total_hours:.1f} hours)")
        print(f"STARTING FROM: {last_timestamp}")
        print(f"ENDING AT: {last_timestamp + pd.Timedelta(minutes=15 * total_steps)}")
        
        # Show all points grouped by hour
        print(f"\nAll {total_steps} forecasted values (15-min intervals):")
        for hour in range(int(total_hours) + (1 if total_steps % steps_per_hour > 0 else 0)):
            start_idx = hour * steps_per_hour
            end_idx = min((hour + 1) * steps_per_hour, total_steps)
            if start_idx < total_steps:
                print(f"\nHour {hour + 1:2d} (steps {start_idx + 1:3d}-{end_idx:3d}):")
                for step_in_hour in range(end_idx - start_idx):
                    idx = start_idx + step_in_hour
                    timestamp = last_timestamp + pd.Timedelta(minutes=15 * (idx + 1))
                    minute = step_in_hour * 15
                    print(f"  {timestamp.strftime('%H:%M')} (+{minute:2d}min): {forecast[idx]:6.2f}°C")
        
        print()
        print("=" * 60)
        print("Forecast completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nFORECAST INTERRUPTED by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
