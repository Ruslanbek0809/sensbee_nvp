# Test script for LLMTime wrapper.

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.models.llmtime_wrapper import (
    forecast_sensor_from_api,
    forecast_sensor_from_local_json,
)


# Test forecast from SensBee API.
def test_api(provider: str = "mistral", horizon: int = 6):
    print("=" * 60)
    print(f"TESTING API source with {provider.upper()}")
    print("=" * 60)
    
    forecast, series = forecast_sensor_from_api(
        horizon_hours=horizon,
        provider=provider,
    )
    
    print(f"Input: {len(series)} points")
    print(f"Forecast: {len(forecast)} points")
    print(f"Values: {forecast[:6].tolist()}...")
    print("API test PASSED!")


# Test forecast from local JSON file.
def test_local(provider: str = "mistral", horizon: int = 6):
    print("=" * 60)
    print(f"TESTING LOCAL source with {provider.upper()}")
    print("=" * 60)
    
    forecast, series = forecast_sensor_from_local_json(
        horizon_hours=horizon,
        provider=provider,
    )
    
    print(f"Input: {len(series)} points")
    print(f"Forecast: {len(forecast)} points")
    print(f"Values: {forecast[:6].tolist()}...")
    print("Local test PASSED!")


def main():
    parser = argparse.ArgumentParser(description="Test LLMTime wrapper")
    parser.add_argument(
        "--source",
        choices=["api", "local", "both"],
        default="api",
        help="Data source to test",
    )
    parser.add_argument(
        "--provider",
        choices=["mistral", "groq", "openai"],
        default="mistral",
        help="LLM provider",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=6,
        help="Forecast horizon in hours",
    )
    
    args = parser.parse_args()
    
    # Check API key
    key_map = {
        "mistral": "MISTRAL_API_KEY",
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    if not os.getenv(key_map[args.provider]):
        print(f"ERROR: {key_map[args.provider]} not set in environment")
        sys.exit(1)
    
    try:
        if args.source in ("api", "both"):
            test_api(args.provider, args.horizon)
        if args.source in ("local", "both"):
            test_local(args.provider, args.horizon)
        print("\nAll tests PASSED!")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
