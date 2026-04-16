# Quick test script for NVP LLMs forecasting. Tests the core forecasting pipeline with local JSON data.

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.models.nvp_llms import forecast_from_json
from src.data_access.data_loader import load_sensor_series_from_json


def test_forecast(
    json_path: str,
    column: str = "temperature",
    provider: str = "groq",
    horizon_hours: int = 6,
    num_forecasts: int = 5,
):
    print("=" * 70)
    print(f"TESTING NVP-LLM FORECAST")
    print("=" * 70)
    print(f"  Data: {json_path}")
    print(f"  Column: {column}")
    print(f"  Provider: {provider}")
    print(f"  Horizon: {horizon_hours} hours")
    print(f"  Forecasts: {num_forecasts}")
    print()
    
    forecast, series = forecast_from_json(
        json_path=json_path,
        column_name=column,
        horizon_hours=horizon_hours,
        provider=provider,
        num_forecasts=num_forecasts,
    )
    
    print(f"Input series: {len(series)} points")
    print(f"  Range: [{series.min():.2f}, {series.max():.2f}]")
    print(f"Forecast: {len(forecast)} points")
    print(f"  Range: [{forecast.min():.2f}, {forecast.max():.2f}]")
    print(f"  First 6 values: {forecast[:6].tolist()}")
    print()
    print("Test PASSED!")
    
    return forecast, series


def main():
    parser = argparse.ArgumentParser(
        description="Quick test for NVP-LLM forecasting pipeline"
    )
    parser.add_argument(
        "--json-path",
        default="data/temp_14day_sensbee_data.json",
        help="Path to JSON data file",
    )
    parser.add_argument(
        "--column",
        default="temperature",
        help="Column to forecast",
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "openai", "mistral", "local"],
        default="groq",
        help="LLM provider",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=6,
        help="Forecast horizon in hours",
    )
    parser.add_argument(
        "--num-forecasts",
        type=int,
        default=5,
        help="Number of independent forecasts",
    )
    
    args = parser.parse_args()
    
    # Check API key for API providers
    key_map = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
    }
    if args.provider in key_map:
        if not os.getenv(key_map[args.provider]):
            print(f"ERROR: {key_map[args.provider]} not set in environment")
            print(f"Set it with: export {key_map[args.provider]}='your-key'")
            sys.exit(1)
    
    # Check if data file exists
    data_path = Path(args.json_path)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Download sensor data or use a different path.")
        sys.exit(1)
    
    try:
        test_forecast(
            json_path=str(data_path),
            column=args.column,
            provider=args.provider,
            horizon_hours=args.horizon,
            num_forecasts=args.num_forecasts,
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
