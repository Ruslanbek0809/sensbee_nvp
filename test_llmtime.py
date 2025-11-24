# Test script for LLMTime wrapper.

import argparse
from numpy._typing._shape import _AnyShape
from numpy import dtype
import os
import sys
from typing import Any

import pandas as pd

# Add project root to path so we can import from src for src/models/llmtime_wrapper.py
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from src.models.llmtime_wrapper import (
    llmtime_forecast,
)

# Test the full pipeline with actual LLM API (requires API key).
def test_full_with_api(provider: str = None):
    # Auto-detect provider if not specified
    if provider is None:
        if os.getenv("MISTRAL_API_KEY"):
            provider = "mistral"
        elif os.getenv("GROQ_API_KEY"):
            provider = "groq"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = os.getenv("LLM_PROVIDER", "mistral")
    
    provider = provider.lower()
    
    print("=" * 60)
    print(f"Testing full pipeline with {provider.upper()} API")
    print("=" * 60)
    
    # Check for API key based on provider
    api_key = None
    if provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            print("MISTRAL_API_KEY not found in environment variables. You can get a free key at: https://console.mistral.ai/")
            return
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("GROQ_API_KEY not found in environment variables. You can get a free key at: https://console.groq.com/")
            return  
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not found in environment variables. You can get a key at: https://platform.openai.com/")
            return
    else:
        print(f"Unknown provider: {provider}. Supported providers for now: mistral, groq, openai")
        return
    
    # Create test series
    series = pd.Series([20.1, 20.3, 20.7, 21.0, 21.2, 21.5, 21.7])
    horizon = 3
    
    print(f"\nInput series: {series.tolist()}")
    print(f"Horizon: {horizon}")
    print(f"\nCalling {provider.upper()} API (this may take a few seconds)...")
    
    try:
        # Test the full pipeline
        forecast = llmtime_forecast(series, horizon, provider=provider)
        
        print(f"\nForecast generated successfully!")
        print(f"Forecast: {forecast}")
        # print(f"Forecast type: {type[_AnyShape, dtype[Any]](forecast)}")
        # print(f"Forecast shape: {forecast.shape}")
        # print(f"Forecast dtype: {forecast.dtype}")
        
        # Validate
        assert len(forecast) == horizon, f"Expected {horizon} values, got {len(forecast)}"
        assert all(isinstance(x, (int, float)) for x in forecast), "All values should be numeric"
        
        print(f"\nFull pipeline ({provider.upper()} API) test passed!")
        
    except Exception as e:
        print(f"\nError during API call: {e}")
        print("\nThis could be due to: - Invalid API key, - Network issues, - API rate limits, - Insufficient API credits")
        print(f"If provider-specific issues (check {provider} documentation)")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Test LLMTime wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_llmtime.py --test-full --provider mistral   # Test with Mistral (recommended)
  python test_llmtime.py --test-full --provider groq      # Test with Groq
  python test_llmtime.py --test-full --provider openai    # Test with OpenAI
  python test_llmtime.py --test-full                      # Auto-detect provider
        """,
    )
    parser.add_argument("--test-encode", action="store_true", help="Test encoding only")
    parser.add_argument("--test-parse", action="store_true", help="Test parsing only")
    parser.add_argument(
        "--test-full",
        action="store_true",
        help="Test full pipeline with LLM API",
    )
    parser.add_argument(
        "--test-mock", action="store_true", help="Test full pipeline with mock"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["mistral", "groq", "openai"],
        default=None,
        help="LLM provider to use (mistral, groq, or openai). Auto-detects if not specified.",
    )
    
    args = parser.parse_args()
    
    if args.test_full:
        test_full_with_api(provider=args.provider)
    else:
        print("No other parts to test")

if __name__ == "__main__":
    main()

