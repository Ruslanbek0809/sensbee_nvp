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
    encode_series_to_prompt,
    parse_forecast_from_text,
    llmtime_forecast,
)

# Test the encode_series_to_prompt function.
def test_encode():
    print("=" * 60)
    print("Testing encode_series_to_prompt")
    print("=" * 60)
    
    # Test with float values
    series1 = pd.Series([20.1, 20.3, 20.7, 21.0, 21.2])
    prompt1 = encode_series_to_prompt(series1, horizon=3)
    print(f"\nSeries: {series1.tolist()}")
    print(f"Horizon: 3")
    print(f"Prompt:\n{prompt1}\n")
    
    # Test with integer values
    series2 = pd.Series([100, 105, 110, 115, 120])
    prompt2 = encode_series_to_prompt(series2, horizon=5)
    print(f"\nSeries: {series2.tolist()}")
    print(f"Horizon: 5")
    print(f"Prompt:\n{prompt2}\n")
    
    print("✓ encode_series_to_prompt tests passed!")


def test_full_with_mock():
    """Test the full pipeline with a mock response (no API call)."""
    print("=" * 60)
    print("Testing full pipeline (mock)")
    print("=" * 60)
    
    # Create test series
    series = pd.Series([20.1, 20.3, 20.7, 21.0, 21.2, 21.5])
    horizon = 3
    
    print(f"\nInput series: {series.tolist()}")
    print(f"Horizon: {horizon}")
    
    # Step 1: Encode
    prompt = encode_series_to_prompt(series, horizon)
    print(f"\nGenerated prompt:\n{prompt}")
    
    # Step 2: Simulate API response
    mock_response = "21.8, 22.0, 22.2"
    print(f"\nMock API response: {mock_response}")
    
    # Step 3: Parse
    forecast = parse_forecast_from_text(mock_response, horizon)
    print(f"\nParsed forecast: {forecast}")
    # print(f"Forecast type: {type(forecast)}")
    # print(f"Forecast shape: {forecast.shape}")
    
    print("\n✓ Full pipeline (mock) test passed!")


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
            print("⚠️ MISTRAL_API_KEY not found in environment variables. You can get a free key at: https://console.mistral.ai/")
            return
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("⚠️ GROQ_API_KEY not found in environment variables. You can get a free key at: https://console.groq.com/")
            return  
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ OPENAI_API_KEY not found in environment variables. You can get a key at: https://platform.openai.com/")
            return
    else:
        print(f"❌ Unknown provider: {provider}. Supported providers for now: mistral, groq, openai")
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
        
        print(f"\n✓ Forecast generated successfully!")
        print(f"Forecast: {forecast}")
        # print(f"Forecast type: {type[_AnyShape, dtype[Any]](forecast)}")
        # print(f"Forecast shape: {forecast.shape}")
        # print(f"Forecast dtype: {forecast.dtype}")
        
        # Validate
        assert len(forecast) == horizon, f"Expected {horizon} values, got {len(forecast)}"
        assert all(isinstance(x, (int, float)) for x in forecast), "All values should be numeric"
        
        print(f"\n✓ Full pipeline ({provider.upper()} API) test passed!")
        
    except Exception as e:
        print(f"\n❌ Error during API call: {e}")
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
    
    if args.test_encode:
        test_encode()
    elif args.test_full:
        test_full_with_api(provider=args.provider)
    elif args.test_mock:
        test_full_with_mock()
    else:
        # Run all tests except API test by default
        print("Running all tests (except API test)...")
        test_encode()
        print()
        test_full_with_mock()

if __name__ == "__main__":
    main()

