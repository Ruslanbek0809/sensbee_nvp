#!/usr/bin/env python3
# Download and cache models on login node for offline use on GPU nodes. Run this on makalu48. python download_models.py --all or python download_models.py --model mistral-7b

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Models to download
MODELS_TO_DOWNLOAD = {
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    # "llama2-70b": "meta-llama/Llama-2-70b-hf",
}

# Default models to download
DEFAULT_MODELS = ["mistral-7b", "mistral-7b-instruct", "llama2-7b", "llama2-7b-chat"]


# Check if internet is available
def check_internet():
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5)
        return True
    except OSError:
        return False


# Download and cache a model
def download_model(model_name: str, model_id: str, hf_home: str):
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"Cache: {hf_home}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Download tokenizer
        print("\nDownloading TOKENIZER...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=hf_home,
        )
        print(f"TOKENIZER DOWNLOADED (vocab size: {len(tokenizer)})")
        
        # Download model (this is the large download)
        print("\nDownloading MODEL ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=hf_home,
            low_cpu_mem_usage=True,  # Don't load into RAM
        )
        
        # Get model size
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ MODEL WEIGHTS DOWNLOADED ({param_count / 1e9:.2f}B parameters)")
        
        # Clean up to free memory
        del model
        del tokenizer
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED TO DOWNLOAD {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and cache models for offline use"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS_TO_DOWNLOAD.keys()),
        help="Specific model to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all supported models"
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default=None,
        help="HuggingFace cache directory"
    )
    
    args = parser.parse_args()
    
    # Set HF cache directory
    work_dir = os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', 'user')}")
    hf_home = args.hf_home or os.environ.get("HF_HOME", f"{work_dir}/sensbee_nvp/hf_cache")
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    
    print("MODEL DOWNLOAD SCRIPT")
    print(f"HF Cache: {hf_home}")
    
    # Check internet
    if not check_internet():
        print("\nERROR: NO INTERNET CONNECTION")
        sys.exit(1)
    print("INTERNET CONNECTION AVAILABLE")
    
    # Determine which models to download
    if args.all:
        models = list(MODELS_TO_DOWNLOAD.keys())
    elif args.model:
        models = [args.model]
    else:
        models = DEFAULT_MODELS
    
    print(f"\nMODELS TO DOWNLOAD: {models}")
    
    # Download models
    results = {}
    for model_name in models:
        model_id = MODELS_TO_DOWNLOAD[model_name]
        results[model_name] = download_model(model_name, model_id, hf_home)
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    
    success_count = 0
    for model_name, success in results.items():
        status = "DOWNLOADED" if success else "FAILED"
        print(f"  {model_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\nSUCCESSFULLY DOWNLOADED: {success_count}/{len(results)} models")
    print(f"CACHE LOCATION: {hf_home}")
    
    if success_count == len(results):
        print("\nALL DOWNLOADS COMPLETE!")
    else:
        print("\nSOME DOWNLOADS FAILED.")
        sys.exit(1) 


if __name__ == "__main__":
    main()
