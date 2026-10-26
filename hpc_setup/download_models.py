#!/usr/bin/env python3
# Downloads and caches models on the login node, so the GPU nodes can use them offline.
# Run this on makalu48:
#   python download_models.py --all
#   python download_models.py --model mistral-7b
# --pinned (benchmark harness, Phase 6a) downloads the pinned revisions of benchmark/models/__init__.py instead, each
# file checked against the Hub's SHA-256, into the layout each loader reads:
# - the LLMTime models (local_llm.py passes cache_dir=$HF_HOME) into $HF_HOME/models--...;
# - the foundation models (from_pretrained with the default cache) into $HF_HOME/hub/models--....
#   python hpc_setup/download_models.py --pinned chronos2 mistral7b llama2_7b
# Llama 2 is gated: accept its licence on Hugging Face and log in (huggingface-cli login) first.

import argparse
import os
import sys
from pathlib import Path

# Put the project root on the import path
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


# Checks whether the internet is reachable
def check_internet():
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5)
        return True
    except OSError:
        return False


# Pinned weights for --pinned: name → (model id and revision, whether local_llm.py loads it).
def pinned_models() -> dict:
    from benchmark.models import CHRONOS2, CHRONOS_BOLT_BASE, LLAMA2_7B, MISTRAL_7B, TIMESFM25, TIREX, TOTO
    return {"chronos2": (CHRONOS2, False), "chronos_bolt_base": (CHRONOS_BOLT_BASE, False),
            "timesfm25": (TIMESFM25, False), "tirex": (TIREX, False), "toto": (TOTO, False),
            "mistral7b": (MISTRAL_7B, True), "llama2_7b": (LLAMA2_7B, True)}


# Pickle-format weights the LLM repos ship next to their safetensors (Mistral-7B-v0.1: 15.0 GB of .bin at the pinned
# revision, HF API 2026-10-26). transformers loads the safetensors, so these are skipped.
LLM_SKIPPED = ["*.bin", "*.pt", "*.pth"]


# Downloads one pinned revision and checks every downloaded LFS file's SHA-256 against the Hub's listing. Returns True
# if all files match.
def download_pinned(name: str, pin: tuple, llm: bool, hf_home: str) -> bool:
    import fnmatch
    import hashlib
    from huggingface_hub import HfApi, snapshot_download
    model_id, revision = pin
    skipped = LLM_SKIPPED if llm else []
    print(f"\nDOWNLOADING PINNED {name}: {model_id} @ {revision}")
    folder = Path(snapshot_download(model_id, revision=revision, cache_dir=hf_home if llm else None,
                                    ignore_patterns=skipped or None))
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    ok = True
    for sibling in info.siblings:
        if sibling.lfs is None or any(fnmatch.fnmatch(sibling.rfilename, p) for p in skipped):
            continue
        digest = hashlib.sha256()
        with open(folder / sibling.rfilename, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        same = digest.hexdigest() == sibling.lfs.sha256
        ok &= same
        print(f"  {sibling.rfilename}: SHA-256 {'OK' if same else 'MISMATCH'}")
    return ok


# Downloads one model and caches it
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
        "--pinned",
        nargs="+",
        help="pinned benchmark models (chronos2, chronos_bolt_base, timesfm25, tirex, toto, mistral7b, llama2_7b)"
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
    
    if args.pinned:
        pins = pinned_models()
        unknown = [n for n in args.pinned if n not in pins]
        if unknown:
            print(f"ERROR: UNKNOWN PINNED MODELS {unknown}; known: {list(pins)}")
            sys.exit(1)
        results = {n: download_pinned(n, *pins[n], hf_home) for n in args.pinned}
        print(f"\nPINNED DOWNLOADS: {results}")
        sys.exit(0 if all(results.values()) else 1)

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
