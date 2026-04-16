# Local LLM implementation for time series forecasting with token control.
# Adapted from LLMTime for local Llama/Mistral models.

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Global cache for loaded models to avoid reloading between requests.
# Keyed on (model_name, device_str) so multi-GPU mode caches one instance per GPU.
_model_cache: Dict[Tuple[str, str], Tuple] = {}


# Configuration for a local LLM model.
@dataclass
class LocalModelConfig:
    model_id: str
    model_type: str  # "llama2", "mistral", or "auto"
    device_map: str = "auto"
    torch_dtype: str = "float16"
    load_in_4bit: bool = False
    load_in_8bit: bool = False


# Pre-defined model configurations for local LLMs.
LOCAL_MODELS = {
    # LLaMA-2 models. Requires HuggingFace access token.
    "llama2-7b": LocalModelConfig(
        model_id="meta-llama/Llama-2-7b-hf",
        model_type="auto",
    ),
    "llama2-7b-chat": LocalModelConfig(
        model_id="meta-llama/Llama-2-7b-chat-hf",
        model_type="auto",
    ),
    "llama2-13b": LocalModelConfig(
        model_id="meta-llama/Llama-2-13b-hf",
        model_type="auto",
    ),
    "llama2-70b": LocalModelConfig(
        model_id="meta-llama/Llama-2-70b-hf",
        model_type="auto",
        load_in_4bit=True,  # 70B needs quantization
    ),
    # Mistral models. No special access required.
    "mistral-7b": LocalModelConfig(
        model_id="mistralai/Mistral-7B-v0.1",
        model_type="auto",
    ),
    "mistral-7b-instruct": LocalModelConfig(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_type="auto",
    ),
}


# Checks GPU availability (CUDA for NVIDIA, MPS for Apple Silicon).
def check_gpu_availability() -> dict:
    try:
        import torch
        
        # Check CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            total_vram = 0
            devices = []
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                total_vram += vram_gb
                devices.append({
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(vram_gb, 2),
                    "backend": "cuda",
                })
            
            return {
                "available": True,
                "backend": "cuda",
                "device_count": gpu_count,
                "total_vram_gb": round(total_vram, 2),
                "devices": devices,
            }
        
        # Check MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return {
                "available": True,
                "backend": "mps",
                "device_count": 1,
                "devices": [{
                    "index": 0,
                    "name": "Apple Silicon GPU",
                    "backend": "mps",
                }],
            }
        
        return {"available": False, "reason": "No GPU backend (CUDA/MPS) available"}
        
    except ImportError:
        return {"available": False, "reason": "PyTorch not installed"}


# Checks if local LLM inference is available.
def is_local_llm_available() -> bool:
    gpu_info = check_gpu_availability()
    return gpu_info.get("available", False)


# Determines the best available device backend.
def get_device_backend() -> str:
    try:
        import torch
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
        
    except ImportError:
        return "cpu"


# Loads a local LLM model and tokenizer.
# device: e.g. "cuda:0", "cuda:1". If None, uses device_map="auto" (model decides).
def load_local_model(
    model_name: str,
    cache: bool = True,
    device: Optional[str] = None,
):
    cache_key = (model_name, device or "auto")
    if cache and cache_key in _model_cache:
        logger.debug(f"Using cached model: {model_name} on {device or 'auto'}")
        return _model_cache[cache_key]
    
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            LlamaForCausalLM,
            LlamaTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as e:
        raise ImportError(
            f"LOCAL LLM REQUIRES: pip install torch transformers accelerate\n"
            f"For quantization (NVIDIA only): pip install bitsandbytes\n"
            f"ERROR: {e}"
        )
    
    # Checks for offline mode (HPC clusters without internet)
    offline_mode = (
        os.getenv("HF_HUB_OFFLINE", "0") == "1" or
        os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
    )
    if offline_mode:
        logger.info("OFFLINE MODE ENABLED. USING CACHED MODELS ONLY.")
    
    # Determines device backend
    device_backend = get_device_backend()
    logger.info(f"USING DEVICE BACKEND: {device_backend}")
    
    # Gets model config
    if model_name in LOCAL_MODELS:
        config = LOCAL_MODELS[model_name]
    else:
        # Assumes it's a HuggingFace model ID
        config = LocalModelConfig(model_id=model_name, model_type="auto")
    
    logger.info(f"LOADING MODEL: {config.model_id}")
    
    # Configures quantization (CUDA only)
    quantization_config = None
    if device_backend == "cuda":
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("USING 4-BIT QUANTIZATION")
        elif config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("USING 8-BIT QUANTIZATION")
    elif device_backend == "mps" and (config.load_in_4bit or config.load_in_8bit):
        logger.warning("QUANTIZATION NOT SUPPORTED ON MPS, LOADING FULL PRECISION MODEL")
    
    # Determines torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if device_backend == "mps":
        torch_dtype = torch.float32  # MPS works best with float32 for better performance
    else:
        torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)
    
    # Determines device_map
    # If a specific device is requested (e.g. "cuda:1"), pin the model there.
    if device is not None and device_backend == "cuda":
        device_map = device
    elif device_backend == "mps":
        device_map = None  # Load on CPU first, move to MPS later
    elif device_backend == "cuda":
        device_map = config.device_map
    else:
        device_map = None
    
    # Loads tokenizer and model
    try:
        hf_cache = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME")
        cache_kwargs = {"cache_dir": hf_cache} if hf_cache else {}

        if config.model_type == "llama2":
            tokenizer = LlamaTokenizer.from_pretrained(
                config.model_id,
                use_fast=False,
                trust_remote_code=True,
                local_files_only=offline_mode,
                **cache_kwargs,
            )
            model = LlamaForCausalLM.from_pretrained(
                config.model_id,
                device_map=device_map,
                dtype=torch_dtype,
                quantization_config=quantization_config,
                trust_remote_code=True,
                local_files_only=offline_mode,
                **cache_kwargs,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id,
                trust_remote_code=True,
                local_files_only=offline_mode,
                **cache_kwargs,
            )
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                device_map=device_map,
                dtype=torch_dtype,
                quantization_config=quantization_config,
                trust_remote_code=True,
                local_files_only=offline_mode,
                **cache_kwargs,
            )
    except Exception as e:
        if offline_mode:
            cache_path = hf_cache or "default (~/.cache/huggingface)"
            raise RuntimeError(
                f"FAILED TO LOAD MODEL IN OFFLINE MODE. "
                f"ENSURE MODEL IS CACHED: {config.model_id}\n"
                f"CACHE DIR: {cache_path}\n"
                f"DEVICE MAP: {device_map}\n"
                f"ERROR TYPE: {type(e).__name__}\n"
                f"ERROR: {e}"
            )
        raise
    
    # Moves model to MPS if using Apple Silicon
    if device_backend == "mps":
        model = model.to("mps")
        logger.info("MODEL MOVED TO MPS DEVICE")
    
    # Sets up special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Sets model to eval mode
    model.eval()
    
    actual_device = device or device_backend
    logger.info(f"MODEL LOADED SUCCESSFULLY ON {actual_device}")

    if cache:
        _model_cache[(model_name, device or "auto")] = (model, tokenizer)

    return model, tokenizer


# Gets list of allowed token IDs for time series generation.
def get_allowed_token_ids(
    tokenizer,
    time_sep: str = ",",
) -> List[int]:
    # Characters allowed in time series output
    allowed_chars = list("0123456789")
    allowed_chars.extend([time_sep, "-", " "])
    
    allowed_ids = set()
    for char in allowed_chars:
        try:
            tokens = tokenizer.encode(char, add_special_tokens=False)
            allowed_ids.update(tokens)
        except Exception:
            pass
    
    return list(allowed_ids)


# Gets bad_words_ids to block non-numeric tokens during generation. 
# This is CRITICAL for reliable time series output. Without this, models often generate explanatory text instead of just numbers.
def get_bad_words_ids(
    tokenizer,
    time_sep: str = ",",
) -> List[List[int]]:
    allowed_ids = set(get_allowed_token_ids(tokenizer, time_sep))
    
    # Blocks all tokens not in allowed set
    bad_ids = []
    vocab_size = len(tokenizer)
    
    for token_id in range(vocab_size):
        if token_id not in allowed_ids:
            bad_ids.append([token_id])
    
    logger.debug(f"BLOCKING {len(bad_ids)}/{vocab_size} NON-NUMERIC TOKENS")
    return bad_ids


# Returns list of GPU indices that have at least `min_free_gb` of free VRAM.
def _get_usable_gpus(min_free_gb: float = 15.0) -> List[int]:
    import torch
    if not torch.cuda.is_available():
        return []
    usable = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        if free_gb >= min_free_gb:
            usable.append(i)
            logger.debug(f"GPU {i}: {free_gb:.1f}/{total_gb:.1f} GiB free — usable")
        else:
            logger.info(f"GPU {i}: {free_gb:.1f}/{total_gb:.1f} GiB free — skipping (need {min_free_gb} GiB)")
    return usable


# Multi-GPU behaviour (automatic):
#   A 7B model in float16 fits entirely in ~14 GB of VRAM. With 4× A100s
#   (4×40 GB = 160 GB total) device_map="auto" puts the whole model on GPU 0
#   and never touches the rest. This function detects the GPU count at runtime.
#   When > 1 GPU is available it loads one model copy per GPU and generates
#   each path on a separate device in parallel threads — close to N_gpu speedup.
#   When only 1 GPU (or CPU) is available the single-device batched path is used.
#   GPUs occupied by other processes (< 15 GiB free) are automatically skipped.
def local_llm_completion(
    input_str: str,
    settings,  # SerializerSettings from serialize.py
    model_name: str = "llama2-7b",
    steps: int = 96,
    num_forecasts: int = 5,
    batch_size: int = 5,
    temperature: float = 0.9,
    top_p: float = 0.9,
    cache_model: bool = True,
) -> List[str]:
    import torch

    usable_gpus = _get_usable_gpus(min_free_gb=15.0)

    if len(usable_gpus) > 1 and num_forecasts > 1:
        try:
            return _generate_forecasts_multi_gpu(
                input_str=input_str,
                settings=settings,
                model_name=model_name,
                steps=steps,
                num_forecasts=num_forecasts,
                temperature=temperature,
                top_p=top_p,
                cache_model=cache_model,
                gpu_ids=usable_gpus,
            )
        except Exception as e:
            logger.warning(
                f"Multi-GPU failed ({e}), falling back to single device"
            )

    return _generate_forecasts_single_device(
        input_str=input_str,
        settings=settings,
        model_name=model_name,
        steps=steps,
        num_forecasts=num_forecasts,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        cache_model=cache_model,
        preferred_gpu=usable_gpus[0] if usable_gpus else None,
    )


def _generate_forecasts_single_device(
    input_str: str,
    settings,
    model_name: str,
    steps: int,
    num_forecasts: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    cache_model: bool,
    preferred_gpu: Optional[int] = None,
) -> List[str]:
    import torch

    device_arg = f"cuda:{preferred_gpu}" if preferred_gpu is not None else None
    model, tokenizer = load_local_model(model_name, cache=cache_model, device=device_arg)

    input_tokens = tokenizer.encode(input_str, add_special_tokens=False)
    num_input_steps = len(input_str.split(settings.time_sep))
    avg_tokens_per_step = len(input_tokens) / max(num_input_steps, 1)
    max_new_tokens = int(avg_tokens_per_step * steps * 1.3)  # 30% buffer
    
    logger.info(
        f"LOCAL LLM (single device): STEPS={steps}, MAX_TOKENS={max_new_tokens}, "
        f"FORECASTS={num_forecasts}, TEMPERATURE={temperature}"
    )
    
    # Gets bad_words_ids for token control
    bad_words_ids = get_bad_words_ids(tokenizer, settings.time_sep)
    logger.info(f"BLOCKING {len(bad_words_ids)} NON-NUMERIC TOKENS")
    
    # Determines device
    device = next(model.parameters()).device

     # Generates completions in batches
    all_completions: List[str] = []
    num_batches = (num_forecasts + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):

        # Tokenizes input
        current_batch_size = min(batch_size, num_forecasts - len(all_completions))

        inputs = tokenizer(
            [input_str] * current_batch_size,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Gets number of input tokens
        num_input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                bad_words_ids=bad_words_ids,
                renormalize_logits=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[:, num_input_tokens:]
        completions = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        all_completions.extend(completions)
        logger.debug(f"BATCH {batch_idx + 1}/{num_batches}: {len(completions)} forecasts")

    return all_completions[:num_forecasts]


def _generate_forecasts_multi_gpu(
    input_str: str,
    settings,
    model_name: str,
    steps: int,
    num_forecasts: int,
    temperature: float,
    top_p: float,
    cache_model: bool,
    gpu_ids: List[int],
) -> List[str]:
    import torch
    from concurrent.futures import ThreadPoolExecutor

    n_gpus = len(gpu_ids)
    logger.info(
        f"LOCAL LLM (multi-GPU): {num_forecasts} FORECASTS ACROSS GPUs {gpu_ids}"
    )

    base, remainder = divmod(num_forecasts, n_gpus)
    paths_per_gpu = [base + (1 if i < remainder else 0) for i in range(n_gpus)]

    # Pre-load models SEQUENTIALLY to avoid HuggingFace cache lock
    # contention.  Once cached in _model_cache, parallel inference is safe.
    for idx, gpu_id in enumerate(gpu_ids):
        if paths_per_gpu[idx] > 0:
            device_str = f"cuda:{gpu_id}"
            cache_key = (model_name, device_str)
            if cache_key not in _model_cache:
                logger.info(f"Pre-loading {model_name} onto {device_str} ...")
                load_local_model(model_name, cache=cache_model, device=device_str)

    def generate_on_gpu(gpu_id: int, n_paths: int) -> List[str]:
        device_str = f"cuda:{gpu_id}"
        model, tokenizer = load_local_model(
            model_name, cache=cache_model, device=device_str
        )

        input_tokens = tokenizer.encode(input_str, add_special_tokens=False)
        num_input_steps = len(input_str.split(settings.time_sep))
        avg_tokens = len(input_tokens) / max(num_input_steps, 1)
        max_new_tokens = int(avg_tokens * steps * 1.3)

        bad_words_ids = get_bad_words_ids(tokenizer, settings.time_sep)
        inputs = tokenizer([input_str] * n_paths, return_tensors="pt", padding=True)
        inputs = {k: v.to(device_str) for k, v in inputs.items()}
        num_input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                bad_words_ids=bad_words_ids,
                renormalize_logits=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[:, num_input_tokens:]
        completions = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        logger.info(f"GPU {gpu_id}: {len(completions)} paths done")
        return completions

    all_completions: List[str] = []
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = [
            executor.submit(generate_on_gpu, gpu_ids[idx], n)
            for idx, n in enumerate(paths_per_gpu)
            if n > 0
        ]
        for future in futures:
            all_completions.extend(future.result())

    return all_completions[:num_forecasts]


# Computes negative log-likelihood of target given input.
def compute_nll(
    model,
    tokenizer,
    input_str: str,
    target_str: str,
    settings,
    transform_fn=None,
) -> float:
    import torch
    import numpy as np
    
    full_series = input_str + target_str
    
    # Tokenizes input
    batch = tokenizer(
        [full_series],
        return_tensors="pt",
        add_special_tokens=True,
    )
    
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Performs forward pass
    with torch.no_grad():
        outputs = model(**batch)
    
    # Gets allowed tokens and masks others
    allowed_ids = set(get_allowed_token_ids(tokenizer, settings.time_sep))
    bad_tokens = [i for i in range(len(tokenizer)) if i not in allowed_ids]
    outputs['logits'][:, :, bad_tokens] = -100
    
    # Computes log probabilities
    input_ids = batch['input_ids'][0][1:]  # Skips BOS token
    logprobs = torch.nn.functional.log_softmax(outputs['logits'], dim=-1)[0][:-1]
    logprobs = logprobs[torch.arange(len(input_ids)), input_ids].cpu().numpy()
    
    # Finds where target starts
    input_len = len(tokenizer([input_str], return_tensors="pt")['input_ids'][0])
    input_len = input_len - 2  # Removes special tokens
    
    # Computes NLL on target portion
    target_logprobs = logprobs[input_len:]
    num_target_steps = len(target_str.split(settings.time_sep)) - 1
    
    if num_target_steps == 0:
        return float('inf')
    
    bpd = -target_logprobs.sum() / num_target_steps
    
    # Converts to continuous NLL
    # log p(x) = log p(token) - log bin_width = log p(token) + prec * log(base)
    continuous_nll = bpd - settings.prec * np.log(settings.base)
    
    return float(continuous_nll)


# Clears the model cache to free GPU memory.
def clear_model_cache():
    global _model_cache
    
    for name in list(_model_cache.keys()):
        del _model_cache[name]
    
    _model_cache = {}
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("MODEL CACHE CLEARED")
    except ImportError:
        pass
