# Local LLM implementation for time series forecasting.

import logging
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Global cache for loaded models (avoids reloading for each request)
_loaded_models = {}

# Configuration for local LLM models.
@dataclass
class LocalModelConfig:
    model_id: str  # HuggingFace model ID
    model_type: str  # "llama2" or "mistral"
    device_map: str = "auto"  # "auto", "cuda", "cpu"
    torch_dtype: str = "float16"  # "float16", "float32", "bfloat16"
    load_in_4bit: bool = False  # Use 4-bit quantization (reduces VRAM)
    load_in_8bit: bool = False  # Use 8-bit quantization


# Pre-defined model configurations
LOCAL_MODELS = {
    # LLaMA-2 models (require HuggingFace access)
    "llama2-7b": LocalModelConfig(
        model_id="meta-llama/Llama-2-7b-hf",
        model_type="llama2",
    ),
    "llama2-7b-chat": LocalModelConfig(
        model_id="meta-llama/Llama-2-7b-chat-hf",
        model_type="llama2",
    ),
    "llama2-13b": LocalModelConfig(
        model_id="meta-llama/Llama-2-13b-hf",
        model_type="llama2",
    ),
    "llama2-70b": LocalModelConfig(
        model_id="meta-llama/Llama-2-70b-hf",
        model_type="llama2",
        load_in_4bit=True,  # 70B needs quantization
    ),
    # Mistral models (no access required)
    "mistral-7b": LocalModelConfig(
        model_id="mistralai/Mistral-7B-v0.1",
        model_type="mistral",
    ),
    "mistral-7b-instruct": LocalModelConfig(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_type="mistral",
    ),
}


def check_gpu_availability() -> dict:
    """Check GPU availability (CUDA for NVIDIA, MPS for Apple Silicon)."""
    try:
        import torch
        
        # Checks for CUDA (NVIDIA GPUs)
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
        # Checks for MPS (Apple Silicon - Mac M1/M2/M3)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't expose VRAM info, but we know it's available
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
        else:
            return {"available": False, "reason": "No GPU backend available (CUDA/MPS)"}
    except ImportError:
        return {"available": False, "reason": "PyTorch not installed"}


# Loads a local LLM model and tokenizer.
def load_local_model(model_name: str, cache: bool = True):
    # Checks cache first
    if cache and model_name in _loaded_models:
        logger.info(f"Using cached model: {model_name}")
        return _loaded_models[model_name]
    
    # Imports here to avoid loading torch if not using local models
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
    
    # Detects available device backend
    use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    
    if use_mps:
        logger.info("USING APPLE SILICON GPU (MPS)")
        device_backend = "mps"
    elif use_cuda:
        logger.info("USING NVIDIA GPU (CUDA)")
        device_backend = "cuda"
    else:
        logger.warning("NO GPU DETECTED, USING CPU (WILL BE SLOW)")
        device_backend = "cpu"
    
    # Gets model config
    if model_name in LOCAL_MODELS:
        config = LOCAL_MODELS[model_name]
    else:
        # Assumes it's a HuggingFace model ID
        config = LocalModelConfig(model_id=model_name, model_type="auto")
    
    logger.info(f"LOADING LOCAL MODEL: {config.model_id}")
    
    # Configures quantization (ONLY for CUDA, not MPS)
    quantization_config = None
    if device_backend == "cuda":
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("USING 4-BIT QUANTIZATION (CUDA)")
        elif config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("USING 8-BIT QUANTIZATION (CUDA)")
    elif device_backend == "mps" and (config.load_in_4bit or config.load_in_8bit):
        logger.warning("QUANTIZATION NOT SUPPORTED ON MPS, LOADING FULL PRECISION MODEL")
    
    # Determines torch dtype (MPS works best with float32 or bfloat16)
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if device_backend == "mps":
        # MPS doesn't support float16 well, uses float32
        torch_dtype = torch.float32
        logger.info("USING FLOAT32 FOR MPS COMPATIBILITY")
    else:
        torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)
    
    # Determines device_map for model loading
    if device_backend == "mps":
        # For MPS, loads on CPU first then moves to MPS
        device_map = None
    elif device_backend == "cuda":
        device_map = config.device_map  # "auto" works well for CUDA
    else:
        device_map = None  # CPU
    
    # Loads model based on type
    if config.model_type == "llama2":
        tokenizer = LlamaTokenizer.from_pretrained(
            config.model_id,
            use_fast=False,
            trust_remote_code=True,
        )
        model = LlamaForCausalLM.from_pretrained(
            config.model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
    else:
        # Generic loads for Mistral and other models
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
    
    # Moves model to MPS device if using MPS
    if device_backend == "mps":
        model = model.to("mps")
        logger.info("MODEL MOVED TO MPS DEVICE")
    
    # Sets up special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Sets model to eval mode
    model.eval()
    
    logger.info(f"MODEL LOADED: {config.model_id} ON {device_backend}")
    
    # Caches if requested
    if cache:
        _loaded_models[model_name] = (model, tokenizer)
    
    return model, tokenizer


# Gets list of allowed token IDs for time series generation.
def get_allowed_tokens(tokenizer, time_sep: str = ", ") -> List[int]:
    allowed_chars = list("0123456789.-") + [time_sep]
    
    allowed_ids = set()
    for char in allowed_chars:
        # Tries different tokenization methods
        tokens = tokenizer.encode(char, add_special_tokens=False)
        allowed_ids.update(tokens)
    
    # Also adds space token if present
    if hasattr(tokenizer, 'sp_model'):
        space_id = tokenizer.encode(" ", add_special_tokens=False)
        allowed_ids.update(space_id)
    
    return list(allowed_ids)


# Gets bad_words_ids for model.generate() to block non-numeric tokens.
def get_bad_words_ids(tokenizer, time_sep: str = ", ") -> List[List[int]]:
    allowed_ids = set(get_allowed_tokens(tokenizer, time_sep))
    
    # All other tokens are "BAD"
    bad_ids = []
    for token_id in range(len(tokenizer)):
        if token_id not in allowed_ids:
            bad_ids.append([token_id])
    
    return bad_ids


# Generates time series continuations using local LLM with token control.   
def local_llm_completion(
    input_str: str,
    model_name: str = "llama2-7b",
    steps: int = 96,
    time_sep: str = ", ",
    num_samples: int = 5,
    batch_size: int = 5,
    temperature: float = 0.9,
    top_p: float = 0.9,
    cache_model: bool = True,
) -> List[str]:
    import torch
    
    # Loads model and tokenizer
    model, tokenizer = load_local_model(model_name, cache=cache_model)
    
    # Calculates average tokens per time step for output length estimation
    input_tokens = tokenizer.encode(input_str, add_special_tokens=False)
    num_steps_in_input = len(input_str.split(time_sep))
    avg_tokens_per_step = len(input_tokens) / max(num_steps_in_input, 1)
    max_new_tokens = int(avg_tokens_per_step * steps * 1.2)  # 20% buffer
    
    logger.info(
        f"LOCAL LLM GENERATION: STEPS={steps}, MAX TOKENS={max_new_tokens}, "
        f"SAMPLES={num_samples}, TEMPERATURE={temperature}"
    )
    
    # Gets bad_words_ids to restrict output to numeric tokens only
    bad_words_ids = get_bad_words_ids(tokenizer, time_sep)
    logger.info(f"BLOCKING {len(bad_words_ids)} NON-NUMERIC TOKENS")
    
    # Determines DEVICES
    device = next(model.parameters()).device
    
    # Generates samples in batches
    all_completions = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - len(all_completions))
        
        # Tokenizes input
        inputs = tokenizer(
            [input_str] * current_batch_size,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        num_input_tokens = inputs['input_ids'].shape[1]
        
        # Generates with token control
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
        
        # Decodes only the generated part (not the input)
        generated_ids = outputs[:, num_input_tokens:]
        completions = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        all_completions.extend(completions)
        logger.debug(f"BATCH {batch_idx + 1}/{num_batches}: GENERATED {len(completions)} SAMPLES")
    
    return all_completions[:num_samples]


# Checks if local LLM inference is available (PyTorch + GPU/CUDA/MPS).
def is_local_llm_available() -> bool:
    try:
        import torch
        # Checks for CUDA (NVIDIA) or MPS (Apple Silicon GPUs)
        return torch.cuda.is_available() or (
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        )
    except ImportError:
        return False
