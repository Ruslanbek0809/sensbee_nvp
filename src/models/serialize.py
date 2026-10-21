# Turns time series into text for an LLM and back. Adapted from LLMTime (Gruver et al., NeurIPS 2023).
#
# The LLMTime approach:
#   1. Normalize the values with a quantile scaler (α=0.95, β=0.3).
#   2. Write them as comma-separated integers. Dropping the decimal point saves tokens.
#   3. GPT models need spaces between digits, so BPE does not merge the digits into one token.
#   4. LLaMA and Mistral models already tokenize each character on its own.
#
# The quantile scaler:
#   - Shifts the minimum down by β × range, to leave room for lower predictions.
#   - Scales the values so the α-percentile equals 1.0.
#   - The LLM then sees the shape of the data, not the raw units.

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Model families, used to pick the tokenization strategy.
class ModelFamily(Enum):
    GPT = "gpt"           # OpenAI GPT models. Need spaces between digits.
    LLAMA = "llama"       # Meta LLaMA family. One token per digit.
    MISTRAL = "mistral"   # Mistral models.
    UNKNOWN = "unknown"   # Fallback for any other model family.

# Holds the scaling functions and their parameters. LLMTime-style: quantile scaling plus a shift.
@dataclass
class Scaler:
    transform: Callable[[np.ndarray], np.ndarray] = field(default=lambda x: x)
    inverse_transform: Callable[[np.ndarray], np.ndarray] = field(default=lambda x: x)
    original_min: float = 0.0
    original_max: float = 1.0
    shift: float = 0.0
    scale: float = 1.0


# Builds a scaler from the history, following LLMTime.
def create_scaler(
    history: np.ndarray,
    alpha: float = 0.95,
    beta: float = 0.3,
    use_basic: bool = False,
) -> Scaler:
    # Drop NaN values before computing statistics
    clean_history = history[~np.isnan(history)]
    
    if len(clean_history) == 0:
        logger.warning("EMPTY HISTORY AFTER REMOVING NaN, USING IDENTITY SCALER")
        return Scaler()
    
    original_min = float(np.min(clean_history))
    original_max = float(np.max(clean_history))
    
    if use_basic:
        # Scale by the quantile only, no shift
        q = max(np.quantile(np.abs(clean_history), alpha), 0.01)
        
        def transform(x: np.ndarray) -> np.ndarray:
            return x / q
        
        def inverse_transform(x: np.ndarray) -> np.ndarray:
            return x * q
        
        return Scaler(
            transform=transform,
            inverse_transform=inverse_transform,
            original_min=original_min,
            original_max=original_max,
            shift=0.0,
            scale=q,
        )
    
    # LLMTime-style scaling with a shift.
    # Move the minimum down to leave room for lower predictions.
    shift = original_min - beta * (original_max - original_min)

    # Scale so the alpha-quantile equals 1.0
    shifted_history = clean_history - shift
    scale = np.quantile(shifted_history, alpha)
    
    if scale == 0:
        scale = 1.0
        logger.warning("SCALE FACTOR IS 0, USING 1.0")
    
    def transform(x: np.ndarray) -> np.ndarray:
        return (x - shift) / scale
    
    def inverse_transform(x: np.ndarray) -> np.ndarray:
        return x * scale + shift
    
    return Scaler(
        transform=transform,
        inverse_transform=inverse_transform,
        original_min=original_min,
        original_max=original_max,
        shift=shift,
        scale=scale,
    )


# Settings for writing numbers as text.
@dataclass
class SerializerSettings:
    base: int = 10
    prec: int = 3
    time_sep: str = ","
    bit_sep: str = ""
    plus_sign: str = ""
    minus_sign: str = "-"
    use_integers: bool = True
    model_family: ModelFamily = ModelFamily.LLAMA


# Finds the model family from the model name, to choose the tokenization strategy.
def detect_model_family(model_name: str) -> ModelFamily:
    if not model_name:
        return ModelFamily.UNKNOWN
    
    model_lower = model_name.lower()
    
    # GPT family. Needs spaces between digits.
    if any(gpt in model_lower for gpt in ["gpt", "openai"]):
        return ModelFamily.GPT
    
    # LLaMA family. One token per digit.
    if any(llama in model_lower for llama in ["llama", "meta-llama"]):
        return ModelFamily.LLAMA
    
    # Mistral family. Treated like LLaMA.
    if any(mistral in model_lower for mistral in ["mistral", "mixtral"]):
        return ModelFamily.MISTRAL
    
    return ModelFamily.UNKNOWN


# Picks the serializer settings that fit the model and provider.
def get_serializer_settings(
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    is_local: bool = False,
    use_integers: Optional[bool] = None,
) -> SerializerSettings:
    model_family = detect_model_family(model_name or provider or "")
    
    if is_local or provider == "local":
        # Local models. Integer format, with control over the allowed tokens.
        # LLaMA and Mistral tokenizers need no spaces.
        return SerializerSettings(
            prec=3,
            time_sep=",",
            bit_sep="",
            use_integers=True,
            model_family=model_family if model_family != ModelFamily.UNKNOWN else ModelFamily.LLAMA,
        )
    
    # API models. GPT needs spaces between digits to tokenize them one by one.
    # LLMTime paper: "We separate digits with spaces to force separate tokenization of each digit"
    # Example: 0.537 -> " 5 3 7" for GPT models
    if model_family == ModelFamily.GPT:
        return SerializerSettings(
            prec=3,
            time_sep=",", # Comma between numbers
            bit_sep=" ", # Space between the digits of a number (GPT)
            use_integers=True,
            model_family=ModelFamily.GPT,
        )

    return SerializerSettings(
        prec=3,
        time_sep=",",
        bit_sep="",
        use_integers=use_integers if use_integers is not None else True,
        model_family=model_family if model_family != ModelFamily.UNKNOWN else ModelFamily.LLAMA,
    )


# Writes a numeric array as a string for the LLM. The array is usually normalized to about [0, 1].
# With a fixed precision the decimal point carries no information, so it is dropped to save context length.
# Examples:
#     With use_integers=True, prec=2:
#         [0.5, 0.52, 0.55] -> "50,52,55," (LLaMA)
#         [0.5, 0.52, 0.55] -> "5 0 ,5 2 ,5 5 ," (GPT)
#     With use_integers=False, prec=3:
#         [0.5, 0.52, 0.55] -> "0.500, 0.520, 0.550, "

# How different models might tokenize "0.537":
# GPT-3:    ["0", ".", "5", "37"]      - 4 tokens
# GPT-4:    ["0.537"]                   - 1 token (different!)
# LLaMA:    ["0", ".", "5", "3", "7"]  - 5 tokens

# How "537" is tokenized:
# GPT-3:    ["5", "3", "7"] or ["537"] - consistent pattern
# LLaMA:    ["5", "3", "7"]            - always 3 tokens
def serialize_array(
    arr: np.ndarray,
    settings: SerializerSettings,
) -> str:
    formatted_parts = []
    
    for val in arr:
        if np.isnan(val):
            formatted_parts.append("NaN")
            continue
        
        if settings.use_integers:
            # Turn the value into an integer
            scaled = int(round(val * (10 ** settings.prec)))

            # Handle the sign
            sign = settings.minus_sign if scaled < 0 else settings.plus_sign
            abs_scaled = abs(scaled)
            
            if settings.bit_sep:
                # GPT style. A space between digits.
                digits = settings.bit_sep.join(str(d) for d in str(abs_scaled))
            else:
                # LLaMA style. No separator.
                digits = str(abs_scaled)
            
            formatted_parts.append(sign + digits)
        else:
            # Decimal format, for API models where the allowed tokens cannot be controlled
            if val >= 0:
                formatted_parts.append(f"{val:.{settings.prec}f}")
            else:
                formatted_parts.append(f"{settings.minus_sign}{abs(val):.{settings.prec}f}")
    
    # Join with the time separator and end with one
    result = settings.time_sep.join(formatted_parts)
    if not result.endswith(settings.time_sep):
        result += settings.time_sep
    
    return result


# Splits a long digit string into chunks of 2-3 digits.
# Used when the model writes numbers without any separators: "1051041031021011009998..."
def chunk_concatenated_digits(digit_string: str, expected_length: int) -> List[str]:
    chunks = []
    i = 0
    
    while i < len(digit_string) and len(chunks) < expected_length:
        # Try a 3-digit chunk first (values 100-300 are common)
        if i + 3 <= len(digit_string):
            three_digit = digit_string[i:i+3]
            two_digit = digit_string[i:i+2]
            
            # Take 3 digits if they start with 1 (100-199 range) or if the 2-digit value is too small
            if three_digit.startswith('1') and int(three_digit) <= 300:
                chunks.append(three_digit)
                i += 3
            elif int(two_digit) >= 10:
                chunks.append(two_digit)
                i += 2
            else:
                # Single digit or leading zero
                chunks.append(two_digit)
                i += 2
        elif i + 2 <= len(digit_string):
            chunks.append(digit_string[i:i+2])
            i += 2
        else:
            break
    
    return chunks


# Turns the LLM output string back into a numeric array.
# Handles comma-separated, space-separated and concatenated outputs.
def deserialize_string(
    text: str,
    settings: SerializerSettings,
    expected_length: Optional[int] = None,
    ignore_last: bool = False,
) -> Optional[np.ndarray]:
    if not text:
        logger.warning("EMPTY TEXT for deserialization")
        return None
    
    try:
        # Trim the text
        clean_text = text.strip()

        if settings.use_integers:
            # Find all integers with a regex.
            # This works for comma-separated (46,67,113) and space-separated (62 72 82) outputs.
            pattern = r"-?\d+"
            matches = re.findall(pattern, clean_text)
            
            # Fallback: if there is only one match and it is very long, the model probably
            # wrote the numbers without separators (e.g., "1051041031021011009998...")
            if len(matches) == 1 and len(matches[0]) > 10 and expected_length:
                logger.debug(f"ATTEMPTING CHUNK SPLIT: {len(matches[0])} digits -> {expected_length} values")
                matches = chunk_concatenated_digits(matches[0], expected_length)
                logger.debug(f"CHUNKED INTO: {len(matches)} values")
            
            if ignore_last and len(matches) > 1:
                matches = matches[:-1]
            
            values = []
            for match in matches:
                if match.startswith("-"):
                    sign = -1
                    digits = match[1:]
                else:
                    sign = 1
                    digits = match
                
                if digits:
                    scaled_val = int(digits)
                    values.append(sign * scaled_val / (10 ** settings.prec))
            
        else:
            # Decimal format. Find all numbers with a regex.
            pattern = r"-?\d+\.?\d*"
            matches = re.findall(pattern, clean_text)
            
            if ignore_last and len(matches) > 1:
                matches = matches[:-1]
            
            values = [float(m) for m in matches if m]
        
        if not values:
            logger.warning(f"NO VALUES PARSED FROM: {text[:100]}...")
            return None
        
        result = np.array(values)
        
        if expected_length is not None:
            if len(result) < expected_length:
                # Repeat the generated pattern instead of padding with a constant.
                # If the model wrote 60 of 96 values, tile the whole 60-step pattern until it reaches 96.
                # This keeps the oscillations and avoids a flat tail that inflates MAE.
                generated = result.copy()
                while len(result) < expected_length:
                    remaining = expected_length - len(result)
                    result = np.concatenate([result, generated[:remaining]])
                logger.debug(f"TILED OUTPUT FROM {len(generated)} to {expected_length}")
            elif len(result) > expected_length:
                result = result[:expected_length]
        
        return result
        
    except Exception as e:
        logger.warning(f"DESERIALIZATION FAILED: {e}, TEXT: {text[:100]}...")
        return None


# Returns the token IDs the model may produce when generating a time series.
def get_allowed_tokens(
    tokenizer,
    settings: SerializerSettings,
) -> List[int]:
    # Characters allowed in the time series output
    allowed_chars = list("0123456789")
    allowed_chars.append(settings.time_sep.strip())
    
    if settings.minus_sign:
        allowed_chars.append(settings.minus_sign)
    if settings.plus_sign:
        allowed_chars.append(settings.plus_sign)
    if settings.bit_sep and settings.bit_sep.strip():
        allowed_chars.append(settings.bit_sep.strip())
    
    # Also allow a space if a separator contains one
    if " " in settings.time_sep or " " in settings.bit_sep:
        allowed_chars.append(" ")
    
    allowed_ids = set()
    for char in allowed_chars:
        try:
            tokens = tokenizer.encode(char, add_special_tokens=False)
            allowed_ids.update(tokens)
        except Exception:
            pass
    
    return list(allowed_ids)


# Returns the bad_words_ids for model.generate(), which block every non-numeric token.
def get_bad_words_ids(
    tokenizer,
    settings: SerializerSettings,
) -> List[List[int]]:
    allowed_ids = set(get_allowed_tokens(tokenizer, settings))
    
    # Block every token that is not allowed
    bad_ids = []
    vocab_size = len(tokenizer)
    
    for token_id in range(vocab_size):
        if token_id not in allowed_ids:
            bad_ids.append([token_id])
    
    logger.debug(f"BLOCKING {len(bad_ids)}/{vocab_size} TOKENS")
    return bad_ids


# Builds a short text with statistics about the series, as in Time-LLM, for the prompt.
def build_statistical_context(
    values: np.ndarray,
    column_name: str,
    horizon: int,
    scaler: Optional[Scaler] = None,
    include_raw_stats: bool = True,
) -> str:
    # Statistics of the input values
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    median_val = float(np.median(values))
    
    # Work out the trend
    if len(values) > 1:
        trend = "upward" if values[-1] > values[0] else "downward"
        # Recent trend: the last value compared with the middle one
        mid_idx = len(values) // 2
        recent_trend = "increasing" if values[-1] > values[mid_idx] else "decreasing"
    else:
        trend = "stable"
        recent_trend = "stable"
    
    context_parts = [f"Sensor: {column_name}"]
    
    # Add the range in the original units, if available
    if include_raw_stats and scaler and hasattr(scaler, 'original_min'):
        context_parts.append(
            f"Original range: [{scaler.original_min:.2f}, {scaler.original_max:.2f}]"
        )
    
    # Add the statistics of the values passed in
    context_parts.extend([
        f"Statistics: min={min_val:.3f}, max={max_val:.3f}, median={median_val:.3f}",
        f"Trend: {trend}, Recent trend: {recent_trend}",
        f"Task: Continue with {horizon} values",
    ])
    
    return "\n".join(context_parts)
