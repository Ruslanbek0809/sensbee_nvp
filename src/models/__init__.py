# Models module for LLM-based time series forecasting.

from .serialize import (
    Scaler,
    SerializerSettings,
    ModelFamily,
    create_scaler,
    get_serializer_settings,
    serialize_array,
    deserialize_string,
    build_statistical_context,
    detect_model_family,
    get_allowed_tokens,
    get_bad_words_ids,
)

from .nvp_llms import (
    nvp_llms_forecast,
    forecast_from_json,
    forecast_from_api,
    aggregate_samples,
)

from .local_llm import (
    LocalModelConfig,
    LOCAL_MODELS,
    check_gpu_availability,
    is_local_llm_available,
    load_local_model,
    local_llm_completion,
    clear_model_cache,
)

__all__ = [
    # Serialization
    "Scaler",
    "SerializerSettings",
    "ModelFamily",
    "create_scaler",
    "get_serializer_settings",
    "serialize_array",
    "deserialize_string",
    "build_statistical_context",
    "detect_model_family",
    "get_allowed_tokens",
    "get_bad_words_ids",
    # Forecasting
    "nvp_llms_forecast",
    "forecast_from_json",
    "forecast_from_api",
    "aggregate_samples",
    # Local LLM
    "LocalModelConfig",
    "LOCAL_MODELS",
    "check_gpu_availability",
    "is_local_llm_available",
    "load_local_model",
    "local_llm_completion",
    "clear_model_cache",
]
