"""Reference-free contracts and local model inference support."""

from small_models_society.inference.config import (
    InferenceConfig,
    load_inference_config,
)
from small_models_society.inference.contracts import (
    ChatMessage,
    GenerationOutput,
    GenerationRequest,
    InferenceExample,
    TextGenerationBackend,
    to_inference_example,
    validate_inference_example,
)

__all__ = [
    "ChatMessage",
    "GenerationOutput",
    "GenerationRequest",
    "InferenceConfig",
    "InferenceExample",
    "TextGenerationBackend",
    "load_inference_config",
    "to_inference_example",
    "validate_inference_example",
]
