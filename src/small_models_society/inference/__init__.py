"""Reference-free contracts and local model inference support."""

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
    "InferenceExample",
    "TextGenerationBackend",
    "to_inference_example",
    "validate_inference_example",
]
