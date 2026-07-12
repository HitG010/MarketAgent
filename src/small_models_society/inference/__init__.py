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
from small_models_society.inference.prompts import (
    PromptCatalog,
    PromptProfileName,
    clean_response,
    load_prompt_catalog,
    render_generation_request,
    render_messages,
)

__all__ = [
    "ChatMessage",
    "GenerationOutput",
    "GenerationRequest",
    "InferenceConfig",
    "InferenceExample",
    "PromptCatalog",
    "PromptProfileName",
    "TextGenerationBackend",
    "clean_response",
    "load_inference_config",
    "load_prompt_catalog",
    "render_generation_request",
    "render_messages",
    "to_inference_example",
    "validate_inference_example",
]
