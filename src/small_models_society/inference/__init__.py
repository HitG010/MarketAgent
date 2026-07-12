"""Reference-free contracts and local model inference support."""

from small_models_society.inference.config import (
    InferenceConfig,
    load_inference_config,
)
from small_models_society.inference.contracts import (
    AdapterReference,
    ChatMessage,
    GenerationOutput,
    GenerationRequest,
    InferenceExample,
    TextGenerationBackend,
    to_inference_example,
    validate_inference_example,
)
from small_models_society.inference.hardware import (
    HardwareReport,
    RuntimeCapabilities,
    detect_hardware,
    select_hardware,
)
from small_models_society.inference.huggingface import (
    HuggingFaceBackend,
    InferenceDependencyError,
    InferenceOutOfMemoryError,
)
from small_models_society.inference.prompts import (
    PromptCatalog,
    PromptProfileName,
    clean_response,
    load_prompt_catalog,
    render_generation_request,
    render_messages,
)
from small_models_society.inference.runner import (
    PredictionRunOptions,
    PredictionRunPlan,
    PredictionRunResult,
    ResumeMismatchError,
    RunManifest,
    inspect_prediction_run,
    run_predictions,
)

__all__ = [
    "AdapterReference",
    "ChatMessage",
    "GenerationOutput",
    "GenerationRequest",
    "HardwareReport",
    "HuggingFaceBackend",
    "InferenceConfig",
    "InferenceDependencyError",
    "InferenceExample",
    "InferenceOutOfMemoryError",
    "PredictionRunOptions",
    "PredictionRunPlan",
    "PredictionRunResult",
    "PromptCatalog",
    "PromptProfileName",
    "ResumeMismatchError",
    "RunManifest",
    "RuntimeCapabilities",
    "TextGenerationBackend",
    "clean_response",
    "detect_hardware",
    "inspect_prediction_run",
    "load_inference_config",
    "load_prompt_catalog",
    "render_generation_request",
    "render_messages",
    "run_predictions",
    "select_hardware",
    "to_inference_example",
    "validate_inference_example",
]
