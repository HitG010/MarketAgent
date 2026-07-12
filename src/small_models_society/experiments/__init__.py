"""Research experiment orchestration."""

from small_models_society.experiments.prompt_matrix import (
    PromptMatrixOptions,
    PromptMatrixPlan,
    PromptMatrixResult,
    inspect_prompt_matrix,
    run_prompt_matrix,
)

__all__ = [
    "PromptMatrixOptions",
    "PromptMatrixPlan",
    "PromptMatrixResult",
    "inspect_prompt_matrix",
    "run_prompt_matrix",
]
