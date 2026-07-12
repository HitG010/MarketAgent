"""Evaluation harness for the Small Models Society project."""

from small_models_society.schemas import (
    BenchmarkExample,
    Domain,
    PredictionRecord,
    validate_example,
)

__all__ = ["BenchmarkExample", "Domain", "PredictionRecord", "validate_example"]
__version__ = "0.2.0"
