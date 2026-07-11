"""Deterministic fixture predictions for validating the evaluation harness."""

from __future__ import annotations

from collections.abc import Sequence

from small_models_society.schemas import (
    BenchmarkExample,
    CodeExample,
    KnowledgeExample,
    LogicExample,
    MathExample,
    PredictionRecord,
)


def _oracle_response(example: BenchmarkExample) -> str:
    if isinstance(example, MathExample):
        return example.reference.answer
    if isinstance(example, CodeExample):
        if example.reference.canonical_solution is None:
            raise ValueError(f"code fixture {example.id} has no canonical solution")
        return example.reference.canonical_solution
    if isinstance(example, LogicExample):
        return example.reference.answer_label
    if isinstance(example, KnowledgeExample):
        return example.reference.answers[0]
    raise TypeError(f"unsupported example type: {type(example).__name__}")


def oracle_predictions(
    examples: Sequence[BenchmarkExample], model_id: str = "fixture-oracle"
) -> list[PredictionRecord]:
    predictions: list[PredictionRecord] = []
    for example in examples:
        predictions.append(
            PredictionRecord(
                example_id=example.id,
                domain=example.domain,
                response=_oracle_response(example),
                model_id=model_id,
                latency_ms=0,
            )
        )
    return predictions
