from __future__ import annotations

import pytest
from pydantic import ValidationError

from small_models_society.schemas import (
    Domain,
    LogicExample,
    PredictionRecord,
    PredictionStatus,
    validate_example,
)


def test_validates_discriminated_math_example() -> None:
    example = validate_example(
        {
            "id": "math-1",
            "domain": "math",
            "input": {"question": "What is 2 + 2?"},
            "reference": {"answer": "4"},
        }
    )

    assert example.domain is Domain.MATH
    assert example.input.question == "What is 2 + 2?"


def test_logic_reference_must_match_a_choice() -> None:
    with pytest.raises(ValidationError, match="answer label must match a choice"):
        LogicExample.model_validate(
            {
                "id": "logic-1",
                "input": {
                    "question": "Pick one",
                    "choices": [
                        {"label": "A", "text": "First"},
                        {"label": "B", "text": "Second"},
                    ],
                },
                "reference": {"answer_label": "C"},
            }
        )


def test_prediction_record_cannot_contain_a_gold_reference() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PredictionRecord.model_validate(
            {
                "example_id": "math-1",
                "domain": "math",
                "response": "4",
                "model_id": "fixture-oracle",
                "latency_ms": 0,
                "reference": {"answer": "4"},
            }
        )


def test_prediction_dump_has_no_reference_field() -> None:
    prediction = PredictionRecord(
        example_id="math-1",
        domain=Domain.MATH,
        response="4",
        model_id="fixture-oracle",
        latency_ms=0,
    )

    assert "reference" not in prediction.model_dump()


def test_prediction_metadata_cannot_hide_a_gold_reference() -> None:
    with pytest.raises(ValidationError, match="cannot contain gold references"):
        PredictionRecord(
            example_id="math-1",
            domain=Domain.MATH,
            response="4",
            model_id="fixture-oracle",
            latency_ms=0,
            metadata={"trace": {"answer": "4"}},
        )


def test_abstention_does_not_require_a_response() -> None:
    prediction = PredictionRecord(
        example_id="math-1",
        domain=Domain.MATH,
        model_id="fixture-abstain",
        status=PredictionStatus.ABSTAINED,
        latency_ms=0,
    )

    assert prediction.response is None
