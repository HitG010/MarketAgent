"""Typed contracts shared by data preparation, inference, and evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


class StrictModel(BaseModel):
    """Base contract that rejects accidental or leaked fields."""

    model_config = ConfigDict(extra="forbid")


class Domain(StrEnum):
    MATH = "math"
    CODE = "code"
    LOGIC = "logic"
    KNOWLEDGE = "knowledge"


class Choice(StrictModel):
    label: str = Field(min_length=1)
    text: str = Field(min_length=1)


class MathInput(StrictModel):
    question: str = Field(min_length=1)


class MathReference(StrictModel):
    answer: str = Field(min_length=1)
    rationale: str | None = None


class MathExample(StrictModel):
    id: str = Field(min_length=1)
    domain: Literal[Domain.MATH] = Domain.MATH
    input: MathInput
    reference: MathReference
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeInput(StrictModel):
    prompt: str = Field(min_length=1)
    entry_point: str | None = None


class CodeReference(StrictModel):
    test_setup: str = ""
    tests: list[str] = Field(min_length=1)
    canonical_solution: str | None = None


class CodeExample(StrictModel):
    id: str = Field(min_length=1)
    domain: Literal[Domain.CODE] = Domain.CODE
    input: CodeInput
    reference: CodeReference
    metadata: dict[str, Any] = Field(default_factory=dict)


class LogicInput(StrictModel):
    question: str = Field(min_length=1)
    choices: list[Choice] = Field(min_length=2)

    @field_validator("choices")
    @classmethod
    def labels_must_be_unique(cls, choices: list[Choice]) -> list[Choice]:
        labels = [choice.label for choice in choices]
        if len(labels) != len(set(labels)):
            raise ValueError("choice labels must be unique")
        return choices


class LogicReference(StrictModel):
    answer_label: str = Field(min_length=1)


class LogicExample(StrictModel):
    id: str = Field(min_length=1)
    domain: Literal[Domain.LOGIC] = Domain.LOGIC
    input: LogicInput
    reference: LogicReference
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reference")
    @classmethod
    def answer_must_match_a_choice(cls, reference: LogicReference, info: Any) -> LogicReference:
        logic_input = info.data.get("input")
        if logic_input is not None:
            labels = {choice.label for choice in logic_input.choices}
            if reference.answer_label not in labels:
                raise ValueError("answer label must match a choice")
        return reference


class KnowledgeInput(StrictModel):
    question: str = Field(min_length=1)
    context: list[str] = Field(default_factory=list)


class SupportingFact(StrictModel):
    title: str = Field(min_length=1)
    sentence_index: int = Field(ge=0)


class KnowledgeReference(StrictModel):
    answers: list[str] = Field(min_length=1)
    supporting_facts: list[SupportingFact] = Field(default_factory=list)


class KnowledgeExample(StrictModel):
    id: str = Field(min_length=1)
    domain: Literal[Domain.KNOWLEDGE] = Domain.KNOWLEDGE
    input: KnowledgeInput
    reference: KnowledgeReference
    metadata: dict[str, Any] = Field(default_factory=dict)


BenchmarkExample: TypeAlias = Annotated[
    MathExample | CodeExample | LogicExample | KnowledgeExample,
    Field(discriminator="domain"),
]
_EXAMPLE_ADAPTER: TypeAdapter[BenchmarkExample] = TypeAdapter(BenchmarkExample)


def validate_example(value: object) -> BenchmarkExample:
    """Validate an unknown object as one of the normalized domain contracts."""

    return _EXAMPLE_ADAPTER.validate_python(value)


class PredictionStatus(StrEnum):
    OK = "ok"
    ABSTAINED = "abstained"
    ERROR = "error"


class PredictionRecord(StrictModel):
    """Model output contract; gold references are deliberately absent."""

    example_id: str = Field(min_length=1)
    domain: Domain
    model_id: str = Field(min_length=1)
    status: PredictionStatus = PredictionStatus.OK
    response: str | None = None
    latency_ms: float = Field(ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def successful_predictions_need_a_response(self) -> Self:
        if self.status == PredictionStatus.OK and (
            self.response is None or not self.response.strip()
        ):
            raise ValueError("successful predictions require a non-empty response")
        return self

    @field_validator("metadata")
    @classmethod
    def metadata_cannot_contain_references(cls, metadata: dict[str, Any]) -> dict[str, Any]:
        reserved = {
            "answer",
            "answers",
            "answer_key",
            "answer_label",
            "answerkey",
            "canonical_solution",
            "gold",
            "gold_answer",
            "rationale",
            "reference",
            "supporting_facts",
            "test_setup",
            "tests",
        }

        def contains_reserved_key(value: object) -> bool:
            if isinstance(value, dict):
                return any(
                    str(key).lower() in reserved or contains_reserved_key(item)
                    for key, item in value.items()
                )
            if isinstance(value, list):
                return any(contains_reserved_key(item) for item in value)
            return False

        if contains_reserved_key(metadata):
            raise ValueError("prediction metadata cannot contain gold references")
        return metadata
