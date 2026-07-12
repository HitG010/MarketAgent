"""Typed contracts for deterministic specialist source data."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from small_models_society.schemas import BenchmarkExample, Domain, StrictModel


class TrainingSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"


class SourceTrainingRecord(StrictModel):
    schema_version: Literal[1] = 1
    source_id: str = Field(min_length=1)
    domain: Domain
    split: TrainingSplit
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    example: BenchmarkExample

    @model_validator(mode="after")
    def domain_must_match_example(self) -> Self:
        if self.domain is not self.example.domain:
            raise ValueError("record domain must match normalized example domain")
        return self


_SOURCE_RECORD_ADAPTER: TypeAdapter[SourceTrainingRecord] = TypeAdapter(SourceTrainingRecord)


def validate_source_training_record(value: object) -> SourceTrainingRecord:
    """Validate an unknown value as a source training record."""

    return _SOURCE_RECORD_ADAPTER.validate_python(value)