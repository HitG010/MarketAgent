"""Strict contracts for corpus construction and retrieval evaluation."""

from __future__ import annotations

import hashlib
import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from small_models_society.data.prepare import canonical_json
from small_models_society.schemas import StrictModel

DocumentId = Annotated[str, Field(pattern=r"^hotpot-[0-9a-f]{64}$")]
RetrievalSplit = Literal["development", "test"]


def retrieval_document_sha256(title: str, text: str) -> str:
    payload = {"title": title, "text": text}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class RetrievalDocument(StrictModel):
    schema_version: Literal[1] = 1
    document_id: DocumentId
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_content_identity(self) -> Self:
        expected = retrieval_document_sha256(self.title, self.text)
        if self.content_sha256 != expected:
            raise ValueError("retrieval document content hash does not match contents")
        if self.document_id != f"hotpot-{expected}":
            raise ValueError("retrieval document ID does not match content hash")
        return self


def create_retrieval_document(title: str, text: str) -> RetrievalDocument:
    content_sha256 = retrieval_document_sha256(title, text)
    return RetrievalDocument(
        document_id=f"hotpot-{content_sha256}",
        title=title,
        text=text,
        content_sha256=content_sha256,
    )


class RetrievalRelevanceRecord(StrictModel):
    schema_version: Literal[1] = 1
    request_id: str = Field(min_length=1)
    split: RetrievalSplit
    relevant_document_ids: tuple[DocumentId, ...] = ()
    unresolved_titles: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_relevance_labels(self) -> Self:
        if not self.relevant_document_ids and not self.unresolved_titles:
            raise ValueError("relevance record requires at least one supporting title")
        if tuple(sorted(set(self.relevant_document_ids))) != self.relevant_document_ids:
            raise ValueError("relevant document IDs must be unique and sorted")
        if tuple(sorted(set(self.unresolved_titles))) != self.unresolved_titles:
            raise ValueError("unresolved titles must be unique and sorted")
        return self

    @property
    def supporting_title_count(self) -> int:
        return len(self.relevant_document_ids) + len(self.unresolved_titles)


class RetrievalCorpusManifest(StrictModel):
    schema_version: Literal[1] = 1
    corpus_id: str = Field(min_length=1)
    split: RetrievalSplit
    tokenizer_id: str = Field(min_length=1)
    routing_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_evaluator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    documents_file: str = Field(min_length=1)
    documents_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_count: int = Field(gt=0)
    relevance_file: str = Field(min_length=1)
    relevance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    relevance_count: int = Field(gt=0)
    indexed_supporting_titles: int = Field(ge=0)
    total_supporting_titles: int = Field(gt=0)
    corpus_coverage: float = Field(ge=0, le=1)
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    def fingerprint_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"manifest_fingerprint"})

    def calculated_fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(self.fingerprint_payload()).encode("utf-8")
        ).hexdigest()

    @model_validator(mode="after")
    def validate_coverage_and_fingerprint(self) -> Self:
        if self.indexed_supporting_titles > self.total_supporting_titles:
            raise ValueError("indexed supporting-title count exceeds total")
        expected = self.indexed_supporting_titles / self.total_supporting_titles
        if not math.isclose(self.corpus_coverage, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("corpus coverage does not match supporting-title counts")
        if self.manifest_fingerprint != self.calculated_fingerprint():
            raise ValueError("retrieval manifest fingerprint does not match contents")
        return self


class RetrievalHit(StrictModel):
    rank: int = Field(gt=0)
    score: float
    document: RetrievalDocument

    @field_validator("score")
    @classmethod
    def score_must_be_finite(cls, score: float) -> float:
        if not math.isfinite(score):
            raise ValueError("retrieval score must be finite")
        return score


class RetrievalResultStatus(StrEnum):
    COMPLETED = "completed"
    EMPTY_QUERY = "empty_query"
    ERROR = "error"


class RetrievalResult(StrictModel):
    schema_version: Literal[1] = 1
    request_id: str = Field(min_length=1)
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_id: str = Field(min_length=1)
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k: int = Field(gt=0)
    status: RetrievalResultStatus
    hits: tuple[RetrievalHit, ...] = ()
    latency_ms: float = Field(ge=0)
    error_code: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_result_state(self) -> Self:
        if self.status is RetrievalResultStatus.COMPLETED:
            if not self.hits or self.error_code is not None:
                raise ValueError("completed retrieval requires hits and no error")
        elif self.hits:
            raise ValueError("empty or failed retrieval cannot contain hits")
        elif self.status is RetrievalResultStatus.ERROR and self.error_code is None:
            raise ValueError("failed retrieval requires an error code")
        elif self.status is RetrievalResultStatus.EMPTY_QUERY and self.error_code is not None:
            raise ValueError("empty query cannot contain an error code")
        ranks = tuple(hit.rank for hit in self.hits)
        if ranks != tuple(range(1, len(self.hits) + 1)):
            raise ValueError("retrieval hit ranks must be contiguous")
        document_ids = [hit.document.document_id for hit in self.hits]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("retrieval result contains duplicate documents")
        if len(self.hits) > self.top_k:
            raise ValueError("retrieval result contains more hits than requested")
        return self


class RankedRetrievalObservation(StrictModel):
    request_id: str = Field(min_length=1)
    status: RetrievalResultStatus
    top_k: int = Field(gt=0)
    ranked_document_ids: tuple[DocumentId, ...] = ()
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_ranked_documents(self) -> Self:
        if len(set(self.ranked_document_ids)) != len(self.ranked_document_ids):
            raise ValueError("ranked retrieval observation contains duplicate documents")
        if len(self.ranked_document_ids) > self.top_k:
            raise ValueError("ranked retrieval observation exceeds requested top-k")
        if self.status is RetrievalResultStatus.COMPLETED and not self.ranked_document_ids:
            raise ValueError("completed ranked retrieval observation requires documents")
        if self.status is not RetrievalResultStatus.COMPLETED and self.ranked_document_ids:
            raise ValueError("non-completed ranked retrieval observation cannot contain documents")
        return self


class RetrievalMetrics(StrictModel):
    request_count: int = Field(gt=0)
    top_k_values: tuple[int, ...] = Field(min_length=1)
    recall_at_k: dict[int, float]
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    empty_query_rate: float = Field(ge=0, le=1)
    error_rate: float = Field(ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)
    corpus_coverage: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_metric_keys(self) -> Self:
        if tuple(sorted(set(self.top_k_values))) != self.top_k_values:
            raise ValueError("retrieval metric top-k values must be unique and sorted")
        if set(self.recall_at_k) != set(self.top_k_values):
            raise ValueError("recall keys must match configured top-k values")
        if any(not 0 <= value <= 1 for value in self.recall_at_k.values()):
            raise ValueError("retrieval recall values must be between zero and one")
        return self
