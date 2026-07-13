from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from small_models_society.retrieval.contracts import (
    RetrievalHit,
    RetrievalRelevanceRecord,
    RetrievalResult,
    create_retrieval_document,
)


def test_document_identity_is_content_addressed_and_tamper_evident() -> None:
    first = create_retrieval_document("Paris", "Paris is the capital of France.")
    second = create_retrieval_document("Paris", "Paris is the capital of France.")

    assert first == second
    assert first.document_id == f"hotpot-{first.content_sha256}"
    assert "request" not in first.model_dump_json()
    assert "supporting" not in first.model_dump_json()

    value = json.loads(first.model_dump_json())
    value["text"] = "Tampered"
    with pytest.raises(ValidationError, match="content hash"):
        type(first).model_validate(value)


def test_relevance_labels_are_hidden_sorted_and_nonempty() -> None:
    document = create_retrieval_document("Paris", "Paris is in France.")
    record = RetrievalRelevanceRecord(
        request_id="routing-opaque",
        split="development",
        relevant_document_ids=(document.document_id,),
        unresolved_titles=("Missing",),
    )

    assert record.supporting_title_count == 2

    with pytest.raises(ValidationError, match="unique and sorted"):
        RetrievalRelevanceRecord(
            request_id="routing-opaque",
            split="development",
            relevant_document_ids=(document.document_id, document.document_id),
        )


def test_retrieval_result_requires_consistent_status_and_ranks() -> None:
    document = create_retrieval_document("Paris", "Paris is in France.")
    completed = RetrievalResult(
        request_id="routing-opaque",
        query_sha256="a" * 64,
        corpus_id="hotpotqa.routing.v1",
        corpus_fingerprint="b" * 64,
        top_k=1,
        status="completed",
        hits=(RetrievalHit(rank=1, score=1.25, document=document),),
        latency_ms=1,
    )

    assert completed.hits[0].document == document

    value = completed.model_dump(mode="json")
    value["status"] = "empty_query"
    with pytest.raises(ValidationError, match="cannot contain hits"):
        RetrievalResult.model_validate(value)
