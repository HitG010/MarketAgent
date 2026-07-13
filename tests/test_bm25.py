from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from small_models_society.inference.contracts import ChatMessage
from small_models_society.retrieval.bm25 import (
    BM25Retriever,
    evaluate_ranked_retrieval_metrics,
    evaluate_retrieval_metrics,
    tokenize_retrieval_text,
)
from small_models_society.retrieval.contracts import (
    RankedRetrievalObservation,
    RetrievalHit,
    RetrievalRelevanceRecord,
    RetrievalResult,
    RetrievalResultStatus,
    create_retrieval_document,
)
from small_models_society.routing.config import RoutingConfig, load_routing_config
from small_models_society.routing.contracts import (
    OutputContract,
    RequestPolicyContext,
    WorkflowRequest,
    create_workflow_request,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "routing.yaml"


def _config() -> RoutingConfig:
    return load_routing_config(CONFIG_PATH)


def _request(query: str, request_id: str = "routing-query") -> WorkflowRequest:
    defaults = _config().policy_defaults
    return create_workflow_request(
        request_id=request_id,
        messages=(ChatMessage(role="user", content="Question only."),),
        output_contract=OutputContract.SHORT_ANSWER,
        policy=RequestPolicyContext(
            data_classification=defaults.data_classification,
            network_allowed=defaults.network_allowed,
            allowed_corpus_ids=defaults.allowed_corpus_ids,
            allowed_tool_ids=defaults.allowed_tool_ids,
            required_quality=defaults.required_quality,
            allow_unknown_output_safety=defaults.allow_unknown_output_safety,
        ),
        attributes={"retrieval_query": query},
    )


def _documents():  # type: ignore[no-untyped-def]
    return (
        create_retrieval_document("Paris", "Paris is the capital of France."),
        create_retrieval_document("Berlin", "Berlin is the capital of Germany."),
        create_retrieval_document("Paris, Texas", "Paris is a city in Texas."),
    )


def test_tokenizer_is_nfkc_lowercase_regex_without_implicit_resources() -> None:
    assert tokenize_retrieval_text("ＣＡＦÉ café_42 -- Straße") == (
        "café",
        "café",
        "42",
        "straße",
    )


def test_bm25_ranks_relevant_document_and_stably_breaks_ties() -> None:
    config = _config().retrieval
    retriever = BM25Retriever(_documents(), config, "a" * 64)

    relevant = retriever.retrieve(_request("capital France"), top_k=3)
    tied = retriever.retrieve(_request("quuxzy"), top_k=3)

    assert relevant.status is RetrievalResultStatus.COMPLETED
    assert relevant.hits[0].document.title == "Paris"
    assert [hit.document.document_id for hit in tied.hits] == sorted(
        document.document_id for document in _documents()
    )


def test_rejects_unpinned_bm25_runtime_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "small_models_society.retrieval.bm25.version",
        lambda _: "0.2.1",
    )

    with pytest.raises(RuntimeError, match="version mismatch"):
        BM25Retriever(_documents(), _config().retrieval, "a" * 64)


def test_empty_query_is_explicit_and_does_not_return_arbitrary_documents() -> None:
    times = iter([1.0, 1.002])
    retriever = BM25Retriever(
        _documents(),
        _config().retrieval,
        "a" * 64,
        clock=lambda: next(times),
    )

    result = retriever.retrieve(_request("  ---  "), top_k=3)

    assert result.status is RetrievalResultStatus.EMPTY_QUERY
    assert result.hits == ()
    assert result.latency_ms == pytest.approx(2)


class _BrokenIndex:
    def get_scores(self, query_tokens: list[str]) -> Iterable[float]:
        del query_tokens
        raise ValueError("synthetic failure")


def test_library_scoring_failure_becomes_explicit_error() -> None:
    retriever = BM25Retriever(
        _documents(),
        _config().retrieval,
        "a" * 64,
        index_factory=lambda _: _BrokenIndex(),
    )

    result = retriever.retrieve(_request("capital"), top_k=3)

    assert result.status is RetrievalResultStatus.ERROR
    assert result.error_code == "bm25_scoring_error"
    assert result.hits == ()


def test_metrics_separate_recall_mrr_errors_and_corpus_coverage() -> None:
    first, second, third = _documents()
    results = [
        RetrievalResult(
            request_id="one",
            query_sha256="a" * 64,
            corpus_id="hotpotqa.routing.v1",
            corpus_fingerprint="b" * 64,
            top_k=2,
            status=RetrievalResultStatus.COMPLETED,
            hits=(
                RetrievalHit(rank=1, score=2, document=first),
                RetrievalHit(rank=2, score=1, document=second),
            ),
            latency_ms=2,
        ),
        RetrievalResult(
            request_id="two",
            query_sha256="c" * 64,
            corpus_id="hotpotqa.routing.v1",
            corpus_fingerprint="b" * 64,
            top_k=2,
            status=RetrievalResultStatus.ERROR,
            latency_ms=4,
            error_code="synthetic",
        ),
    ]
    relevance = [
        RetrievalRelevanceRecord(
            request_id="one",
            split="development",
            relevant_document_ids=tuple(sorted((first.document_id, second.document_id))),
        ),
        RetrievalRelevanceRecord(
            request_id="two",
            split="development",
            relevant_document_ids=(third.document_id,),
            unresolved_titles=("Missing",),
        ),
    ]

    metrics = evaluate_retrieval_metrics(results, relevance, (1, 2))

    assert metrics.recall_at_k[1] == pytest.approx(0.25)
    assert metrics.recall_at_k[2] == pytest.approx(0.5)
    assert metrics.mean_reciprocal_rank == pytest.approx(0.5)
    assert metrics.empty_query_rate == 0
    assert metrics.error_rate == 0.5
    assert metrics.mean_latency_ms == 3
    assert metrics.corpus_coverage == 0.75

    observations = [
        RankedRetrievalObservation(
            request_id=result.request_id,
            status=result.status,
            top_k=result.top_k,
            ranked_document_ids=tuple(hit.document.document_id for hit in result.hits),
            latency_ms=result.latency_ms,
        )
        for result in results
    ]
    assert evaluate_ranked_retrieval_metrics(observations, relevance, (1, 2)) == metrics


def test_metrics_require_exact_request_alignment() -> None:
    result = BM25Retriever(_documents(), _config().retrieval, "a" * 64).retrieve(
        _request("capital", "result-id"),
        top_k=1,
    )
    relevance = RetrievalRelevanceRecord(
        request_id="different-id",
        split="development",
        relevant_document_ids=(_documents()[0].document_id,),
    )

    with pytest.raises(ValueError, match="identical request IDs"):
        evaluate_retrieval_metrics([result], [relevance], (1,))
