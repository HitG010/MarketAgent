"""Pinned deterministic BM25 retrieval and evaluator metrics."""

from __future__ import annotations

import hashlib
import importlib
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import Protocol, Self, cast

from small_models_society.data.prepare import canonical_json
from small_models_society.retrieval.contracts import (
    RankedRetrievalObservation,
    RetrievalDocument,
    RetrievalHit,
    RetrievalMetrics,
    RetrievalRelevanceRecord,
    RetrievalResult,
    RetrievalResultStatus,
)
from small_models_society.retrieval.corpus import LoadedRetrievalCorpus
from small_models_society.routing.config import RetrievalConfig
from small_models_society.routing.contracts import WorkflowRequest

_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def retrieval_config_fingerprint(config: RetrievalConfig) -> str:
    return hashlib.sha256(
        canonical_json(config.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


class _BM25Index(Protocol):
    def get_scores(self, query_tokens: list[str]) -> Iterable[float]: ...


IndexFactory = Callable[[list[list[str]]], _BM25Index]
Clock = Callable[[], float]


def tokenize_retrieval_text(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return tuple(_TOKEN.findall(normalized))


def retrieval_query(request: WorkflowRequest) -> str:
    value = request.attributes.get("retrieval_query")
    return value if isinstance(value, str) else ""


def _load_index_factory(expected_version: str) -> IndexFactory:
    try:
        installed_version = version("rank-bm25")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "rank-bm25 is required for routing retrieval; install requirements-routing.lock"
        ) from error
    if installed_version != expected_version:
        raise RuntimeError(
            f"rank-bm25 version mismatch: expected {expected_version}, got {installed_version}"
        )
    module = importlib.import_module("rank_bm25")
    return cast(IndexFactory, module.BM25Okapi)


class BM25Retriever:
    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        config: RetrievalConfig,
        corpus_fingerprint: str,
        *,
        clock: Clock = perf_counter,
        index_factory: IndexFactory | None = None,
    ) -> None:
        if not documents:
            raise ValueError("BM25 retrieval requires at least one corpus document")
        self.documents = tuple(sorted(documents, key=lambda document: document.document_id))
        if len({document.document_id for document in self.documents}) != len(self.documents):
            raise ValueError("BM25 corpus contains duplicate document IDs")
        self.config = config
        self.corpus_fingerprint = corpus_fingerprint
        self.clock = clock
        factory = index_factory or _load_index_factory(config.library_version)
        tokenized_documents = [
            list(tokenize_retrieval_text(f"{document.title} {document.text}"))
            for document in self.documents
        ]
        self.index = factory(tokenized_documents)

    @classmethod
    def from_loaded_corpus(
        cls,
        corpus: LoadedRetrievalCorpus,
        config: RetrievalConfig,
        *,
        clock: Clock = perf_counter,
        index_factory: IndexFactory | None = None,
    ) -> Self:
        if corpus.manifest.corpus_id != config.corpus_id:
            raise ValueError("loaded corpus ID does not match retrieval configuration")
        if corpus.manifest.tokenizer_id != config.tokenizer:
            raise ValueError("loaded corpus tokenizer does not match retrieval configuration")
        return cls(
            corpus.documents,
            config,
            corpus.manifest.corpus_fingerprint,
            clock=clock,
            index_factory=index_factory,
        )

    def retrieve(self, request: WorkflowRequest, top_k: int) -> RetrievalResult:
        if top_k <= 0:
            raise ValueError("BM25 top_k must be positive")
        query = retrieval_query(request)
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        started = self.clock()
        query_tokens = list(tokenize_retrieval_text(query))
        if not query_tokens:
            return RetrievalResult(
                request_id=request.request_id,
                query_sha256=query_sha256,
                corpus_id=self.config.corpus_id,
                corpus_fingerprint=self.corpus_fingerprint,
                top_k=top_k,
                status=RetrievalResultStatus.EMPTY_QUERY,
                latency_ms=(self.clock() - started) * 1000,
            )
        try:
            scores = [float(score) for score in self.index.get_scores(query_tokens)]
        except (ArithmeticError, TypeError, ValueError):
            return RetrievalResult(
                request_id=request.request_id,
                query_sha256=query_sha256,
                corpus_id=self.config.corpus_id,
                corpus_fingerprint=self.corpus_fingerprint,
                top_k=top_k,
                status=RetrievalResultStatus.ERROR,
                latency_ms=(self.clock() - started) * 1000,
                error_code="bm25_scoring_error",
            )
        if len(scores) != len(self.documents) or any(not math.isfinite(score) for score in scores):
            return RetrievalResult(
                request_id=request.request_id,
                query_sha256=query_sha256,
                corpus_id=self.config.corpus_id,
                corpus_fingerprint=self.corpus_fingerprint,
                top_k=top_k,
                status=RetrievalResultStatus.ERROR,
                latency_ms=(self.clock() - started) * 1000,
                error_code="invalid_bm25_scores",
            )
        ranked = sorted(
            zip(self.documents, scores, strict=True),
            key=lambda item: (-item[1], item[0].document_id),
        )[:top_k]
        hits = tuple(
            RetrievalHit(rank=rank, score=score, document=document)
            for rank, (document, score) in enumerate(ranked, start=1)
        )
        return RetrievalResult(
            request_id=request.request_id,
            query_sha256=query_sha256,
            corpus_id=self.config.corpus_id,
            corpus_fingerprint=self.corpus_fingerprint,
            top_k=top_k,
            status=RetrievalResultStatus.COMPLETED,
            hits=hits,
            latency_ms=(self.clock() - started) * 1000,
        )


def evaluate_retrieval_metrics(
    results: Sequence[RetrievalResult],
    relevance: Sequence[RetrievalRelevanceRecord],
    top_k_values: tuple[int, ...],
) -> RetrievalMetrics:
    if not results:
        raise ValueError("retrieval metrics require at least one result")
    if len({(result.corpus_id, result.corpus_fingerprint) for result in results}) != 1:
        raise ValueError("retrieval metrics require one corpus identity")
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
    return evaluate_ranked_retrieval_metrics(observations, relevance, top_k_values)


def evaluate_ranked_retrieval_metrics(
    observations: Sequence[RankedRetrievalObservation],
    relevance: Sequence[RetrievalRelevanceRecord],
    top_k_values: tuple[int, ...],
) -> RetrievalMetrics:
    if not observations:
        raise ValueError("retrieval metrics require at least one result")
    if any(value <= 0 for value in top_k_values):
        raise ValueError("retrieval metric top-k values must be positive")
    if tuple(sorted(set(top_k_values))) != top_k_values:
        raise ValueError("retrieval metric top-k values must be unique and sorted")
    results_by_id = {result.request_id: result for result in observations}
    relevance_by_id = {record.request_id: record for record in relevance}
    if len(results_by_id) != len(observations):
        raise ValueError("retrieval results contain duplicate request IDs")
    if len(relevance_by_id) != len(relevance):
        raise ValueError("retrieval relevance contains duplicate request IDs")
    if set(results_by_id) != set(relevance_by_id):
        raise ValueError("retrieval results and relevance must have identical request IDs")
    required_depth = max(top_k_values)
    if any(result.top_k < required_depth for result in observations):
        raise ValueError("retrieval results do not cover every requested top-k metric")

    recall_sums = {top_k: 0.0 for top_k in top_k_values}
    reciprocal_rank_sum = 0.0
    indexed_supporting_titles = 0
    total_supporting_titles = 0
    for request_id in sorted(results_by_id):
        result = results_by_id[request_id]
        labels = relevance_by_id[request_id]
        relevant_ids = set(labels.relevant_document_ids)
        total_supporting_titles += labels.supporting_title_count
        indexed_supporting_titles += len(relevant_ids)
        for top_k in top_k_values:
            retrieved_ids = set(result.ranked_document_ids[:top_k])
            recall_sums[top_k] += len(retrieved_ids & relevant_ids) / labels.supporting_title_count
        first_relevant_rank = next(
            (
                rank
                for rank, document_id in enumerate(result.ranked_document_ids, start=1)
                if document_id in relevant_ids
            ),
            None,
        )
        if first_relevant_rank is not None:
            reciprocal_rank_sum += 1 / first_relevant_rank

    request_count = len(observations)
    return RetrievalMetrics(
        request_count=request_count,
        top_k_values=top_k_values,
        recall_at_k={top_k: recall_sums[top_k] / request_count for top_k in top_k_values},
        mean_reciprocal_rank=reciprocal_rank_sum / request_count,
        empty_query_rate=(
            sum(result.status is RetrievalResultStatus.EMPTY_QUERY for result in observations)
            / request_count
        ),
        error_rate=(
            sum(result.status is RetrievalResultStatus.ERROR for result in observations)
            / request_count
        ),
        mean_latency_ms=(sum(result.latency_ms for result in observations) / request_count),
        corpus_coverage=indexed_supporting_titles / total_supporting_titles,
    )
