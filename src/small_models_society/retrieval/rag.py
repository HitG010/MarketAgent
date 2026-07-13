"""BM25 retrieval-augmented generation workflow adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from time import perf_counter

from small_models_society.inference.contracts import (
    ChatMessage,
    GenerationRequest,
    TextGenerationBackend,
)
from small_models_society.retrieval.bm25 import (
    BM25Retriever,
    retrieval_config_fingerprint,
    retrieval_query,
)
from small_models_society.retrieval.contracts import RetrievalResult, RetrievalResultStatus
from small_models_society.routing.config import ActionKind, RoutingConfig
from small_models_society.routing.contracts import (
    ActionAvailability,
    ActionOutcome,
    ActionOutcomeStatus,
    ActionTelemetry,
    AvailabilityStatus,
    EnergyProvenance,
    SafetyAssessment,
    SafetyStatus,
    WorkflowAction,
    WorkflowRequest,
)

RAG_EXECUTOR_ID = "bm25-rag.qwen.v1"
Clock = Callable[[], float]


def _validate_rag_inputs(
    request: WorkflowRequest,
    action: WorkflowAction,
    availability: ActionAvailability,
    retriever: BM25Retriever,
) -> None:
    if (
        action.kind is not ActionKind.RETRIEVAL
        or action.executor_id != RAG_EXECUTOR_ID
        or action.corpus_id is None
        or action.max_new_tokens is None
    ):
        raise ValueError("RAG executor received an incompatible workflow action")
    if (
        availability.action_id != action.action_id
        or availability.action_fingerprint != action.action_fingerprint
    ):
        raise ValueError("RAG availability does not match the workflow action")
    if availability.status is not AvailabilityStatus.AVAILABLE:
        raise ValueError("blocked or unavailable RAG actions must not be executed")
    if action.corpus_id not in set(request.policy.allowed_corpus_ids):
        raise ValueError("RAG corpus is not authorized by the request policy")
    if action.corpus_id != retriever.config.corpus_id:
        raise ValueError("RAG action and retriever corpus IDs do not match")


def render_rag_generation_request(
    request: WorkflowRequest,
    retrieval: RetrievalResult,
    action: WorkflowAction,
    profile: str,
) -> GenerationRequest:
    if retrieval.status is not RetrievalResultStatus.COMPLETED:
        raise ValueError("RAG generation requires a completed retrieval result")
    if action.max_new_tokens is None or action.corpus_id != retrieval.corpus_id:
        raise ValueError("RAG action does not match the retrieval result")
    query = retrieval_query(request)
    if retrieval.request_id != request.request_id:
        raise ValueError("retrieval result belongs to a different request")
    if retrieval.query_sha256 != hashlib.sha256(query.encode("utf-8")).hexdigest():
        raise ValueError("retrieval result belongs to a different query")
    evidence = "\n\n".join(
        f"[{hit.document.document_id}] {hit.document.title}\n{hit.document.text}"
        for hit in retrieval.hits
    )
    user_message = ChatMessage(
        role="user",
        content=(
            "Question:\n"
            f"{query}\n\n"
            "Answer the question using only the retrieved evidence. Give a concise answer.\n\n"
            "Retrieved evidence:\n"
            f"{evidence}"
        ),
    )
    messages = [*request.messages[:-1], user_message]
    return GenerationRequest(
        request_id=request.request_id,
        profile=profile,
        adapter=None,
        messages=messages,
        max_new_tokens=action.max_new_tokens,
    )


def _telemetry(
    wall_latency_ms: float,
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    device: str | None = None,
) -> ActionTelemetry:
    return ActionTelemetry(
        wall_latency_ms=wall_latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        provider_fee_usd=0,
        compute_cost_usd=None,
        total_cost_usd=None,
        energy_provenance=EnergyProvenance.UNAVAILABLE,
        device=device,
    )


def _retrieval_metadata(
    retrieval: RetrievalResult,
    config: RoutingConfig,
    generator_execution_fingerprint: str | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "executor_id": RAG_EXECUTOR_ID,
        "corpus_id": retrieval.corpus_id,
        "corpus_fingerprint": retrieval.corpus_fingerprint,
        "retrieval_config_fingerprint": retrieval_config_fingerprint(config.retrieval),
        "query_sha256": retrieval.query_sha256,
        "retrieval_top_k": retrieval.top_k,
        "retrieval_status": retrieval.status.value,
        "retrieval_latency_ms": retrieval.latency_ms,
        "hit_document_ids": [hit.document.document_id for hit in retrieval.hits],
        "hit_scores": [hit.score for hit in retrieval.hits],
    }
    if generator_execution_fingerprint is not None:
        metadata["generator_execution_fingerprint"] = generator_execution_fingerprint
    return metadata


def execute_rag(
    request: WorkflowRequest,
    action: WorkflowAction,
    availability: ActionAvailability,
    config: RoutingConfig,
    retriever: BM25Retriever,
    backend: TextGenerationBackend,
    *,
    generator_execution_fingerprint: str | None = None,
    clock: Clock = perf_counter,
) -> ActionOutcome:
    _validate_rag_inputs(request, action, availability, retriever)
    started = clock()
    retrieval = retriever.retrieve(request, max(config.retrieval.top_k_values))
    if retrieval.status is RetrievalResultStatus.EMPTY_QUERY:
        return ActionOutcome(
            request_id=request.request_id,
            request_fingerprint=request.request_fingerprint,
            action_id=action.action_id,
            action_fingerprint=action.action_fingerprint,
            status=ActionOutcomeStatus.UNSUPPORTED,
            availability=availability,
            safety=SafetyAssessment(
                status=SafetyStatus.NOT_ASSESSED,
                source=RAG_EXECUTOR_ID,
            ),
            telemetry=_telemetry((clock() - started) * 1000),
            metadata=_retrieval_metadata(
                retrieval,
                config,
                generator_execution_fingerprint,
            ),
        )
    if retrieval.status is RetrievalResultStatus.ERROR:
        return ActionOutcome(
            request_id=request.request_id,
            request_fingerprint=request.request_fingerprint,
            action_id=action.action_id,
            action_fingerprint=action.action_fingerprint,
            status=ActionOutcomeStatus.ERROR,
            availability=availability,
            safety=SafetyAssessment(
                status=SafetyStatus.NOT_ASSESSED,
                source=RAG_EXECUTOR_ID,
            ),
            telemetry=_telemetry((clock() - started) * 1000),
            error_type="RetrievalError",
            error_message=f"BM25 retrieval failed: {retrieval.error_code}",
            metadata=_retrieval_metadata(
                retrieval,
                config,
                generator_execution_fingerprint,
            ),
        )

    generation_retrieval = retrieval.model_copy(
        update={
            "top_k": config.retrieval.generation_top_k,
            "hits": retrieval.hits[: config.retrieval.generation_top_k],
        }
    )
    generation_request = render_rag_generation_request(
        request,
        generation_retrieval,
        action,
        config.model.prompt_profile,
    )
    generation = backend.generate(generation_request)
    wall_latency_ms = (clock() - started) * 1000
    device_value = generation.metadata.get("device")
    device = device_value if isinstance(device_value, str) and device_value else None
    if not generation.text.strip():
        return ActionOutcome(
            request_id=request.request_id,
            request_fingerprint=request.request_fingerprint,
            action_id=action.action_id,
            action_fingerprint=action.action_fingerprint,
            status=ActionOutcomeStatus.ERROR,
            availability=availability,
            safety=SafetyAssessment(
                status=SafetyStatus.NOT_ASSESSED,
                source=RAG_EXECUTOR_ID,
            ),
            telemetry=_telemetry(
                wall_latency_ms,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                device=device,
            ),
            error_type="EmptyGenerationError",
            error_message="RAG generator returned an empty response",
            metadata=_retrieval_metadata(
                retrieval,
                config,
                generator_execution_fingerprint,
            ),
        )
    return ActionOutcome(
        request_id=request.request_id,
        request_fingerprint=request.request_fingerprint,
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=ActionOutcomeStatus.COMPLETED,
        availability=availability,
        response=generation.text.strip(),
        safety=SafetyAssessment(
            status=SafetyStatus.UNKNOWN,
            source="not_assessed",
        ),
        telemetry=_telemetry(
            wall_latency_ms,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            device=device,
        ),
        metadata=_retrieval_metadata(
            retrieval,
            config,
            generator_execution_fingerprint,
        ),
    )
