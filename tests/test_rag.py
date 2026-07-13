from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from small_models_society.inference.contracts import (
    ChatMessage,
    GenerationOutput,
    GenerationRequest,
)
from small_models_society.retrieval.bm25 import BM25Retriever, retrieval_config_fingerprint
from small_models_society.retrieval.contracts import (
    RetrievalResultStatus,
    create_retrieval_document,
)
from small_models_society.retrieval.rag import execute_rag, render_rag_generation_request
from small_models_society.routing.config import RoutingConfig, load_routing_config
from small_models_society.routing.contracts import (
    ActionAvailability,
    ActionOutcomeStatus,
    AvailabilityStatus,
    EnergyProvenance,
    OutputContract,
    RequestPolicyContext,
    SafetyStatus,
    WorkflowRequest,
    create_workflow_request,
)
from small_models_society.routing.registry import build_action_registry

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "routing.yaml"


def _config() -> RoutingConfig:
    return load_routing_config(CONFIG_PATH)


def _request(
    query: str | None = "What is the capital of France?",
    *,
    corpus_authorized: bool = True,
) -> WorkflowRequest:
    defaults = _config().policy_defaults
    attributes = {"retrieval_query": query} if query is not None else {}
    return create_workflow_request(
        request_id="routing-knowledge",
        messages=(
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="Question-only request without evidence."),
        ),
        output_contract=OutputContract.SHORT_ANSWER,
        policy=RequestPolicyContext(
            data_classification=defaults.data_classification,
            network_allowed=defaults.network_allowed,
            allowed_corpus_ids=(defaults.allowed_corpus_ids if corpus_authorized else ()),
            allowed_tool_ids=defaults.allowed_tool_ids,
            required_quality=defaults.required_quality,
            allow_unknown_output_safety=defaults.allow_unknown_output_safety,
        ),
        attributes=attributes,
    )


def _retriever(*, index_factory=None) -> BM25Retriever:  # type: ignore[no-untyped-def]
    documents = (
        create_retrieval_document("Paris", "Paris is the capital of France."),
        create_retrieval_document("Berlin", "Berlin is the capital of Germany."),
    )
    return BM25Retriever(
        documents,
        _config().retrieval,
        "a" * 64,
        index_factory=index_factory,
    )


def _action_and_availability():  # type: ignore[no-untyped-def]
    action = build_action_registry(_config()).actions["rag.bm25-qwen-base.v1"].action
    availability = ActionAvailability(
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=AvailabilityStatus.AVAILABLE,
    )
    return action, availability


class FakeBackend:
    def __init__(self, text: str = "Paris") -> None:
        self.text = text
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.requests.append(request)
        return GenerationOutput(
            text=self.text,
            prompt_tokens=40,
            completion_tokens=2,
            latency_ms=5,
            metadata={"device": "cpu"},
        )


def test_rag_rendering_contains_ranked_evidence_but_no_gold_labels() -> None:
    request = _request()
    action, _ = _action_and_availability()
    retrieval = _retriever().retrieve(request, top_k=1)

    generation = render_rag_generation_request(request, retrieval, action, "general")

    assert retrieval.status is RetrievalResultStatus.COMPLETED
    assert generation.request_id == request.request_id
    assert generation.max_new_tokens == 128
    assert generation.adapter is None
    user_text = generation.messages[-1].content
    assert "What is the capital of France?" in user_text
    assert retrieval.hits[0].document.document_id in user_text
    assert retrieval.hits[0].document.title in user_text
    assert retrieval.hits[0].document.text in user_text
    serialized = generation.model_dump_json()
    assert '"reference"' not in serialized
    assert '"supporting_facts"' not in serialized
    assert '"relevant_document_ids"' not in serialized


def test_rag_rendering_rejects_cross_request_or_stale_query_results() -> None:
    request = _request()
    action, _ = _action_and_availability()
    retrieval = _retriever().retrieve(request, top_k=1)

    with pytest.raises(ValueError, match="different request"):
        render_rag_generation_request(
            request.model_copy(update={"request_id": "different"}),
            retrieval,
            action,
            "general",
        )
    changed_query = request.model_copy(
        update={"attributes": {"retrieval_query": "A changed question?"}}
    )
    with pytest.raises(ValueError, match="different query"):
        render_rag_generation_request(changed_query, retrieval, action, "general")


def test_rag_executes_backend_with_local_unknown_cost_and_energy() -> None:
    request = _request()
    action, availability = _action_and_availability()
    backend = FakeBackend()
    times = iter([1.0, 1.01])

    outcome = execute_rag(
        request,
        action,
        availability,
        _config(),
        _retriever(),
        backend,
        generator_execution_fingerprint="f" * 64,
        clock=lambda: next(times),
    )

    assert len(backend.requests) == 1
    assert outcome.status is ActionOutcomeStatus.COMPLETED
    assert outcome.response == "Paris"
    assert outcome.safety.status is SafetyStatus.UNKNOWN
    assert outcome.telemetry is not None
    assert outcome.telemetry.wall_latency_ms == pytest.approx(10)
    assert outcome.telemetry.prompt_tokens == 40
    assert outcome.telemetry.completion_tokens == 2
    assert outcome.telemetry.provider_fee_usd == 0
    assert outcome.telemetry.compute_cost_usd is None
    assert outcome.telemetry.total_cost_usd is None
    assert outcome.telemetry.energy_provenance is EnergyProvenance.UNAVAILABLE
    assert len(outcome.metadata["hit_document_ids"]) > 0
    assert outcome.metadata["retrieval_config_fingerprint"] == (
        retrieval_config_fingerprint(_config().retrieval)
    )
    assert outcome.metadata["generator_execution_fingerprint"] == "f" * 64
    assert outcome.metadata["retrieval_top_k"] == 10


def test_empty_query_is_unsupported_without_model_invocation() -> None:
    request = _request(None)
    action, availability = _action_and_availability()
    backend = FakeBackend()

    outcome = execute_rag(
        request,
        action,
        availability,
        _config(),
        _retriever(),
        backend,
    )

    assert outcome.status is ActionOutcomeStatus.UNSUPPORTED
    assert outcome.response is None
    assert outcome.safety.status is SafetyStatus.NOT_ASSESSED
    assert backend.requests == []


class _BrokenIndex:
    def get_scores(self, query_tokens: list[str]) -> Iterable[float]:
        del query_tokens
        raise ValueError("synthetic retrieval failure")


def test_retrieval_error_is_explicit_without_model_invocation() -> None:
    request = _request()
    action, availability = _action_and_availability()
    backend = FakeBackend()

    outcome = execute_rag(
        request,
        action,
        availability,
        _config(),
        _retriever(index_factory=lambda _: _BrokenIndex()),
        backend,
    )

    assert outcome.status is ActionOutcomeStatus.ERROR
    assert outcome.error_type == "RetrievalError"
    assert outcome.metadata["retrieval_status"] == "error"
    assert backend.requests == []


def test_executor_rechecks_corpus_authorization_before_backend() -> None:
    request = _request(corpus_authorized=False)
    action, availability = _action_and_availability()
    backend = FakeBackend()

    with pytest.raises(ValueError, match="not authorized"):
        execute_rag(
            request,
            action,
            availability,
            _config(),
            _retriever(),
            backend,
        )

    assert backend.requests == []


def test_executor_refuses_blocked_action_before_backend() -> None:
    request = _request()
    action, availability = _action_and_availability()
    blocked = availability.model_copy(
        update={
            "status": AvailabilityStatus.BLOCKED,
            "reason_code": "corpus_not_authorized",
            "rule_ids": ("policy.corpus-authorization.v1",),
        }
    )
    backend = FakeBackend()

    with pytest.raises(ValueError, match="must not be executed"):
        execute_rag(
            request,
            action,
            blocked,
            _config(),
            _retriever(),
            backend,
        )

    assert backend.requests == []
