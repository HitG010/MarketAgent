from __future__ import annotations

import os
from pathlib import Path

import pytest

from small_models_society.experiments.workflow_matrix import CandidateWorkflowRuntime
from small_models_society.inference.config import load_inference_config
from small_models_society.inference.contracts import ChatMessage
from small_models_society.inference.hardware import detect_hardware
from small_models_society.inference.prompts import load_prompt_catalog
from small_models_society.retrieval.bm25 import BM25Retriever
from small_models_society.retrieval.contracts import create_retrieval_document
from small_models_society.routing.artifacts import load_workflow_requests
from small_models_society.routing.config import DataClassification, load_routing_config
from small_models_society.routing.contracts import (
    ActionOutcomeStatus,
    AvailabilityStatus,
    OutputContract,
    RequestPolicyContext,
    create_workflow_request,
)
from small_models_society.routing.local import LocalModelExecutor
from small_models_society.routing.registry import build_action_registry
from small_models_society.routing.replay import (
    import_replay_captures,
    load_pricing_catalog,
    load_verified_replay_catalog,
)

ROOT = Path(__file__).parents[1]
ROUTING_CONFIG = ROOT / "configs" / "routing.yaml"
INFERENCE_CONFIG = ROOT / "configs" / "inference.yaml"
PROMPT_CONFIG = ROOT / "configs" / "prompt_profiles.yaml"
REPLAY_FIXTURES = Path(__file__).parent / "fixtures" / "routing"

pytestmark = pytest.mark.workflow


def _policy(
    *,
    classification: DataClassification = DataClassification.PUBLIC,
    allow_corpus: bool = True,
) -> RequestPolicyContext:
    defaults = load_routing_config(ROUTING_CONFIG).policy_defaults
    return RequestPolicyContext(
        data_classification=classification,
        network_allowed=defaults.network_allowed,
        allowed_corpus_ids=(defaults.allowed_corpus_ids if allow_corpus else ()),
        allowed_tool_ids=defaults.allowed_tool_ids,
        required_quality=defaults.required_quality,
        allow_unknown_output_safety=defaults.allow_unknown_output_safety,
    )


def test_production_candidate_workflows_share_one_local_model(tmp_path: Path) -> None:
    if os.getenv("SMS_RUN_WORKFLOW_TESTS") != "1":
        pytest.skip("set SMS_RUN_WORKFLOW_TESTS=1 to run the candidate workflow smoke")

    routing = load_routing_config(ROUTING_CONFIG)
    inference = load_inference_config(INFERENCE_CONFIG)
    inference = inference.model_copy(
        update={"model": inference.model.model_copy(update={"local_files_only": True})}
    )
    hardware = detect_hardware(inference)
    if not hardware.ready:
        pytest.fail("; ".join(hardware.errors))
    prompts = load_prompt_catalog(PROMPT_CONFIG)
    registry = build_action_registry(routing)
    local = LocalModelExecutor(routing, inference, prompts, hardware)

    document = create_retrieval_document(
        "France",
        "Paris is the capital and most populous city of France.",
    )
    retriever = BM25Retriever(
        (document,),
        routing.retrieval,
        "a" * 64,
    )

    pricing = load_pricing_catalog(REPLAY_FIXTURES / "pricing.json")
    replay_import = import_replay_captures(
        REPLAY_FIXTURES / "replay_captures.jsonl",
        REPLAY_FIXTURES / "replay_requests.jsonl",
        tmp_path / "replay" / "rows.jsonl",
        routing,
        registry,
        pricing,
    )
    replay = load_verified_replay_catalog(
        replay_import.rows_path,
        REPLAY_FIXTURES / "replay_requests.jsonl",
        routing,
        registry,
        pricing,
    )
    runtime = CandidateWorkflowRuntime(
        routing,
        registry,
        local=local,
        retriever=retriever,
        replay=replay,
    )

    calculator_request = create_workflow_request(
        request_id="smoke-calculator",
        messages=(ChatMessage(role="user", content="7 + 5"),),
        output_contract=OutputContract.NUMERIC,
        policy=_policy(),
    )
    knowledge_request = create_workflow_request(
        request_id="smoke-knowledge",
        messages=(
            ChatMessage(role="system", content="Answer concisely."),
            ChatMessage(role="user", content="What is the capital of France?"),
        ),
        output_contract=OutputContract.SHORT_ANSWER,
        policy=_policy(),
        attributes={"retrieval_query": "What is the capital of France?"},
    )
    calculator = registry.actions["tool.calculator.v1"]
    base = registry.actions["local.qwen-base.v1"]
    rag = registry.actions["rag.bm25-qwen-base.v1"]
    replay_action = registry.actions["remote.strong-replay.reference.v1"]

    calculator_outcome = runtime.execute(
        calculator_request,
        calculator,
        runtime.availability(calculator_request, calculator),
    )
    base_outcome = runtime.execute(
        knowledge_request,
        base,
        runtime.availability(knowledge_request, base),
    )
    rag_outcome = runtime.execute(
        knowledge_request,
        rag,
        runtime.availability(knowledge_request, rag),
    )
    replay_request = load_workflow_requests(REPLAY_FIXTURES / "replay_requests.jsonl")[0]
    replay_outcome = runtime.execute(
        replay_request,
        replay_action,
        runtime.availability(replay_request, replay_action),
    )

    assert calculator_outcome.response == "12"
    assert base_outcome.status is ActionOutcomeStatus.COMPLETED
    assert base_outcome.response and base_outcome.response.strip()
    assert rag_outcome.status is ActionOutcomeStatus.COMPLETED
    assert rag_outcome.response and rag_outcome.response.strip()
    assert rag_outcome.metadata["hit_document_ids"] == [document.document_id]
    assert replay_outcome.status is ActionOutcomeStatus.COMPLETED
    assert replay_outcome.response == "Paris"
    assert local.backend_loaded is True

    restricted_request = create_workflow_request(
        request_id="smoke-restricted",
        messages=replay_request.messages,
        output_contract=replay_request.output_contract,
        policy=_policy(classification=DataClassification.RESTRICTED),
    )
    remote_block = runtime.availability(restricted_request, replay_action)
    assert remote_block.status is AvailabilityStatus.BLOCKED
    assert remote_block.reason_code == "data_not_remote_eligible"

    unauthorized_rag = create_workflow_request(
        request_id="smoke-unauthorized-rag",
        messages=knowledge_request.messages,
        output_contract=knowledge_request.output_contract,
        policy=_policy(allow_corpus=False),
        attributes=knowledge_request.attributes,
    )
    corpus_block = runtime.availability(unauthorized_rag, rag)
    assert corpus_block.status is AvailabilityStatus.BLOCKED
    assert corpus_block.reason_code == "corpus_not_authorized"
