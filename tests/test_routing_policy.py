from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from small_models_society.inference.contracts import ChatMessage
from small_models_society.routing.config import RoutingConfig, load_routing_config
from small_models_society.routing.contracts import (
    AvailabilityStatus,
    OutputContract,
    RequestPolicyContext,
    WorkflowRequest,
    create_workflow_request,
)
from small_models_society.routing.policy import (
    RULE_ADAPTER_APPROVAL,
    RULE_ADAPTER_ARTIFACT,
    RULE_CALCULATOR_INPUT,
    RULE_CORPUS_AUTHORIZATION,
    RULE_CORPUS_RUNTIME,
    RULE_LOCAL_MODEL_RUNTIME,
    RULE_NETWORK,
    RULE_REMOTE_DATA,
    RULE_REPLAY_ROW,
    RULE_TOOL_ALLOWLIST,
    ActionRuntimeContext,
    available_action_ids,
    evaluate_registry_availability,
)
from small_models_society.routing.registry import ActionRegistry, build_action_registry

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "routing.yaml"
POLICY_FIXTURES = Path(__file__).parent / "fixtures" / "routing" / "policy_contexts.jsonl"


def _fixture_policies() -> dict[str, RequestPolicyContext]:
    fixtures: dict[str, RequestPolicyContext] = {}
    for line in POLICY_FIXTURES.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        fixtures[str(value["name"])] = RequestPolicyContext.model_validate(value["policy"])
    return fixtures


def _request(policy_name: str = "public") -> WorkflowRequest:
    return create_workflow_request(
        request_id=f"request-{policy_name}",
        messages=(ChatMessage(role="user", content="What is one plus two?"),),
        output_contract=OutputContract.NUMERIC,
        policy=_fixture_policies()[policy_name],
    )


def _runtime(**updates: object) -> ActionRuntimeContext:
    values: dict[str, object] = {
        "local_model_ready": True,
        "verified_adapter_ids": ("math", "code", "logic", "knowledge"),
        "available_corpus_ids": ("hotpotqa.routing.v1",),
        "replay_action_ids": ("remote.strong-replay.reference.v1",),
        "calculator_supported": True,
    }
    values.update(updates)
    return ActionRuntimeContext.model_validate(values)


def _approved_config(domain: str) -> RoutingConfig:
    config = load_routing_config(CONFIG_PATH)
    action_id = f"local.qwen-lora-{domain}.v1"
    configured = config.actions[action_id]
    actions = {
        **config.actions,
        action_id: configured.model_copy(update={"approved": True}),
    }
    return config.model_copy(update={"actions": actions})


def test_registry_builds_stable_composable_action_identities() -> None:
    config = load_routing_config(CONFIG_PATH)
    first = build_action_registry(config)
    second = build_action_registry(config)

    assert first == second
    assert len(first.registry_fingerprint) == 64
    assert first.actions["local.qwen-base.v1"].action.max_new_tokens == 512
    assert first.actions["local.qwen-lora-math.v1"].action.adapter_id == "math"
    assert first.actions["rag.bm25-qwen-base.v1"].action.corpus_id == ("hotpotqa.routing.v1")
    assert first.actions["remote.strong-replay.reference.v1"].requires_network is True
    assert first.actions["tool.calculator.v1"].action.tool_id == "calculator.v1"


def test_registry_fingerprint_detects_mutation() -> None:
    registry = build_action_registry(load_routing_config(CONFIG_PATH))
    value = registry.model_dump(mode="json")
    value["actions"]["local.qwen-base.v1"]["approved"] = False

    with pytest.raises(ValidationError, match="registry_fingerprint"):
        ActionRegistry.model_validate(value)


def test_public_request_exposes_only_approved_ready_actions() -> None:
    registry = build_action_registry(load_routing_config(CONFIG_PATH))
    decisions = evaluate_registry_availability(_request(), registry, _runtime())

    assert available_action_ids(decisions) == (
        "local.qwen-base.v1",
        "rag.bm25-qwen-base.v1",
        "remote.strong-replay.reference.v1",
        "tool.calculator.v1",
    )
    for domain in ("math", "code", "logic", "knowledge"):
        decision = decisions[f"local.qwen-lora-{domain}.v1"]
        assert decision.status is AvailabilityStatus.UNAVAILABLE
        assert decision.reason_code == "adapter_not_approved"
        assert decision.rule_ids == (RULE_ADAPTER_APPROVAL,)


def test_approved_lora_still_requires_verified_artifact() -> None:
    registry = build_action_registry(_approved_config("math"))
    missing = evaluate_registry_availability(
        _request(),
        registry,
        _runtime(verified_adapter_ids=()),
    )
    ready = evaluate_registry_availability(
        _request(),
        registry,
        _runtime(verified_adapter_ids=("math",)),
    )

    assert missing["local.qwen-lora-math.v1"].reason_code == "adapter_artifact_missing"
    assert missing["local.qwen-lora-math.v1"].rule_ids == (RULE_ADAPTER_ARTIFACT,)
    assert ready["local.qwen-lora-math.v1"].status is AvailabilityStatus.AVAILABLE


def test_adapter_unapproval_precedes_transient_local_runtime_failure() -> None:
    registry = build_action_registry(load_routing_config(CONFIG_PATH))
    decision = evaluate_registry_availability(
        _request(),
        registry,
        _runtime(local_model_ready=False),
    )["local.qwen-lora-math.v1"]

    assert decision.reason_code == "adapter_not_approved"
    assert decision.rule_ids == (RULE_ADAPTER_APPROVAL,)


@pytest.mark.parametrize(
    ("policy_name", "reason", "rule"),
    [
        ("confidential", "data_not_remote_eligible", RULE_REMOTE_DATA),
        ("restricted", "data_not_remote_eligible", RULE_REMOTE_DATA),
        ("network-disabled", "network_forbidden", RULE_NETWORK),
    ],
)
def test_remote_action_respects_data_and_network_policy(
    policy_name: str,
    reason: str,
    rule: str,
) -> None:
    registry = build_action_registry(load_routing_config(CONFIG_PATH))
    decision = evaluate_registry_availability(
        _request(policy_name),
        registry,
        _runtime(),
    )["remote.strong-replay.reference.v1"]

    assert decision.status is AvailabilityStatus.BLOCKED
    assert decision.reason_code == reason
    assert rule in decision.rule_ids


def test_corpus_and_tool_authorization_are_hard_blocks() -> None:
    registry = build_action_registry(load_routing_config(CONFIG_PATH))
    unauthorized_corpus = evaluate_registry_availability(
        _request("unauthorized-corpus"),
        registry,
        _runtime(),
    )
    disallowed_tool = evaluate_registry_availability(
        _request("disallowed-tool"),
        registry,
        _runtime(),
    )

    rag = unauthorized_corpus["rag.bm25-qwen-base.v1"]
    calculator = disallowed_tool["tool.calculator.v1"]
    assert rag.status is AvailabilityStatus.BLOCKED
    assert rag.reason_code == "corpus_not_authorized"
    assert rag.rule_ids == (RULE_CORPUS_AUTHORIZATION,)
    assert calculator.status is AvailabilityStatus.BLOCKED
    assert calculator.reason_code == "tool_not_allowed"
    assert calculator.rule_ids == (RULE_TOOL_ALLOWLIST,)


def test_runtime_failures_are_unavailable_not_policy_blocked() -> None:
    registry = build_action_registry(load_routing_config(CONFIG_PATH))
    decisions = evaluate_registry_availability(
        _request(),
        registry,
        _runtime(
            local_model_ready=False,
            available_corpus_ids=(),
            replay_action_ids=(),
            calculator_supported=False,
        ),
    )

    expected = {
        "local.qwen-base.v1": ("local_model_not_ready", RULE_LOCAL_MODEL_RUNTIME),
        "rag.bm25-qwen-base.v1": ("corpus_not_ready", RULE_CORPUS_RUNTIME),
        "remote.strong-replay.reference.v1": ("replay_row_missing", RULE_REPLAY_ROW),
        "tool.calculator.v1": ("calculator_input_unsupported", RULE_CALCULATOR_INPUT),
    }
    for action_id, (reason, rule) in expected.items():
        decision = decisions[action_id]
        assert decision.status is AvailabilityStatus.UNAVAILABLE
        assert decision.reason_code == reason
        assert decision.rule_ids == (rule,)


def test_blocked_actions_are_excluded_before_executor_iteration() -> None:
    registry = build_action_registry(load_routing_config(CONFIG_PATH))
    decisions = evaluate_registry_availability(
        _request("restricted"),
        registry,
        _runtime(),
    )
    executed: list[str] = []

    for action_id in available_action_ids(decisions):
        executed.append(action_id)

    assert "remote.strong-replay.reference.v1" not in executed
    assert decisions["remote.strong-replay.reference.v1"].status is AvailabilityStatus.BLOCKED


def test_policy_fixtures_cover_declared_research_cases() -> None:
    assert set(_fixture_policies()) == {
        "public",
        "internal",
        "confidential",
        "restricted",
        "network-disabled",
        "unauthorized-corpus",
        "disallowed-tool",
    }


def test_runtime_context_rejects_duplicate_entries() -> None:
    with pytest.raises(ValidationError, match="duplicate entries"):
        _runtime(verified_adapter_ids=("math", "math"))
