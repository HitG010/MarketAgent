from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from small_models_society.inference.contracts import ChatMessage
from small_models_society.routing.artifacts import (
    load_action_outcomes,
    load_workflow_actions,
    load_workflow_requests,
    write_action_outcomes,
    write_workflow_actions,
    write_workflow_requests,
)
from small_models_society.routing.config import ActionKind, DataClassification
from small_models_society.routing.contracts import (
    ActionAvailability,
    ActionOutcome,
    ActionOutcomeStatus,
    ActionTelemetry,
    AvailabilityStatus,
    EnergyProvenance,
    OutputContract,
    RequestPolicyContext,
    SafetyAssessment,
    SafetyStatus,
    WorkflowAction,
    WorkflowRequest,
    create_workflow_action,
    create_workflow_request,
)

MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def _policy() -> RequestPolicyContext:
    return RequestPolicyContext(
        data_classification=DataClassification.PUBLIC,
        network_allowed=True,
        allowed_corpus_ids=("hotpotqa.routing.v1",),
        allowed_tool_ids=("calculator.v1",),
        required_quality=0.8,
        allow_unknown_output_safety=False,
    )


def _request(request_id: str = "request-1") -> WorkflowRequest:
    return create_workflow_request(
        request_id=request_id,
        messages=(
            ChatMessage(role="system", content="Be precise."),
            ChatMessage(role="user", content="What is one plus two?"),
        ),
        output_contract=OutputContract.NUMERIC,
        policy=_policy(),
        attributes={"locale": "en-US", "max_input_tokens": 64},
    )


def _action(action_id: str = "local.qwen-base.v1") -> WorkflowAction:
    return create_workflow_action(
        action_id=action_id,
        kind=ActionKind.LOCAL_MODEL,
        executor_id="huggingface.qwen.v1",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        model_revision=MODEL_REVISION,
        max_new_tokens=128,
    )


def _available(action: WorkflowAction) -> ActionAvailability:
    return ActionAvailability(
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=AvailabilityStatus.AVAILABLE,
    )


def _completed_outcome(
    request: WorkflowRequest | None = None,
    action: WorkflowAction | None = None,
) -> ActionOutcome:
    resolved_request = request or _request()
    resolved_action = action or _action()
    return ActionOutcome(
        request_id=resolved_request.request_id,
        request_fingerprint=resolved_request.request_fingerprint,
        action_id=resolved_action.action_id,
        action_fingerprint=resolved_action.action_fingerprint,
        status=ActionOutcomeStatus.COMPLETED,
        availability=_available(resolved_action),
        response="3",
        safety=SafetyAssessment(status=SafetyStatus.UNKNOWN, source="not_assessed"),
        telemetry=ActionTelemetry(
            wall_latency_ms=12.5,
            prompt_tokens=10,
            completion_tokens=1,
            provider_fee_usd=0,
            compute_cost_usd=None,
            total_cost_usd=None,
            energy_provenance=EnergyProvenance.UNAVAILABLE,
            device="mps",
        ),
        metadata={"executor": "fake"},
    )


def test_workflow_request_is_reference_and_domain_free() -> None:
    request = _request()
    serialized = request.model_dump_json()

    assert request.request_fingerprint == request.calculated_fingerprint()
    assert set(json.loads(serialized)) == {
        "schema_version",
        "request_id",
        "messages",
        "output_contract",
        "policy",
        "attributes",
        "request_fingerprint",
    }
    assert '"domain"' not in serialized
    assert '"reference"' not in serialized
    assert '"answer"' not in serialized


def test_workflow_request_rejects_evaluator_attributes_and_stale_fingerprint() -> None:
    with pytest.raises(ValidationError, match="evaluator fields"):
        create_workflow_request(
            request_id="leaked",
            messages=(ChatMessage(role="user", content="Question"),),
            output_contract=OutputContract.FREE_TEXT,
            policy=_policy(),
            attributes={"nested": {"domain": "math"}},
        )

    value = _request().model_dump(mode="json")
    value["messages"][-1]["content"] = "Changed question"
    with pytest.raises(ValidationError, match="request_fingerprint"):
        WorkflowRequest.model_validate(value)


def test_workflow_request_must_end_with_user_message() -> None:
    with pytest.raises(ValidationError, match="end with a user"):
        create_workflow_request(
            request_id="invalid-conversation",
            messages=(ChatMessage(role="assistant", content="Previous answer"),),
            output_contract=OutputContract.FREE_TEXT,
            policy=_policy(),
        )


def test_workflow_action_shape_and_fingerprint_are_strict() -> None:
    action = _action()

    assert action.action_fingerprint == action.calculated_fingerprint()
    assert action.kind is ActionKind.LOCAL_MODEL
    assert action.adapter_id is None

    with pytest.raises(ValidationError, match="cannot include retrieval or model"):
        create_workflow_action(
            action_id="tool.invalid.v1",
            kind=ActionKind.TOOL,
            executor_id="calculator.v1",
            tool_id="calculator.v1",
            model_id="unexpected/model",
        )

    value = action.model_dump(mode="json")
    value["max_new_tokens"] = 256
    with pytest.raises(ValidationError, match="action_fingerprint"):
        WorkflowAction.model_validate(value)


def test_cost_and_energy_unknowns_are_not_imputed_as_zero() -> None:
    telemetry = ActionTelemetry(
        wall_latency_ms=5,
        provider_fee_usd=0,
        compute_cost_usd=None,
        total_cost_usd=None,
        energy_provenance=EnergyProvenance.UNAVAILABLE,
    )

    assert telemetry.provider_fee_usd == 0
    assert telemetry.compute_cost_usd is None
    assert telemetry.total_cost_usd is None
    assert telemetry.energy_joules is None


def test_complete_cost_and_measured_energy_require_provenance() -> None:
    telemetry = ActionTelemetry(
        wall_latency_ms=20,
        provider_fee_usd=0.01,
        compute_cost_usd=0.02,
        total_cost_usd=0.03,
        energy_joules=12.5,
        energy_provenance=EnergyProvenance.MEASURED,
        energy_measurement_source="external-meter-v1",
    )

    assert telemetry.total_cost_usd == pytest.approx(0.03)

    with pytest.raises(ValidationError, match="total_cost_usd"):
        ActionTelemetry(
            wall_latency_ms=20,
            provider_fee_usd=0.01,
            compute_cost_usd=None,
            total_cost_usd=0.01,
        )
    with pytest.raises(ValidationError, match="measurement source"):
        ActionTelemetry(
            wall_latency_ms=20,
            energy_joules=12.5,
            energy_provenance=EnergyProvenance.MEASURED,
        )


def test_availability_and_outcome_states_cannot_disagree() -> None:
    request = _request()
    action = _action()
    blocked = ActionAvailability(
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=AvailabilityStatus.BLOCKED,
        reason_code="network_forbidden",
        rule_ids=("policy.network.v1",),
    )
    outcome = ActionOutcome(
        request_id=request.request_id,
        request_fingerprint=request.request_fingerprint,
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=ActionOutcomeStatus.BLOCKED,
        availability=blocked,
        safety=SafetyAssessment(
            status=SafetyStatus.BLOCKED,
            source="policy",
            rule_ids=("policy.network.v1",),
        ),
    )

    assert outcome.telemetry is None
    assert outcome.response is None

    value = outcome.model_dump(mode="json")
    value["status"] = "completed"
    value["response"] = "should not exist"
    with pytest.raises(ValidationError, match="availability"):
        ActionOutcome.model_validate(value)


def test_completed_outcome_requires_response_telemetry_and_clean_metadata() -> None:
    outcome = _completed_outcome()

    assert outcome.status is ActionOutcomeStatus.COMPLETED
    assert outcome.telemetry is not None
    assert outcome.telemetry.total_cost_usd is None

    value = outcome.model_dump(mode="json")
    value["metadata"] = {"nested": {"supporting_facts": ["hidden"]}}
    with pytest.raises(ValidationError, match="evaluator fields"):
        ActionOutcome.model_validate(value)


def test_routing_artifacts_are_canonical_atomic_and_round_trip(tmp_path: Path) -> None:
    requests = [_request("request-1"), _request("request-2")]
    actions = [_action()]
    outcomes = [_completed_outcome(request=request, action=actions[0]) for request in requests]

    first = write_workflow_requests(tmp_path / "first" / "requests.jsonl", requests)
    second = write_workflow_requests(tmp_path / "second" / "requests.jsonl", requests)
    action_artifact = write_workflow_actions(tmp_path / "actions.jsonl", actions)
    outcome_artifact = write_action_outcomes(tmp_path / "outcomes.jsonl", outcomes)

    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.row_count == 2
    assert action_artifact.row_count == 1
    assert outcome_artifact.row_count == 2
    assert load_workflow_requests(first.path) == requests
    assert load_workflow_actions(action_artifact.path) == actions
    assert load_action_outcomes(outcome_artifact.path) == outcomes
    assert not list(tmp_path.rglob("*.tmp"))


def test_routing_artifacts_reject_duplicates_and_tampering(tmp_path: Path) -> None:
    request = _request()
    with pytest.raises(ValueError, match="duplicate request IDs"):
        write_workflow_requests(tmp_path / "duplicates.jsonl", [request, request])

    path = tmp_path / "tampered.jsonl"
    write_workflow_requests(path, [request])
    value = json.loads(path.read_text(encoding="utf-8"))
    value["attributes"]["changed"] = True
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid workflow request row"):
        load_workflow_requests(path)
