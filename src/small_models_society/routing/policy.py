"""Hard policy and runtime availability gates for workflow actions."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from small_models_society.routing.config import ActionKind, DataClassification
from small_models_society.routing.contracts import (
    ActionAvailability,
    AvailabilityStatus,
    WorkflowRequest,
)
from small_models_society.routing.registry import ActionRegistry, RegisteredAction
from small_models_society.schemas import StrictModel

RULE_REMOTE_DATA = "policy.remote-data.v1"
RULE_NETWORK = "policy.network.v1"
RULE_CORPUS_AUTHORIZATION = "policy.corpus-authorization.v1"
RULE_TOOL_ALLOWLIST = "policy.tool-allowlist.v1"
RULE_LOCAL_MODEL_RUNTIME = "runtime.local-model.v1"
RULE_ADAPTER_APPROVAL = "runtime.adapter-approval.v1"
RULE_ADAPTER_ARTIFACT = "runtime.adapter-artifact.v1"
RULE_CORPUS_RUNTIME = "runtime.corpus.v1"
RULE_REPLAY_ROW = "runtime.replay-row.v1"
RULE_CALCULATOR_INPUT = "runtime.calculator-input.v1"


class ActionRuntimeContext(StrictModel):
    local_model_ready: bool
    verified_adapter_ids: tuple[str, ...] = ()
    available_corpus_ids: tuple[str, ...] = ()
    replay_action_ids: tuple[str, ...] = ()
    calculator_supported: bool

    @model_validator(mode="after")
    def reject_duplicate_runtime_entries(self) -> Self:
        fields = {
            "verified_adapter_ids": self.verified_adapter_ids,
            "available_corpus_ids": self.available_corpus_ids,
            "replay_action_ids": self.replay_action_ids,
        }
        duplicates = [name for name, values in fields.items() if len(set(values)) != len(values)]
        if duplicates:
            raise ValueError(f"runtime context contains duplicate entries: {duplicates}")
        return self


def _availability(
    registered: RegisteredAction,
    status: AvailabilityStatus,
    reason_code: str | None = None,
    rule_ids: tuple[str, ...] = (),
) -> ActionAvailability:
    return ActionAvailability(
        action_id=registered.action.action_id,
        action_fingerprint=registered.action.action_fingerprint,
        status=status,
        reason_code=reason_code,
        rule_ids=rule_ids,
    )


def evaluate_action_availability(
    request: WorkflowRequest,
    registered: RegisteredAction,
    runtime: ActionRuntimeContext,
) -> ActionAvailability:
    """Apply hard policy blocks before runtime availability checks."""

    action = registered.action
    blocked_rules: list[str] = []
    blocked_reason: str | None = None

    if registered.requires_network:
        if request.policy.data_classification in {
            DataClassification.CONFIDENTIAL,
            DataClassification.RESTRICTED,
        }:
            blocked_reason = "data_not_remote_eligible"
            blocked_rules.append(RULE_REMOTE_DATA)
        if not request.policy.network_allowed:
            blocked_reason = blocked_reason or "network_forbidden"
            blocked_rules.append(RULE_NETWORK)
    if action.kind is ActionKind.RETRIEVAL and action.corpus_id not in set(
        request.policy.allowed_corpus_ids
    ):
        blocked_reason = blocked_reason or "corpus_not_authorized"
        blocked_rules.append(RULE_CORPUS_AUTHORIZATION)
    if action.kind is ActionKind.TOOL and action.tool_id not in set(
        request.policy.allowed_tool_ids
    ):
        blocked_reason = blocked_reason or "tool_not_allowed"
        blocked_rules.append(RULE_TOOL_ALLOWLIST)
    if blocked_reason is not None:
        return _availability(
            registered,
            AvailabilityStatus.BLOCKED,
            blocked_reason,
            tuple(blocked_rules),
        )

    if action.kind is ActionKind.TOOL:
        if not runtime.calculator_supported:
            return _availability(
                registered,
                AvailabilityStatus.UNAVAILABLE,
                "calculator_input_unsupported",
                (RULE_CALCULATOR_INPUT,),
            )
    elif action.kind is ActionKind.LOCAL_MODEL:
        if action.adapter_id is not None and not registered.approved:
            return _availability(
                registered,
                AvailabilityStatus.UNAVAILABLE,
                "adapter_not_approved",
                (RULE_ADAPTER_APPROVAL,),
            )
        if not runtime.local_model_ready:
            return _availability(
                registered,
                AvailabilityStatus.UNAVAILABLE,
                "local_model_not_ready",
                (RULE_LOCAL_MODEL_RUNTIME,),
            )
        if action.adapter_id is not None and action.adapter_id not in set(
            runtime.verified_adapter_ids
        ):
            return _availability(
                registered,
                AvailabilityStatus.UNAVAILABLE,
                "adapter_artifact_missing",
                (RULE_ADAPTER_ARTIFACT,),
            )
    elif action.kind is ActionKind.RETRIEVAL:
        if action.corpus_id not in set(runtime.available_corpus_ids):
            return _availability(
                registered,
                AvailabilityStatus.UNAVAILABLE,
                "corpus_not_ready",
                (RULE_CORPUS_RUNTIME,),
            )
        if not runtime.local_model_ready:
            return _availability(
                registered,
                AvailabilityStatus.UNAVAILABLE,
                "local_generator_not_ready",
                (RULE_LOCAL_MODEL_RUNTIME,),
            )
    elif action.kind is ActionKind.STRONG_REPLAY and action.action_id not in set(
        runtime.replay_action_ids
    ):
        return _availability(
            registered,
            AvailabilityStatus.UNAVAILABLE,
            "replay_row_missing",
            (RULE_REPLAY_ROW,),
        )

    return _availability(registered, AvailabilityStatus.AVAILABLE)


def evaluate_registry_availability(
    request: WorkflowRequest,
    registry: ActionRegistry,
    runtime: ActionRuntimeContext,
) -> dict[str, ActionAvailability]:
    """Evaluate every registered action in stable action-ID order."""

    return {
        action_id: evaluate_action_availability(request, registry.actions[action_id], runtime)
        for action_id in sorted(registry.actions)
    }


def available_action_ids(
    decisions: dict[str, ActionAvailability],
) -> tuple[str, ...]:
    """Return only actions that passed both policy and runtime gates."""

    return tuple(
        action_id
        for action_id in sorted(decisions)
        if decisions[action_id].status is AvailabilityStatus.AVAILABLE
    )
