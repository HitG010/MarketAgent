"""Deterministic registry of Phase 4 candidate workflow actions."""

from __future__ import annotations

import hashlib
from typing import Literal, Self

from pydantic import Field, model_validator

from small_models_society.data.prepare import canonical_json
from small_models_society.routing.config import (
    EXPECTED_ACTION_IDS,
    LocalModelActionConfig,
    RetrievalActionConfig,
    RoutingConfig,
    StrongReplayActionConfig,
    ToolActionConfig,
)
from small_models_society.routing.contracts import WorkflowAction, create_workflow_action
from small_models_society.schemas import StrictModel


class RegisteredAction(StrictModel):
    action: WorkflowAction
    approved: bool
    requires_network: bool


class ActionRegistry(StrictModel):
    schema_version: Literal[1] = 1
    routing_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    actions: dict[str, RegisteredAction]
    registry_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    def fingerprint_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"registry_fingerprint"})

    def calculated_fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(self.fingerprint_payload()).encode("utf-8")
        ).hexdigest()

    @model_validator(mode="after")
    def validate_keys_and_fingerprint(self) -> Self:
        if set(self.actions) != EXPECTED_ACTION_IDS:
            raise ValueError("registry must contain exactly the configured candidate action IDs")
        mismatched = [
            action_id
            for action_id, registered in self.actions.items()
            if registered.action.action_id != action_id
        ]
        if mismatched:
            raise ValueError(f"registry keys do not match action IDs: {sorted(mismatched)}")
        if self.registry_fingerprint != self.calculated_fingerprint():
            raise ValueError("registry_fingerprint does not match registry contents")
        return self


def build_action_registry(config: RoutingConfig) -> ActionRegistry:
    """Translate strict routing configuration into immutable action identities."""

    registered_actions: dict[str, RegisteredAction] = {}
    for action_id, configured in config.actions.items():
        if isinstance(configured, ToolActionConfig):
            action = create_workflow_action(
                action_id=action_id,
                kind=configured.kind,
                executor_id="calculator.ast.v1",
                tool_id=configured.tool_id,
            )
            approved = True
            requires_network = False
        elif isinstance(configured, LocalModelActionConfig):
            action = create_workflow_action(
                action_id=action_id,
                kind=configured.kind,
                executor_id="huggingface.qwen.v1",
                model_id=config.model.model_id,
                model_revision=config.model.revision,
                adapter_id=(configured.adapter.value if configured.adapter is not None else None),
                max_new_tokens=configured.max_new_tokens,
            )
            approved = configured.approved
            requires_network = False
        elif isinstance(configured, RetrievalActionConfig):
            action = create_workflow_action(
                action_id=action_id,
                kind=configured.kind,
                executor_id="bm25-rag.qwen.v1",
                retriever_id=configured.retriever_id,
                corpus_id=configured.corpus_id,
                model_id=config.model.model_id,
                model_revision=config.model.revision,
                max_new_tokens=configured.max_new_tokens,
            )
            approved = True
            requires_network = False
        elif isinstance(configured, StrongReplayActionConfig):
            action = create_workflow_action(
                action_id=action_id,
                kind=configured.kind,
                executor_id="strong-model.replay.v1",
                provider_id=configured.provider_id,
                model_id=configured.model_id,
                model_version=configured.model_version,
                pricing_schedule_id=configured.pricing_schedule_id,
                max_new_tokens=configured.max_new_tokens,
            )
            approved = True
            requires_network = True
        else:
            raise TypeError(f"unsupported configured action: {type(configured).__name__}")
        registered_actions[action_id] = RegisteredAction(
            action=action,
            approved=approved,
            requires_network=requires_network,
        )

    values: dict[str, object] = {
        "schema_version": 1,
        "routing_config_fingerprint": config.fingerprint(),
        "actions": {
            action_id: registered.model_dump(mode="json")
            for action_id, registered in registered_actions.items()
        },
    }
    fingerprint = hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()
    return ActionRegistry.model_validate({**values, "registry_fingerprint": fingerprint})
