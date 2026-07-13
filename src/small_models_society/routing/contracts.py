"""Reference-free contracts for candidate workflow research."""

from __future__ import annotations

import hashlib
import math
from enum import StrEnum
from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from small_models_society.data.prepare import canonical_json
from small_models_society.inference.contracts import ChatMessage
from small_models_society.routing.config import ActionKind, DataClassification
from small_models_society.schemas import StrictModel

_RESERVED_ROUTER_KEYS = frozenset(
    {
        "answer",
        "answers",
        "answer_key",
        "answer_label",
        "answerkey",
        "benchmark_domain",
        "canonical_solution",
        "domain",
        "gold",
        "gold_answer",
        "label",
        "rationale",
        "reference",
        "supporting_facts",
        "test_setup",
        "tests",
    }
)


def _contains_reserved_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() in _RESERVED_ROUTER_KEYS or _contains_reserved_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_reserved_key(item) for item in value)
    return False


class OutputContract(StrEnum):
    FREE_TEXT = "free_text"
    NUMERIC = "numeric"
    PYTHON_SOURCE = "python_source"
    CHOICE_LABEL = "choice_label"
    SHORT_ANSWER = "short_answer"


class RequestPolicyContext(StrictModel):
    data_classification: DataClassification
    network_allowed: bool
    allowed_corpus_ids: tuple[str, ...] = ()
    allowed_tool_ids: tuple[str, ...] = ()
    required_quality: float = Field(ge=0, le=1)
    allow_unknown_output_safety: bool = False

    @model_validator(mode="after")
    def reject_duplicate_allowlist_entries(self) -> Self:
        if len(set(self.allowed_corpus_ids)) != len(self.allowed_corpus_ids):
            raise ValueError("allowed_corpus_ids must not contain duplicates")
        if len(set(self.allowed_tool_ids)) != len(self.allowed_tool_ids):
            raise ValueError("allowed_tool_ids must not contain duplicates")
        return self


class WorkflowRequest(StrictModel):
    schema_version: int = Field(default=1, ge=1)
    request_id: str = Field(min_length=1)
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    output_contract: OutputContract
    policy: RequestPolicyContext
    attributes: dict[str, Any] = Field(default_factory=dict)
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    def fingerprint_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"request_fingerprint"})

    def calculated_fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(self.fingerprint_payload()).encode("utf-8")
        ).hexdigest()

    @field_validator("attributes")
    @classmethod
    def attributes_cannot_contain_evaluator_fields(
        cls,
        attributes: dict[str, Any],
    ) -> dict[str, Any]:
        if _contains_reserved_key(attributes):
            raise ValueError("workflow request attributes cannot contain evaluator fields")
        return attributes

    @model_validator(mode="after")
    def validate_messages_and_fingerprint(self) -> Self:
        if self.messages[-1].role != "user":
            raise ValueError("workflow request must end with a user message")
        if self.request_fingerprint != self.calculated_fingerprint():
            raise ValueError("request_fingerprint does not match request contents")
        return self


def create_workflow_request(
    *,
    request_id: str,
    messages: tuple[ChatMessage, ...],
    output_contract: OutputContract,
    policy: RequestPolicyContext,
    attributes: dict[str, Any] | None = None,
) -> WorkflowRequest:
    values: dict[str, object] = {
        "schema_version": 1,
        "request_id": request_id,
        "messages": [message.model_dump(mode="json") for message in messages],
        "output_contract": output_contract.value,
        "policy": policy.model_dump(mode="json"),
        "attributes": attributes or {},
    }
    fingerprint = hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()
    return WorkflowRequest.model_validate({**values, "request_fingerprint": fingerprint})


class WorkflowAction(StrictModel):
    schema_version: int = Field(default=1, ge=1)
    action_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    kind: ActionKind
    executor_id: str = Field(min_length=1)
    tool_id: str | None = Field(default=None, min_length=1)
    retriever_id: str | None = Field(default=None, min_length=1)
    corpus_id: str | None = Field(default=None, min_length=1)
    provider_id: str | None = Field(default=None, min_length=1)
    model_id: str | None = Field(default=None, min_length=1)
    model_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    model_version: str | None = Field(default=None, min_length=1)
    pricing_schedule_id: str | None = Field(default=None, min_length=1)
    adapter_id: str | None = Field(default=None, min_length=1)
    max_new_tokens: int | None = Field(default=None, gt=0)
    action_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    def fingerprint_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"action_fingerprint"})

    def calculated_fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(self.fingerprint_payload()).encode("utf-8")
        ).hexdigest()

    @model_validator(mode="after")
    def validate_action_shape_and_fingerprint(self) -> Self:
        if self.kind is ActionKind.TOOL:
            if self.tool_id is None:
                raise ValueError("tool action requires tool_id")
            if any(
                value is not None
                for value in (
                    self.retriever_id,
                    self.corpus_id,
                    self.provider_id,
                    self.model_id,
                    self.adapter_id,
                )
            ):
                raise ValueError("tool action cannot include retrieval or model fields")
        elif self.kind is ActionKind.LOCAL_MODEL:
            if self.model_id is None or self.model_revision is None or self.max_new_tokens is None:
                raise ValueError("local model action requires model, revision, and token budget")
            if any(
                value is not None
                for value in (self.tool_id, self.retriever_id, self.corpus_id, self.provider_id)
            ):
                raise ValueError("local model action cannot include tool, retrieval, or provider")
        elif self.kind is ActionKind.RETRIEVAL:
            if any(
                value is None
                for value in (
                    self.retriever_id,
                    self.corpus_id,
                    self.model_id,
                    self.model_revision,
                    self.max_new_tokens,
                )
            ):
                raise ValueError("retrieval action requires retriever, corpus, and local generator")
            if self.tool_id is not None or self.provider_id is not None:
                raise ValueError("retrieval action cannot include tool or remote provider")
        elif self.kind is ActionKind.STRONG_REPLAY:
            if any(
                value is None
                for value in (
                    self.provider_id,
                    self.model_id,
                    self.model_version,
                    self.pricing_schedule_id,
                )
            ):
                raise ValueError(
                    "strong replay action requires provider, model, version, and pricing"
                )
            if any(
                value is not None
                for value in (self.tool_id, self.retriever_id, self.corpus_id, self.adapter_id)
            ):
                raise ValueError("strong replay action cannot include tool, retrieval, or adapter")
        if self.action_fingerprint != self.calculated_fingerprint():
            raise ValueError("action_fingerprint does not match action contents")
        return self


def create_workflow_action(
    *,
    action_id: str,
    kind: ActionKind,
    executor_id: str,
    tool_id: str | None = None,
    retriever_id: str | None = None,
    corpus_id: str | None = None,
    provider_id: str | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    model_version: str | None = None,
    pricing_schedule_id: str | None = None,
    adapter_id: str | None = None,
    max_new_tokens: int | None = None,
) -> WorkflowAction:
    values: dict[str, object] = {
        "schema_version": 1,
        "action_id": action_id,
        "kind": kind.value,
        "executor_id": executor_id,
        "tool_id": tool_id,
        "retriever_id": retriever_id,
        "corpus_id": corpus_id,
        "provider_id": provider_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_version": model_version,
        "pricing_schedule_id": pricing_schedule_id,
        "adapter_id": adapter_id,
        "max_new_tokens": max_new_tokens,
    }
    fingerprint = hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()
    return WorkflowAction.model_validate({**values, "action_fingerprint": fingerprint})


class AvailabilityStatus(StrEnum):
    AVAILABLE = "available"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"


class ActionAvailability(StrictModel):
    action_id: str = Field(min_length=1)
    action_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: AvailabilityStatus
    reason_code: str | None = Field(default=None, min_length=1)
    rule_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_reason_for_non_available_action(self) -> Self:
        if self.status is AvailabilityStatus.AVAILABLE:
            if self.reason_code is not None or self.rule_ids:
                raise ValueError("available action cannot include block or unavailability reasons")
        elif self.reason_code is None:
            raise ValueError("blocked or unavailable action requires a reason_code")
        return self


class SafetyStatus(StrEnum):
    SAFE = "safe"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    NOT_ASSESSED = "not_assessed"


class SafetyAssessment(StrictModel):
    status: SafetyStatus
    source: str = Field(min_length=1)
    rule_ids: tuple[str, ...] = ()
    details: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def blocked_safety_requires_rules(self) -> Self:
        if self.status is SafetyStatus.BLOCKED and not self.rule_ids:
            raise ValueError("blocked safety assessment requires at least one rule ID")
        return self


class EnergyProvenance(StrEnum):
    MEASURED = "measured"
    REPLAY = "replay"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class ActionTelemetry(StrictModel):
    wall_latency_ms: float = Field(ge=0)
    queue_latency_ms: float | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    provider_fee_usd: float | None = Field(default=None, ge=0)
    compute_cost_usd: float | None = Field(default=None, ge=0)
    total_cost_usd: float | None = Field(default=None, ge=0)
    energy_joules: float | None = Field(default=None, ge=0)
    energy_provenance: EnergyProvenance = EnergyProvenance.UNAVAILABLE
    energy_measurement_source: str | None = Field(default=None, min_length=1)
    device: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_cost_and_energy_completeness(self) -> Self:
        if self.provider_fee_usd is not None and self.compute_cost_usd is not None:
            expected = self.provider_fee_usd + self.compute_cost_usd
            if self.total_cost_usd is None or not math.isclose(
                self.total_cost_usd,
                expected,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError("total_cost_usd must equal complete provider and compute costs")
        elif self.total_cost_usd is not None:
            raise ValueError("total_cost_usd must be null when a cost component is unknown")

        if self.energy_provenance is EnergyProvenance.UNAVAILABLE:
            if self.energy_joules is not None or self.energy_measurement_source is not None:
                raise ValueError("unavailable energy cannot include joules or measurement source")
        elif self.energy_joules is None or self.energy_measurement_source is None:
            raise ValueError("known energy requires joules and measurement source")
        return self


class ActionOutcomeStatus(StrEnum):
    COMPLETED = "completed"
    ERROR = "error"
    UNSUPPORTED = "unsupported"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"


class ActionOutcome(StrictModel):
    schema_version: int = Field(default=1, ge=1)
    request_id: str = Field(min_length=1)
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    action_id: str = Field(min_length=1)
    action_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ActionOutcomeStatus
    availability: ActionAvailability
    response: str | None = None
    safety: SafetyAssessment
    telemetry: ActionTelemetry | None = None
    error_type: str | None = Field(default=None, min_length=1)
    error_message: str | None = Field(default=None, min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def metadata_cannot_contain_evaluator_fields(
        cls,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if _contains_reserved_key(metadata):
            raise ValueError("action outcome metadata cannot contain evaluator fields")
        return metadata

    @model_validator(mode="after")
    def validate_state_consistency(self) -> Self:
        if self.availability.action_id != self.action_id:
            raise ValueError("availability action ID does not match outcome")
        if self.availability.action_fingerprint != self.action_fingerprint:
            raise ValueError("availability action fingerprint does not match outcome")

        expected_by_availability = {
            AvailabilityStatus.BLOCKED: ActionOutcomeStatus.BLOCKED,
            AvailabilityStatus.UNAVAILABLE: ActionOutcomeStatus.UNAVAILABLE,
        }
        expected = expected_by_availability.get(self.availability.status)
        if expected is not None and self.status is not expected:
            raise ValueError("outcome status does not match action availability")
        if self.availability.status is AvailabilityStatus.AVAILABLE and self.status in {
            ActionOutcomeStatus.BLOCKED,
            ActionOutcomeStatus.UNAVAILABLE,
        }:
            raise ValueError("available action cannot produce blocked or unavailable outcome")

        if self.status is ActionOutcomeStatus.COMPLETED:
            if self.response is None or not self.response.strip():
                raise ValueError("completed outcome requires a non-empty response")
            if self.telemetry is None:
                raise ValueError("completed outcome requires telemetry")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("completed outcome cannot contain an error")
        elif self.status is ActionOutcomeStatus.ERROR:
            if self.telemetry is None or self.error_type is None or self.error_message is None:
                raise ValueError("error outcome requires telemetry and error details")
            if self.response is not None:
                raise ValueError("error outcome cannot contain a response")
        elif self.status is ActionOutcomeStatus.UNSUPPORTED:
            if self.telemetry is None or self.response is not None:
                raise ValueError("unsupported outcome requires telemetry and no response")
        elif self.response is not None or self.telemetry is not None:
            raise ValueError("blocked or unavailable outcome cannot contain response or telemetry")
        return self
