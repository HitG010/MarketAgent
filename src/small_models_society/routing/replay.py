"""Provider-neutral strong-model replay import, validation, and execution."""

from __future__ import annotations

import hashlib
import math
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Self, TypeVar

from pydantic import Field, field_validator, model_validator

from small_models_society.data.prepare import canonical_json, sha256_bytes
from small_models_society.routing.artifacts import (
    RoutingArtifact,
    load_workflow_requests,
)
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
from small_models_society.routing.registry import ActionRegistry
from small_models_society.schemas import StrictModel

REPLAY_EXECUTOR_ID = "strong-model.replay.v1"
_RESERVED_CAPTURE_KEYS = frozenset(
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
_SENSITIVE_CAPTURE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
    }
)


def _contains_forbidden_capture_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() in _RESERVED_CAPTURE_KEYS
            or str(key).casefold() in _SENSITIVE_CAPTURE_KEYS
            or _contains_forbidden_capture_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_capture_key(item) for item in value)
    return False


class PricingEntry(StrictModel):
    pricing_schedule_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prompt_usd_per_million_tokens: float = Field(ge=0)
    completion_usd_per_million_tokens: float = Field(ge=0)

    @field_validator(
        "prompt_usd_per_million_tokens",
        "completion_usd_per_million_tokens",
    )
    @classmethod
    def pricing_rates_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("pricing rates must be finite")
        return value

    def provider_fee(self, prompt_tokens: int, completion_tokens: int) -> float:
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("token counts must be nonnegative")
        return (
            prompt_tokens * self.prompt_usd_per_million_tokens
            + completion_tokens * self.completion_usd_per_million_tokens
        ) / 1_000_000


class PricingCatalog(StrictModel):
    schema_version: Literal[1] = 1
    currency: Literal["USD"] = "USD"
    entries: dict[str, PricingEntry]
    catalog_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    def fingerprint_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"catalog_fingerprint"})

    def calculated_fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(self.fingerprint_payload()).encode("utf-8")
        ).hexdigest()

    @model_validator(mode="after")
    def validate_entries_and_fingerprint(self) -> Self:
        if not self.entries:
            raise ValueError("pricing catalog requires at least one entry")
        mismatched = [
            schedule_id
            for schedule_id, entry in self.entries.items()
            if entry.pricing_schedule_id != schedule_id
        ]
        if mismatched:
            raise ValueError(f"pricing entry keys do not match schedule IDs: {mismatched}")
        if self.catalog_fingerprint != self.calculated_fingerprint():
            raise ValueError("pricing catalog fingerprint does not match contents")
        return self


def create_pricing_catalog(entries: Mapping[str, PricingEntry]) -> PricingCatalog:
    values: dict[str, object] = {
        "schema_version": 1,
        "currency": "USD",
        "entries": {
            schedule_id: entry.model_dump(mode="json") for schedule_id, entry in entries.items()
        },
    }
    return PricingCatalog.model_validate(
        {
            **values,
            "catalog_fingerprint": hashlib.sha256(
                canonical_json(values).encode("utf-8")
            ).hexdigest(),
        }
    )


def load_pricing_catalog(path: Path) -> PricingCatalog:
    try:
        return PricingCatalog.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid replay pricing catalog: {path}") from error


class ReplayCapture(StrictModel):
    schema_version: Literal[1] = 1
    request_id: str = Field(min_length=1)
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    action_id: str = Field(min_length=1)
    action_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    response: str = Field(min_length=1)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    observed_latency_ms: float = Field(ge=0)
    provider_fee_usd: float = Field(ge=0)
    pricing_schedule_id: str = Field(min_length=1)
    provider_safety: SafetyAssessment
    energy_joules: float | None = Field(default=None, ge=0)
    energy_provenance: EnergyProvenance = EnergyProvenance.UNAVAILABLE
    energy_measurement_source: str | None = Field(default=None, min_length=1)
    captured_at_utc: datetime
    capture_source: str = Field(min_length=1)
    capture_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("observed_latency_ms", "provider_fee_usd", "energy_joules")
    @classmethod
    def telemetry_values_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("replay telemetry values must be finite")
        return value

    @field_validator("response")
    @classmethod
    def response_must_not_be_whitespace(cls, response: str) -> str:
        if not response.strip():
            raise ValueError("replay response must not be whitespace")
        return response

    @field_validator("captured_at_utc")
    @classmethod
    def capture_time_must_be_utc(cls, captured_at: datetime) -> datetime:
        if captured_at.utcoffset() != timedelta(0):
            raise ValueError("captured_at_utc must use an explicit UTC offset")
        return captured_at

    @field_validator("capture_metadata")
    @classmethod
    def capture_metadata_must_not_leak_or_store_credentials(
        cls,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if _contains_forbidden_capture_key(metadata):
            raise ValueError("capture metadata contains evaluator or credential fields")
        return metadata

    @model_validator(mode="after")
    def validate_energy_provenance(self) -> Self:
        if self.energy_provenance is EnergyProvenance.ESTIMATED:
            raise ValueError("strong replay rows do not accept estimated energy")
        if self.energy_provenance is EnergyProvenance.UNAVAILABLE:
            if self.energy_joules is not None or self.energy_measurement_source is not None:
                raise ValueError("unavailable replay energy cannot include measurement fields")
        elif self.energy_joules is None or self.energy_measurement_source is None:
            raise ValueError("known replay energy requires joules and measurement source")
        return self


class StrongReplayRow(ReplayCapture):
    pricing_catalog_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def fingerprint_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"row_sha256"})

    def calculated_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json(self.fingerprint_payload()).encode("utf-8")
        ).hexdigest()

    @model_validator(mode="after")
    def validate_row_hash(self) -> Self:
        if self.row_sha256 != self.calculated_sha256():
            raise ValueError("replay row hash does not match contents")
        return self


ReplayT = TypeVar("ReplayT", bound=ReplayCapture)


def create_strong_replay_row(
    capture: ReplayCapture,
    pricing_catalog_fingerprint: str,
) -> StrongReplayRow:
    values = {
        **capture.model_dump(mode="json"),
        "pricing_catalog_fingerprint": pricing_catalog_fingerprint,
    }
    return StrongReplayRow.model_validate(
        {
            **values,
            "row_sha256": hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest(),
        }
    )


class ReplayImportManifest(StrictModel):
    schema_version: Literal[1] = 1
    replay_compatibility_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    requests_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    captures_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pricing_catalog_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows_file: str = Field(min_length=1)
    rows_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(gt=0)
    manifest_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    def fingerprint_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"manifest_fingerprint"})

    def calculated_fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(self.fingerprint_payload()).encode("utf-8")
        ).hexdigest()

    @model_validator(mode="after")
    def validate_manifest_fingerprint(self) -> Self:
        if Path(self.rows_file).name != self.rows_file:
            raise ValueError("replay rows_file must be a local filename")
        if self.manifest_fingerprint != self.calculated_fingerprint():
            raise ValueError("replay import manifest fingerprint does not match contents")
        return self


class ReplayInspection(StrictModel):
    row_count: int = Field(gt=0)
    request_count: int = Field(gt=0)
    covered_request_count: int = Field(ge=0)
    request_coverage: float = Field(ge=0, le=1)
    rows_by_action: dict[str, int]
    safety_status_counts: dict[str, int]
    provider_fee_usd: float = Field(ge=0)
    energy_known_count: int = Field(ge=0)
    energy_known_rate: float = Field(ge=0, le=1)


@dataclass(frozen=True)
class ReplayImportResult:
    rows_path: Path
    manifest_path: Path
    rows_artifact: RoutingArtifact
    manifest: ReplayImportManifest


@dataclass(frozen=True)
class ReplayCatalog:
    rows: tuple[StrongReplayRow, ...]
    rows_by_key: dict[tuple[str, str], StrongReplayRow]

    def row_for(self, request_id: str, action_id: str) -> StrongReplayRow | None:
        return self.rows_by_key.get((request_id, action_id))

    def action_ids_for_request(self, request_id: str) -> tuple[str, ...]:
        return tuple(sorted(row.action_id for row in self.rows if row.request_id == request_id))


def replay_compatibility_fingerprint(
    config: RoutingConfig,
    registry: ActionRegistry,
) -> str:
    replay_behavior = config.replay.model_dump(
        mode="json",
        exclude={"directory", "pricing_path"},
    )
    replay_actions = {
        action_id: registered.action.action_fingerprint
        for action_id, registered in registry.actions.items()
        if registered.action.kind is ActionKind.STRONG_REPLAY
    }
    payload = {
        "schema_version": 1,
        "replay_behavior": replay_behavior,
        "replay_actions": dict(sorted(replay_actions.items())),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_jsonl(
    path: Path,
    contract: type[ReplayT],
) -> list[ReplayT]:
    records: list[ReplayT] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(contract.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid replay row at {path}:{line_number}") from error
    if not records:
        raise ValueError(f"replay artifact contains no rows: {path}")
    return records


def load_replay_captures(path: Path) -> list[ReplayCapture]:
    return _load_jsonl(path, ReplayCapture)


def load_strong_replay_rows(path: Path) -> list[StrongReplayRow]:
    rows = _load_jsonl(path, StrongReplayRow)
    keys = [(row.request_id, row.action_id) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"replay rows contain duplicate request/action pairs: {path}")
    return rows


def load_replay_import_manifest(path: Path) -> ReplayImportManifest:
    try:
        return ReplayImportManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid replay import manifest: {path}") from error


def _validate_pricing(
    row: ReplayCapture | StrongReplayRow,
    action: WorkflowAction,
    pricing: PricingCatalog,
) -> None:
    if row.pricing_schedule_id != action.pricing_schedule_id:
        raise ValueError("replay pricing schedule does not match action")
    try:
        entry = pricing.entries[row.pricing_schedule_id]
    except KeyError as error:
        raise ValueError(f"unknown replay pricing schedule: {row.pricing_schedule_id}") from error
    expected_identity = (
        action.provider_id,
        action.model_id,
        action.model_version,
    )
    if (entry.provider_id, entry.model_id, entry.model_version) != expected_identity:
        raise ValueError("pricing entry identity does not match replay action")
    expected_fee = entry.provider_fee(row.prompt_tokens, row.completion_tokens)
    if not math.isclose(row.provider_fee_usd, expected_fee, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"replay provider fee does not match pricing schedule: "
            f"expected {expected_fee:.12g}, got {row.provider_fee_usd:.12g}"
        )


def _validate_replay_record(
    row: ReplayCapture | StrongReplayRow,
    requests_by_id: Mapping[str, WorkflowRequest],
    registry: ActionRegistry,
    pricing: PricingCatalog,
    config: RoutingConfig,
) -> None:
    try:
        request = requests_by_id[row.request_id]
    except KeyError as error:
        raise ValueError(f"replay row references an unknown request: {row.request_id}") from error
    try:
        registered = registry.actions[row.action_id]
    except KeyError as error:
        raise ValueError(f"replay row references an unknown action: {row.action_id}") from error
    action = registered.action
    if action.kind is not ActionKind.STRONG_REPLAY or action.executor_id != REPLAY_EXECUTOR_ID:
        raise ValueError("replay row action is not a strong-model replay action")
    if row.request_fingerprint != request.request_fingerprint:
        raise ValueError("replay request fingerprint does not match current request")
    if row.action_fingerprint != action.action_fingerprint:
        raise ValueError("replay action fingerprint does not match current action")
    if (row.provider_id, row.model_id, row.model_version) != (
        action.provider_id,
        action.model_id,
        action.model_version,
    ):
        raise ValueError("replay provider/model identity does not match action")
    if action.max_new_tokens is not None and row.completion_tokens > action.max_new_tokens:
        raise ValueError("replay completion tokens exceed the action token budget")
    if config.replay.require_verified_safety_status and row.provider_safety.status not in {
        SafetyStatus.SAFE,
        SafetyStatus.BLOCKED,
    }:
        raise ValueError("replay row requires a verified provider safety status")
    _validate_pricing(row, action, pricing)
    if (
        isinstance(row, StrongReplayRow)
        and row.pricing_catalog_fingerprint != pricing.catalog_fingerprint
    ):
        raise ValueError("replay row pricing fingerprint does not match current catalog")


def verify_replay_rows(
    rows: Sequence[StrongReplayRow],
    requests: Sequence[WorkflowRequest],
    registry: ActionRegistry,
    pricing: PricingCatalog,
    config: RoutingConfig,
) -> ReplayCatalog:
    requests_by_id = {request.request_id: request for request in requests}
    if len(requests_by_id) != len(requests):
        raise ValueError("workflow requests contain duplicate request IDs")
    rows_by_key: dict[tuple[str, str], StrongReplayRow] = {}
    for row in rows:
        _validate_replay_record(row, requests_by_id, registry, pricing, config)
        key = (row.request_id, row.action_id)
        if key in rows_by_key:
            raise ValueError("replay rows contain duplicate request/action pairs")
        rows_by_key[key] = row
    if not rows_by_key:
        raise ValueError("verified replay catalog requires at least one row")
    ordered = tuple(rows_by_key[key] for key in sorted(rows_by_key))
    return ReplayCatalog(rows=ordered, rows_by_key=rows_by_key)


def _records_artifact(path: Path, rows: Sequence[StrongReplayRow]) -> RoutingArtifact:
    content = (
        "\n".join(canonical_json(row.model_dump(mode="json")) for row in rows) + "\n"
    ).encode("utf-8")
    _write_atomic(path, content)
    return RoutingArtifact(path=path, sha256=sha256_bytes(content), row_count=len(rows))


def import_replay_captures(
    captures_path: Path,
    requests_path: Path,
    output_path: Path,
    config: RoutingConfig,
    registry: ActionRegistry,
    pricing: PricingCatalog,
    *,
    overwrite: bool = False,
) -> ReplayImportResult:
    """Validate raw captures and publish canonical, self-verifying replay rows."""

    manifest_path = output_path.with_suffix(".manifest.json")
    if (output_path.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError("replay artifacts already exist; use overwrite explicitly")
    requests = load_workflow_requests(requests_path)
    requests_by_id = {request.request_id: request for request in requests}
    captures = load_replay_captures(captures_path)
    capture_keys = [(capture.request_id, capture.action_id) for capture in captures]
    if len(set(capture_keys)) != len(capture_keys):
        raise ValueError("replay captures contain duplicate request/action pairs")
    rows: list[StrongReplayRow] = []
    for capture in captures:
        _validate_replay_record(capture, requests_by_id, registry, pricing, config)
        rows.append(create_strong_replay_row(capture, pricing.catalog_fingerprint))
    rows.sort(key=lambda row: (row.request_id, row.action_id))
    verify_replay_rows(rows, requests, registry, pricing, config)
    rows_artifact = _records_artifact(output_path, rows)
    manifest_values: dict[str, object] = {
        "schema_version": 1,
        "replay_compatibility_fingerprint": replay_compatibility_fingerprint(
            config,
            registry,
        ),
        "requests_sha256": sha256_bytes(requests_path.read_bytes()),
        "captures_sha256": sha256_bytes(captures_path.read_bytes()),
        "pricing_catalog_fingerprint": pricing.catalog_fingerprint,
        "rows_file": output_path.name,
        "rows_sha256": rows_artifact.sha256,
        "row_count": rows_artifact.row_count,
    }
    manifest = ReplayImportManifest.model_validate(
        {
            **manifest_values,
            "manifest_fingerprint": hashlib.sha256(
                canonical_json(manifest_values).encode("utf-8")
            ).hexdigest(),
        }
    )
    _write_atomic(
        manifest_path,
        (canonical_json(manifest.model_dump(mode="json")) + "\n").encode("utf-8"),
    )
    return ReplayImportResult(
        rows_path=output_path,
        manifest_path=manifest_path,
        rows_artifact=rows_artifact,
        manifest=manifest,
    )


def load_verified_replay_catalog(
    rows_path: Path,
    requests_path: Path,
    config: RoutingConfig,
    registry: ActionRegistry,
    pricing: PricingCatalog,
) -> ReplayCatalog:
    rows = load_strong_replay_rows(rows_path)
    requests = load_workflow_requests(requests_path)
    manifest_path = rows_path.with_suffix(".manifest.json")
    manifest = load_replay_import_manifest(manifest_path)
    if manifest.rows_file != rows_path.name:
        raise ValueError("replay import manifest references a different rows file")
    if manifest.rows_sha256 != sha256_bytes(rows_path.read_bytes()):
        raise ValueError("replay rows hash does not match import manifest")
    if manifest.row_count != len(rows):
        raise ValueError("replay row count does not match import manifest")
    if manifest.requests_sha256 != sha256_bytes(requests_path.read_bytes()):
        raise ValueError("replay requests hash does not match import manifest")
    if manifest.replay_compatibility_fingerprint != replay_compatibility_fingerprint(
        config,
        registry,
    ):
        raise ValueError("replay import manifest uses incompatible replay behavior")
    if manifest.pricing_catalog_fingerprint != pricing.catalog_fingerprint:
        raise ValueError("replay import manifest uses a different pricing catalog")
    return verify_replay_rows(rows, requests, registry, pricing, config)


def inspect_replay_catalog(
    catalog: ReplayCatalog,
    requests: Sequence[WorkflowRequest],
) -> ReplayInspection:
    request_ids = {request.request_id for request in requests}
    covered_request_ids = {row.request_id for row in catalog.rows}
    if not covered_request_ids.issubset(request_ids):
        raise ValueError("replay catalog contains rows outside the inspected request set")
    safety_counts = Counter(row.provider_safety.status.value for row in catalog.rows)
    action_counts = Counter(row.action_id for row in catalog.rows)
    energy_known_count = sum(
        row.energy_provenance is not EnergyProvenance.UNAVAILABLE for row in catalog.rows
    )
    return ReplayInspection(
        row_count=len(catalog.rows),
        request_count=len(requests),
        covered_request_count=len(covered_request_ids),
        request_coverage=len(covered_request_ids) / len(requests),
        rows_by_action=dict(sorted(action_counts.items())),
        safety_status_counts=dict(sorted(safety_counts.items())),
        provider_fee_usd=sum(row.provider_fee_usd for row in catalog.rows),
        energy_known_count=energy_known_count,
        energy_known_rate=energy_known_count / len(catalog.rows),
    )


def replay_action_ids_for_request(
    catalog: ReplayCatalog,
    request_id: str,
) -> tuple[str, ...]:
    return catalog.action_ids_for_request(request_id)


def execute_strong_replay(
    request: WorkflowRequest,
    action: WorkflowAction,
    availability: ActionAvailability,
    catalog: ReplayCatalog,
) -> ActionOutcome:
    """Materialize one verified replay observation without any provider call."""

    if action.kind is not ActionKind.STRONG_REPLAY or action.executor_id != REPLAY_EXECUTOR_ID:
        raise ValueError("replay executor received an incompatible workflow action")
    if (
        availability.action_id != action.action_id
        or availability.action_fingerprint != action.action_fingerprint
    ):
        raise ValueError("replay availability does not match the workflow action")
    if availability.status is not AvailabilityStatus.AVAILABLE:
        raise ValueError("blocked or unavailable replay actions must not be executed")
    row = catalog.row_for(request.request_id, action.action_id)
    if row is None:
        raise ValueError("verified replay row is missing; replay never fabricates a response")
    if row.request_fingerprint != request.request_fingerprint:
        raise ValueError("verified replay row does not match the current request")
    if row.action_fingerprint != action.action_fingerprint:
        raise ValueError("verified replay row does not match the current action")
    return ActionOutcome(
        request_id=request.request_id,
        request_fingerprint=request.request_fingerprint,
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=ActionOutcomeStatus.COMPLETED,
        availability=availability,
        response=row.response,
        safety=row.provider_safety,
        telemetry=ActionTelemetry(
            wall_latency_ms=row.observed_latency_ms,
            prompt_tokens=row.prompt_tokens,
            completion_tokens=row.completion_tokens,
            provider_fee_usd=row.provider_fee_usd,
            compute_cost_usd=None,
            total_cost_usd=None,
            energy_joules=row.energy_joules,
            energy_provenance=row.energy_provenance,
            energy_measurement_source=row.energy_measurement_source,
            device=f"remote:{row.provider_id}",
        ),
        metadata={
            "executor_id": REPLAY_EXECUTOR_ID,
            "replay_row_sha256": row.row_sha256,
            "provider_id": row.provider_id,
            "model_id": row.model_id,
            "model_version": row.model_version,
            "pricing_schedule_id": row.pricing_schedule_id,
            "pricing_catalog_fingerprint": row.pricing_catalog_fingerprint,
            "captured_at_utc": row.captured_at_utc.isoformat(),
            "capture_source": row.capture_source,
        },
    )
