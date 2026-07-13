"""Full-information evaluation of policy-gated candidate workflows."""

from __future__ import annotations

import hashlib
import json
import math
import random
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import Field, model_validator

from small_models_society.data.prepare import canonical_json
from small_models_society.evaluation import (
    CodeSandbox,
    EvaluationStatus,
    evaluate_predictions,
)
from small_models_society.inference.prompts import clean_response
from small_models_society.inference.runner import ResumeMismatchError, acquire_run_lock
from small_models_society.retrieval.bm25 import (
    BM25Retriever,
    evaluate_ranked_retrieval_metrics,
    retrieval_config_fingerprint,
)
from small_models_society.retrieval.contracts import (
    RankedRetrievalObservation,
    RetrievalMetrics,
    RetrievalRelevanceRecord,
    RetrievalResultStatus,
)
from small_models_society.retrieval.rag import execute_rag
from small_models_society.routing.config import ActionKind, RoutingConfig
from small_models_society.routing.contracts import (
    ActionAvailability,
    ActionOutcome,
    ActionOutcomeStatus,
    AvailabilityStatus,
    EnergyProvenance,
    SafetyAssessment,
    SafetyStatus,
    WorkflowAction,
    WorkflowRequest,
)
from small_models_society.routing.data import (
    RoutingEvaluatorRecord,
    RoutingSplit,
    VerifiedRoutingSplit,
    load_verified_routing_split,
)
from small_models_society.routing.local import LocalModelExecutor
from small_models_society.routing.policy import (
    ActionRuntimeContext,
    evaluate_action_availability,
)
from small_models_society.routing.registry import (
    ActionRegistry,
    RegisteredAction,
)
from small_models_society.routing.replay import (
    ReplayCatalog,
    execute_strong_replay,
    replay_action_ids_for_request,
)
from small_models_society.schemas import (
    Domain,
    PredictionRecord,
    PredictionStatus,
    StrictModel,
)
from small_models_society.tools.calculator import (
    calculator_config_fingerprint,
    calculator_supported,
    execute_calculator,
)

MATRIX_IMPLEMENTATION_VERSION: Literal[1] = 1
_TRANSIENT_RUNTIME_REASONS = frozenset(
    {
        "adapter_artifact_missing",
        "corpus_not_ready",
        "local_generator_not_ready",
        "local_model_not_ready",
        "replay_row_missing",
    }
)


class WorkflowMatrixOptions(StrictModel):
    split: RoutingSplit = RoutingSplit.DEVELOPMENT
    limit: int | None = Field(default=None, gt=0)
    resume: bool = False
    overwrite: bool = False
    fail_fast: bool = False
    checkpoint_interval: int = Field(default=5, gt=0)

    @model_validator(mode="after")
    def validate_collision_policy(self) -> Self:
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        return self


class ActionRunManifest(StrictModel):
    schema_version: Literal[1] = 1
    implementation_version: Literal[1] = MATRIX_IMPLEMENTATION_VERSION
    action_id: str = Field(min_length=1)
    action_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    executor_id: str = Field(min_length=1)
    requests_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_ids: tuple[str, ...] = Field(min_length=1)
    request_fingerprints: tuple[str, ...] = Field(min_length=1)
    outcomes_file: str = Field(min_length=1)
    run_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    def fingerprint_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"run_fingerprint"})

    def calculated_fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(self.fingerprint_payload()).encode("utf-8")
        ).hexdigest()

    @model_validator(mode="after")
    def validate_request_slice_and_fingerprint(self) -> Self:
        if len(self.request_ids) != len(self.request_fingerprints):
            raise ValueError("action manifest request IDs and fingerprints must align")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("action manifest request IDs must be unique")
        if Path(self.outcomes_file).name != self.outcomes_file:
            raise ValueError("action manifest outcomes_file must be a local filename")
        if self.run_fingerprint != self.calculated_fingerprint():
            raise ValueError("action run fingerprint does not match manifest contents")
        return self


class BootstrapInterval(StrictModel):
    estimate: float
    lower: float
    upper: float
    confidence_level: float = Field(gt=0, lt=1)
    resamples: int = Field(gt=0)


class RetrievalMatrixAnalysis(StrictModel):
    candidate_request_count: int = Field(gt=0)
    observed_request_count: int = Field(ge=0)
    observation_coverage: float = Field(ge=0, le=1)
    metrics: RetrievalMetrics | None = None

    @model_validator(mode="after")
    def validate_observation_coverage(self) -> Self:
        if self.observed_request_count > self.candidate_request_count:
            raise ValueError("observed retrieval requests exceed candidate requests")
        expected = self.observed_request_count / self.candidate_request_count
        if not math.isclose(self.observation_coverage, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("retrieval observation coverage does not match counts")
        if (self.observed_request_count == 0) != (self.metrics is None):
            raise ValueError("retrieval metrics require at least one observed request")
        return self


class ScoredActionCell(StrictModel):
    request_id: str
    action_id: str
    domain: Domain
    outcome_status: ActionOutcomeStatus
    quality_metric: str
    quality_score: float = Field(ge=0, le=1)
    safety_status: SafetyStatus
    wall_latency_ms: float | None = Field(default=None, ge=0)
    provider_fee_usd: float | None = Field(default=None, ge=0)
    energy_joules: float | None = Field(default=None, ge=0)
    energy_provenance: EnergyProvenance
    energy_measurement_source: str | None = Field(default=None, min_length=1)


@dataclass(frozen=True)
class WorkflowMatrixPlan:
    request_count: int
    action_count: int
    cell_count: int
    completed_cell_count: int
    pending_cell_count: int


@dataclass(frozen=True)
class WorkflowMatrixResult:
    outcomes_path: Path
    summary_path: Path
    report_path: Path
    summary: dict[str, Any]
    action_outcome_paths: dict[str, Path]


class MatrixRuntime(Protocol):
    def availability(
        self,
        request: WorkflowRequest,
        registered: RegisteredAction,
    ) -> ActionAvailability: ...

    def execute(
        self,
        request: WorkflowRequest,
        registered: RegisteredAction,
        availability: ActionAvailability,
    ) -> ActionOutcome: ...

    def outcome_is_current(
        self,
        outcome: ActionOutcome,
        request: WorkflowRequest,
        registered: RegisteredAction,
        availability: ActionAvailability,
    ) -> bool: ...

    def retrieval_metrics(
        self,
        outcomes: Sequence[ActionOutcome],
        selected_request_ids: frozenset[str],
    ) -> RetrievalMatrixAnalysis | None: ...


RuntimeFactory = Callable[[VerifiedRoutingSplit], MatrixRuntime]
PlanCallback = Callable[[WorkflowMatrixPlan], None]


class CandidateWorkflowRuntime:
    """Production dispatch across calculator, local, RAG, and replay actions."""

    def __init__(
        self,
        config: RoutingConfig,
        registry: ActionRegistry,
        *,
        local: LocalModelExecutor | None = None,
        retriever: BM25Retriever | None = None,
        replay: ReplayCatalog | None = None,
        retrieval_relevance: Sequence[RetrievalRelevanceRecord] = (),
        local_execution_fingerprints: Mapping[str, str] | None = None,
    ) -> None:
        if registry.routing_config_fingerprint != config.fingerprint():
            raise ValueError("action registry does not match routing configuration")
        if retriever is not None and retriever.config.corpus_id != config.retrieval.corpus_id:
            raise ValueError("retriever corpus does not match routing configuration")
        self.config = config
        self.registry = registry
        self.local = local
        self.retriever = retriever
        self.replay = replay
        self.retrieval_relevance = tuple(retrieval_relevance)
        self.local_execution_fingerprints = dict(local_execution_fingerprints or {})
        if local is not None:
            for action_id, registered in registry.actions.items():
                if registered.action.kind is not ActionKind.LOCAL_MODEL:
                    continue
                try:
                    fingerprint = local.execution_fingerprint(registered.action)
                except ValueError:
                    continue
                existing = self.local_execution_fingerprints.get(action_id)
                if existing is not None and existing != fingerprint:
                    raise ValueError("local execution fingerprint mapping is inconsistent")
                self.local_execution_fingerprints[action_id] = fingerprint

    def _runtime_context(self, request: WorkflowRequest) -> ActionRuntimeContext:
        return ActionRuntimeContext(
            local_model_ready=self.local is not None,
            verified_adapter_ids=(
                self.local.verified_adapter_ids if self.local is not None else ()
            ),
            available_corpus_ids=(
                (self.retriever.config.corpus_id,) if self.retriever is not None else ()
            ),
            replay_action_ids=(
                replay_action_ids_for_request(self.replay, request.request_id)
                if self.replay is not None
                else ()
            ),
            calculator_supported=calculator_supported(request, self.config.calculator),
        )

    def availability(
        self,
        request: WorkflowRequest,
        registered: RegisteredAction,
    ) -> ActionAvailability:
        return evaluate_action_availability(
            request,
            registered,
            self._runtime_context(request),
        )

    def execute(
        self,
        request: WorkflowRequest,
        registered: RegisteredAction,
        availability: ActionAvailability,
    ) -> ActionOutcome:
        action = registered.action
        if action.kind is ActionKind.TOOL:
            return execute_calculator(request, action, availability, self.config.calculator)
        if action.kind is ActionKind.LOCAL_MODEL:
            if self.local is None:
                raise ValueError("local action execution requires a local model executor")
            return self.local.execute(request, action, availability)
        if action.kind is ActionKind.RETRIEVAL:
            if self.local is None or self.retriever is None:
                raise ValueError("RAG execution requires local model and retrieval runtimes")
            base_action = self.registry.actions["local.qwen-base.v1"].action
            return execute_rag(
                request,
                action,
                availability,
                self.config,
                self.retriever,
                self.local.generation_backend(),
                generator_execution_fingerprint=self.local.execution_fingerprint(base_action),
            )
        if action.kind is ActionKind.STRONG_REPLAY:
            if self.replay is None:
                raise ValueError("strong replay execution requires a verified replay catalog")
            return execute_strong_replay(request, action, availability, self.replay)
        raise TypeError(f"unsupported workflow action kind: {action.kind}")

    def outcome_is_current(
        self,
        outcome: ActionOutcome,
        request: WorkflowRequest,
        registered: RegisteredAction,
        availability: ActionAvailability,
    ) -> bool:
        action = registered.action
        if outcome.request_fingerprint != request.request_fingerprint:
            return False
        if outcome.action_fingerprint != action.action_fingerprint:
            return False
        if outcome.status is ActionOutcomeStatus.ERROR:
            return False
        if outcome.status in {
            ActionOutcomeStatus.BLOCKED,
            ActionOutcomeStatus.UNAVAILABLE,
        }:
            return outcome.availability == availability
        if availability.status is AvailabilityStatus.BLOCKED:
            return False
        if (
            availability.status is AvailabilityStatus.UNAVAILABLE
            and availability.reason_code not in _TRANSIENT_RUNTIME_REASONS
        ):
            return False

        metadata = outcome.metadata
        if action.kind is ActionKind.TOOL:
            return metadata.get("calculator_config_fingerprint") == (
                calculator_config_fingerprint(self.config.calculator)
            )
        if action.kind is ActionKind.LOCAL_MODEL:
            expected = self.local_execution_fingerprints.get(action.action_id)
            matches = expected is not None and metadata.get("execution_fingerprint") == expected
            if not matches and availability.reason_code in _TRANSIENT_RUNTIME_REASONS:
                raise ResumeMismatchError(
                    f"cannot verify historical local outcome for {action.action_id}"
                )
            return matches
        if action.kind is ActionKind.RETRIEVAL:
            base_action = self.registry.actions["local.qwen-base.v1"].action
            generator_fingerprint = self.local_execution_fingerprints.get(base_action.action_id)
            matches = (
                self.retriever is not None
                and generator_fingerprint is not None
                and metadata.get("corpus_fingerprint") == self.retriever.corpus_fingerprint
                and metadata.get("retrieval_config_fingerprint")
                == retrieval_config_fingerprint(self.config.retrieval)
                and metadata.get("generator_execution_fingerprint") == generator_fingerprint
            )
            if not matches and availability.reason_code in _TRANSIENT_RUNTIME_REASONS:
                raise ResumeMismatchError(
                    f"cannot verify historical RAG outcome for {action.action_id}"
                )
            return matches
        if action.kind is ActionKind.STRONG_REPLAY:
            row = (
                self.replay.row_for(request.request_id, action.action_id)
                if self.replay is not None
                else None
            )
            matches = row is not None and metadata.get("replay_row_sha256") == row.row_sha256
            if not matches and availability.reason_code in _TRANSIENT_RUNTIME_REASONS:
                raise ResumeMismatchError(
                    f"cannot verify historical replay outcome for {action.action_id}"
                )
            return matches
        return False

    def retrieval_metrics(
        self,
        outcomes: Sequence[ActionOutcome],
        selected_request_ids: frozenset[str],
    ) -> RetrievalMatrixAnalysis | None:
        selected_relevance = [
            record
            for record in self.retrieval_relevance
            if record.request_id in selected_request_ids
        ]
        if not selected_relevance:
            return None
        outcomes_by_key = {(outcome.request_id, outcome.action_id): outcome for outcome in outcomes}
        observations: list[RankedRetrievalObservation] = []
        observed_relevance: list[RetrievalRelevanceRecord] = []
        rag_action_id = "rag.bm25-qwen-base.v1"
        observed_count = 0
        for relevance in selected_relevance:
            outcome = outcomes_by_key.get((relevance.request_id, rag_action_id))
            if outcome is None:
                raise RuntimeError("complete matrix is missing a RAG outcome")
            metadata = outcome.metadata
            if "retrieval_status" not in metadata:
                if outcome.status in {
                    ActionOutcomeStatus.BLOCKED,
                    ActionOutcomeStatus.UNAVAILABLE,
                }:
                    continue
                raise RuntimeError("RAG outcome is missing persisted retrieval metadata")
            observed_count += 1
            try:
                status = RetrievalResultStatus(str(metadata["retrieval_status"]))
                top_k = int(metadata["retrieval_top_k"])
                latency_ms = float(metadata["retrieval_latency_ms"])
                raw_ids = metadata["hit_document_ids"]
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("RAG outcome has invalid retrieval metadata") from error
            if not isinstance(raw_ids, list) or not all(
                isinstance(document_id, str) for document_id in raw_ids
            ):
                raise RuntimeError("RAG outcome has invalid ranked document IDs")
            observations.append(
                RankedRetrievalObservation(
                    request_id=relevance.request_id,
                    status=status,
                    top_k=top_k,
                    ranked_document_ids=tuple(raw_ids),
                    latency_ms=latency_ms,
                )
            )
            observed_relevance.append(relevance)
        if observed_count == 0:
            return RetrievalMatrixAnalysis(
                candidate_request_count=len(selected_relevance),
                observed_request_count=0,
                observation_coverage=0,
            )
        return RetrievalMatrixAnalysis(
            candidate_request_count=len(selected_relevance),
            observed_request_count=observed_count,
            observation_coverage=observed_count / len(selected_relevance),
            metrics=evaluate_ranked_retrieval_metrics(
                observations,
                observed_relevance,
                self.config.retrieval.top_k_values,
            ),
        )


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _action_paths(output_dir: Path, action_id: str) -> tuple[Path, Path]:
    action_dir = output_dir / "actions" / action_id
    return action_dir / "outcomes.jsonl", action_dir / "manifest.json"


def _action_manifest(
    action: WorkflowAction,
    requests: Sequence[WorkflowRequest],
    requests_sha256: str,
) -> ActionRunManifest:
    values: dict[str, object] = {
        "schema_version": 1,
        "implementation_version": MATRIX_IMPLEMENTATION_VERSION,
        "action_id": action.action_id,
        "action_fingerprint": action.action_fingerprint,
        "executor_id": action.executor_id,
        "requests_sha256": requests_sha256,
        "request_ids": [request.request_id for request in requests],
        "request_fingerprints": [request.request_fingerprint for request in requests],
        "outcomes_file": "outcomes.jsonl",
    }
    return ActionRunManifest.model_validate(
        {
            **values,
            "run_fingerprint": hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest(),
        }
    )


def _write_action_manifest(path: Path, manifest: ActionRunManifest) -> None:
    _write_atomic(
        path,
        (canonical_json(manifest.model_dump(mode="json")) + "\n").encode("utf-8"),
    )


def _load_action_manifest(path: Path) -> ActionRunManifest:
    try:
        return ActionRunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ResumeMismatchError(f"invalid action run manifest: {path}") from error


def _write_outcomes(path: Path, outcomes: Sequence[ActionOutcome]) -> None:
    lines = [canonical_json(outcome.model_dump(mode="json")) for outcome in outcomes]
    content = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
    _write_atomic(path, content)


def _load_outcomes(path: Path) -> list[ActionOutcome]:
    outcomes: list[ActionOutcome] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                outcomes.append(ActionOutcome.model_validate_json(line))
            except ValueError as error:
                raise ResumeMismatchError(
                    f"invalid action outcome at {path}:{line_number}"
                ) from error
    return outcomes


def _existing_action_outcomes(
    output_dir: Path,
    action: WorkflowAction,
    requests: Sequence[WorkflowRequest],
    requests_sha256: str,
    options: WorkflowMatrixOptions,
) -> tuple[Path, Path, ActionRunManifest, dict[str, ActionOutcome], bool]:
    outcomes_path, manifest_path = _action_paths(output_dir, action.action_id)
    expected_manifest = _action_manifest(action, requests, requests_sha256)
    outcomes_exist = outcomes_path.exists()
    manifest_exists = manifest_path.exists()
    if options.resume:
        if outcomes_exist != manifest_exists:
            raise ResumeMismatchError(f"action {action.action_id} has only one run artifact")
        if not outcomes_exist:
            return outcomes_path, manifest_path, expected_manifest, {}, True
        existing_manifest = _load_action_manifest(manifest_path)
        if existing_manifest.run_fingerprint != expected_manifest.run_fingerprint:
            raise ResumeMismatchError(f"action {action.action_id} run fingerprint does not match")
        outcomes = _load_outcomes(outcomes_path)
    elif options.overwrite:
        outcomes = []
    else:
        if outcomes_exist or manifest_exists:
            raise FileExistsError(
                f"action {action.action_id} artifacts already exist; use resume or overwrite"
            )
        outcomes = []

    requests_by_id = {request.request_id: request for request in requests}
    outcomes_by_id: dict[str, ActionOutcome] = {}
    for outcome in outcomes:
        request = requests_by_id.get(outcome.request_id)
        if request is None:
            raise ResumeMismatchError(
                f"action {action.action_id} outcome references an unknown request"
            )
        if outcome.request_fingerprint != request.request_fingerprint:
            raise ResumeMismatchError("existing outcome request fingerprint does not match")
        if outcome.action_id != action.action_id:
            raise ResumeMismatchError("existing outcome action ID does not match")
        if outcome.action_fingerprint != action.action_fingerprint:
            raise ResumeMismatchError("existing outcome action fingerprint does not match")
        if outcome.request_id in outcomes_by_id:
            raise ResumeMismatchError("existing action outcomes contain duplicate requests")
        outcomes_by_id[outcome.request_id] = outcome
    return outcomes_path, manifest_path, expected_manifest, outcomes_by_id, not outcomes_exist


def _non_available_outcome(
    request: WorkflowRequest,
    action: WorkflowAction,
    availability: ActionAvailability,
) -> ActionOutcome:
    if availability.status is AvailabilityStatus.AVAILABLE:
        raise ValueError("available actions require executor dispatch")
    blocked = availability.status is AvailabilityStatus.BLOCKED
    return ActionOutcome(
        request_id=request.request_id,
        request_fingerprint=request.request_fingerprint,
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=(ActionOutcomeStatus.BLOCKED if blocked else ActionOutcomeStatus.UNAVAILABLE),
        availability=availability,
        safety=SafetyAssessment(
            status=(SafetyStatus.BLOCKED if blocked else SafetyStatus.NOT_ASSESSED),
            source=("policy" if blocked else "availability"),
            rule_ids=(availability.rule_ids if blocked else ())
            or (("policy.unspecified.v1",) if blocked else ()),
        ),
    )


def _ordered_action_outcomes(
    requests: Sequence[WorkflowRequest],
    outcomes_by_id: Mapping[str, ActionOutcome],
) -> list[ActionOutcome]:
    return [
        outcomes_by_id[request.request_id]
        for request in requests
        if request.request_id in outcomes_by_id
    ]


def _run_action(
    output_dir: Path,
    registered: RegisteredAction,
    requests: Sequence[WorkflowRequest],
    requests_sha256: str,
    runtime: MatrixRuntime,
    options: WorkflowMatrixOptions,
) -> tuple[list[ActionOutcome], Path]:
    action = registered.action
    (
        outcomes_path,
        manifest_path,
        manifest,
        outcomes_by_id,
        initialize,
    ) = _existing_action_outcomes(
        output_dir,
        action,
        requests,
        requests_sha256,
        options,
    )
    if initialize or options.overwrite:
        _write_action_manifest(manifest_path, manifest)
        _write_outcomes(outcomes_path, [])

    completed_since_checkpoint = 0
    for request in requests:
        availability = runtime.availability(request, registered)
        existing = outcomes_by_id.get(request.request_id)
        if existing is not None and runtime.outcome_is_current(
            existing,
            request,
            registered,
            availability,
        ):
            continue
        try:
            outcome = (
                runtime.execute(request, registered, availability)
                if availability.status is AvailabilityStatus.AVAILABLE
                else _non_available_outcome(request, action, availability)
            )
        except BaseException:
            _write_outcomes(
                outcomes_path,
                _ordered_action_outcomes(requests, outcomes_by_id),
            )
            raise
        if outcome.request_id != request.request_id:
            raise ValueError("executor returned an outcome for a different request")
        if outcome.action_id != action.action_id:
            raise ValueError("executor returned an outcome for a different action")
        outcomes_by_id[request.request_id] = outcome
        completed_since_checkpoint += 1
        if options.fail_fast and outcome.status is ActionOutcomeStatus.ERROR:
            _write_outcomes(
                outcomes_path,
                _ordered_action_outcomes(requests, outcomes_by_id),
            )
            raise RuntimeError(f"action {action.action_id} failed for request {request.request_id}")
        if completed_since_checkpoint >= options.checkpoint_interval:
            _write_outcomes(
                outcomes_path,
                _ordered_action_outcomes(requests, outcomes_by_id),
            )
            completed_since_checkpoint = 0

    ordered = _ordered_action_outcomes(requests, outcomes_by_id)
    if len(ordered) != len(requests):
        raise RuntimeError(f"action {action.action_id} did not produce a complete outcome slice")
    _write_outcomes(outcomes_path, ordered)
    return ordered, outcomes_path


def _selected_split(
    verified: VerifiedRoutingSplit,
    limit: int | None,
) -> tuple[list[WorkflowRequest], list[RoutingEvaluatorRecord]]:
    requests = list(verified.requests)
    if limit is not None:
        requests = requests[:limit]
    if not requests:
        raise ValueError("workflow matrix selected no requests")
    evaluators_by_id = {record.request_id: record for record in verified.evaluator_records}
    return requests, [evaluators_by_id[request.request_id] for request in requests]


def _check_aggregate_collision(
    output_dir: Path,
    options: WorkflowMatrixOptions,
) -> None:
    paths = _aggregate_paths(output_dir)
    if any(path.exists() for path in paths) and not options.resume and not options.overwrite:
        raise FileExistsError(
            "workflow matrix aggregate artifacts already exist; use resume or overwrite"
        )


def _aggregate_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "action_outcomes.jsonl",
        output_dir / "workflow_matrix_summary.json",
        output_dir / "workflow_matrix_report.md",
    )


def _invalidate_aggregate_artifacts(output_dir: Path) -> None:
    for path in _aggregate_paths(output_dir):
        path.unlink(missing_ok=True)


def _action_reusable_count(
    output_dir: Path,
    registered: RegisteredAction,
    requests: Sequence[WorkflowRequest],
    requests_sha256: str,
    runtime: MatrixRuntime,
    options: WorkflowMatrixOptions,
) -> int:
    _, _, _, outcomes_by_id, _ = _existing_action_outcomes(
        output_dir,
        registered.action,
        requests,
        requests_sha256,
        options,
    )
    reusable = 0
    for request in requests:
        outcome = outcomes_by_id.get(request.request_id)
        if outcome is None:
            continue
        availability = runtime.availability(request, registered)
        if runtime.outcome_is_current(
            outcome,
            request,
            registered,
            availability,
        ):
            reusable += 1
    return reusable


def inspect_workflow_matrix(
    requests_path: Path,
    evaluator_path: Path,
    data_manifest_path: Path,
    output_dir: Path,
    registry: ActionRegistry,
    runtime: MatrixRuntime,
    options: WorkflowMatrixOptions | None = None,
) -> WorkflowMatrixPlan:
    resolved = options or WorkflowMatrixOptions()
    verified = load_verified_routing_split(
        requests_path,
        evaluator_path,
        data_manifest_path,
        resolved.split,
    )
    requests, _ = _selected_split(verified, resolved.limit)
    lock_target = output_dir / "action_outcomes.jsonl"
    with acquire_run_lock(lock_target):
        _check_aggregate_collision(output_dir, resolved)
        reusable = _reusable_cell_count(
            verified,
            output_dir,
            registry,
            runtime,
            resolved,
        )
    cell_count = len(requests) * len(registry.actions)
    return WorkflowMatrixPlan(
        request_count=len(requests),
        action_count=len(registry.actions),
        cell_count=cell_count,
        completed_cell_count=reusable,
        pending_cell_count=cell_count - reusable,
    )


def _reusable_cell_count(
    verified: VerifiedRoutingSplit,
    output_dir: Path,
    registry: ActionRegistry,
    runtime: MatrixRuntime,
    options: WorkflowMatrixOptions,
) -> int:
    requests, _ = _selected_split(verified, options.limit)
    return sum(
        _action_reusable_count(
            output_dir,
            registry.actions[action_id],
            requests,
            verified.requests_sha256,
            runtime,
            options,
        )
        for action_id in sorted(registry.actions)
    )


def _workflow_matrix_plan_unlocked(
    verified: VerifiedRoutingSplit,
    output_dir: Path,
    registry: ActionRegistry,
    runtime: MatrixRuntime,
    options: WorkflowMatrixOptions,
) -> WorkflowMatrixPlan:
    requests, _ = _selected_split(verified, options.limit)
    reusable = _reusable_cell_count(
        verified,
        output_dir,
        registry,
        runtime,
        options,
    )
    cell_count = len(requests) * len(registry.actions)
    return WorkflowMatrixPlan(
        request_count=len(requests),
        action_count=len(registry.actions),
        cell_count=cell_count,
        completed_cell_count=reusable,
        pending_cell_count=cell_count - reusable,
    )


def _prediction_for_outcome(
    outcome: ActionOutcome,
    evaluator: RoutingEvaluatorRecord,
) -> PredictionRecord:
    telemetry = outcome.telemetry
    latency_ms = telemetry.wall_latency_ms if telemetry is not None else 0
    prompt_tokens = (
        telemetry.prompt_tokens
        if telemetry is not None and telemetry.prompt_tokens is not None
        else 0
    )
    completion_tokens = (
        telemetry.completion_tokens
        if telemetry is not None and telemetry.completion_tokens is not None
        else 0
    )
    cost_usd = (
        telemetry.provider_fee_usd
        if telemetry is not None and telemetry.provider_fee_usd is not None
        else 0
    )
    if outcome.status is ActionOutcomeStatus.COMPLETED:
        assert outcome.response is not None
        return PredictionRecord(
            example_id=evaluator.request_id,
            domain=evaluator.example.domain,
            model_id=outcome.action_id,
            response=clean_response(evaluator.example.domain, outcome.response),
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            metadata={"action_fingerprint": outcome.action_fingerprint},
        )
    if outcome.status is ActionOutcomeStatus.ERROR:
        status = PredictionStatus.ERROR
    else:
        status = PredictionStatus.ABSTAINED
    return PredictionRecord(
        example_id=evaluator.request_id,
        domain=evaluator.example.domain,
        model_id=outcome.action_id,
        status=status,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        metadata={"action_fingerprint": outcome.action_fingerprint},
    )


def _score_outcomes(
    outcomes: Sequence[ActionOutcome],
    evaluators: Sequence[RoutingEvaluatorRecord],
    sandbox: CodeSandbox,
) -> list[ScoredActionCell]:
    execution_errors = [
        outcome for outcome in outcomes if outcome.status is ActionOutcomeStatus.ERROR
    ]
    if execution_errors:
        first = execution_errors[0]
        raise RuntimeError(
            "action infrastructure error for "
            f"{first.request_id}/{first.action_id}: "
            f"{first.error_type or 'unknown error'}"
        )
    evaluators_by_id = {record.request_id: record for record in evaluators}
    cells: list[ScoredActionCell] = []
    for outcome in outcomes:
        evaluator = evaluators_by_id[outcome.request_id]
        prediction = _prediction_for_outcome(outcome, evaluator)
        evaluation = evaluate_predictions(
            [evaluator.example],
            [prediction],
            sandbox,
        )[0]
        if evaluation.status is EvaluationStatus.SANDBOX_ERROR:
            raise RuntimeError(
                f"sandbox infrastructure error for {outcome.request_id}/{outcome.action_id}"
            )
        telemetry = outcome.telemetry
        cells.append(
            ScoredActionCell(
                request_id=outcome.request_id,
                action_id=outcome.action_id,
                domain=evaluator.example.domain,
                outcome_status=outcome.status,
                quality_metric=evaluation.primary_metric,
                quality_score=evaluation.primary_score,
                safety_status=outcome.safety.status,
                wall_latency_ms=(telemetry.wall_latency_ms if telemetry is not None else None),
                provider_fee_usd=(telemetry.provider_fee_usd if telemetry is not None else None),
                energy_joules=(telemetry.energy_joules if telemetry is not None else None),
                energy_provenance=(
                    telemetry.energy_provenance
                    if telemetry is not None
                    else EnergyProvenance.UNAVAILABLE
                ),
                energy_measurement_source=(
                    telemetry.energy_measurement_source if telemetry is not None else None
                ),
            )
        )
    return cells


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
    label: str,
) -> BootstrapInterval:
    if not values:
        raise ValueError("bootstrap interval requires at least one observation")
    label_seed = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:16], 16)
    generator = random.Random(seed ^ label_seed)
    count = len(values)
    means = sorted(
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    alpha = (1 - confidence_level) / 2
    lower_index = max(0, math.floor(alpha * resamples))
    upper_index = min(resamples - 1, math.ceil((1 - alpha) * resamples) - 1)
    return BootstrapInterval(
        estimate=sum(values) / count,
        lower=means[lower_index],
        upper=means[upper_index],
        confidence_level=confidence_level,
        resamples=resamples,
    )


def _safety_feasible(
    outcome: ActionOutcome,
    request: WorkflowRequest,
) -> bool:
    return outcome.safety.status is SafetyStatus.SAFE or (
        outcome.safety.status is SafetyStatus.UNKNOWN and request.policy.allow_unknown_output_safety
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _action_summaries(
    registry: ActionRegistry,
    outcomes: Sequence[ActionOutcome],
    cells: Sequence[ScoredActionCell],
    requests: Sequence[WorkflowRequest],
    config: RoutingConfig,
) -> dict[str, dict[str, Any]]:
    outcomes_by_action: dict[str, list[ActionOutcome]] = defaultdict(list)
    cells_by_action: dict[str, list[ScoredActionCell]] = defaultdict(list)
    requests_by_id = {request.request_id: request for request in requests}
    for outcome in outcomes:
        outcomes_by_action[outcome.action_id].append(outcome)
    for cell in cells:
        cells_by_action[cell.action_id].append(cell)
    summaries: dict[str, dict[str, Any]] = {}
    for action_id in sorted(registry.actions):
        action_outcomes = outcomes_by_action[action_id]
        action_cells = cells_by_action[action_id]
        status_counts = Counter(outcome.status.value for outcome in action_outcomes)
        availability_counts = Counter(
            outcome.availability.status.value for outcome in action_outcomes
        )
        safety_counts = Counter(outcome.safety.status.value for outcome in action_outcomes)
        completed = [
            outcome
            for outcome in action_outcomes
            if outcome.status is ActionOutcomeStatus.COMPLETED
        ]
        telemetry = [
            outcome.telemetry for outcome in action_outcomes if outcome.telemetry is not None
        ]
        fees = [item.provider_fee_usd for item in telemetry if item.provider_fee_usd is not None]
        energies = [
            item.energy_joules
            for item in telemetry
            if item.energy_joules is not None
            and item.energy_provenance
            in {
                EnergyProvenance.MEASURED,
                EnergyProvenance.REPLAY,
            }
        ]
        energy_boundaries = {
            (item.energy_provenance.value, item.energy_measurement_source)
            for item in telemetry
            if item.energy_joules is not None
            and item.energy_provenance in {EnergyProvenance.MEASURED, EnergyProvenance.REPLAY}
        }
        completed_ids = {outcome.request_id for outcome in completed}
        completed_scores = [
            cell.quality_score for cell in action_cells if cell.request_id in completed_ids
        ]
        safety_feasible_completed = sum(
            _safety_feasible(outcome, requests_by_id[outcome.request_id]) for outcome in completed
        )
        action_cells_by_request = {cell.request_id: cell for cell in action_cells}
        hard_constraint_feasible_count = sum(
            outcome.status is ActionOutcomeStatus.COMPLETED
            and _safety_feasible(outcome, requests_by_id[outcome.request_id])
            and action_cells_by_request[outcome.request_id].quality_score
            >= requests_by_id[outcome.request_id].policy.required_quality
            for outcome in action_outcomes
        )
        domain_quality = {
            domain.value: _mean(
                [cell.quality_score for cell in action_cells if cell.domain is domain]
            )
            for domain in Domain
            if any(cell.domain is domain for cell in action_cells)
        }
        coverage_values = [
            float(outcome.status is ActionOutcomeStatus.COMPLETED) for outcome in action_outcomes
        ]
        summaries[action_id] = {
            "action_fingerprint": registry.actions[action_id].action.action_fingerprint,
            "kind": registry.actions[action_id].action.kind.value,
            "request_count": len(action_outcomes),
            "status_counts": {
                status.value: status_counts[status.value] for status in ActionOutcomeStatus
            },
            "availability_counts": {
                status.value: availability_counts[status.value] for status in AvailabilityStatus
            },
            "safety_status_counts": {
                status.value: safety_counts[status.value] for status in SafetyStatus
            },
            "completed_count": len(completed),
            "completion_coverage": len(completed) / len(action_outcomes),
            "completion_coverage_ci": bootstrap_mean_interval(
                coverage_values,
                seed=config.seed,
                resamples=config.analysis.bootstrap_resamples,
                confidence_level=config.analysis.confidence_level,
                label=f"action-coverage:{action_id}",
            ).model_dump(mode="json"),
            "micro_quality": _mean([cell.quality_score for cell in action_cells]),
            "minimum_quality": min(
                (cell.quality_score for cell in action_cells),
                default=0.0,
            ),
            "completed_quality": _mean(completed_scores),
            "domain_quality": domain_quality,
            "safety_feasible_completed_count": safety_feasible_completed,
            "safety_feasible_completed_rate": (
                safety_feasible_completed / len(completed) if completed else 0.0
            ),
            "hard_constraint_feasible_count": hard_constraint_feasible_count,
            "hard_constraint_feasible_rate": (
                hard_constraint_feasible_count / len(action_outcomes)
            ),
            "mean_wall_latency_ms": _mean([item.wall_latency_ms for item in telemetry]),
            "provider_fee_known_count": len(fees),
            "provider_fee_known_rate": (
                len(fees) / len(action_outcomes) if action_outcomes else 0.0
            ),
            "provider_fee_usd": (sum(fees) if len(fees) == len(action_outcomes) else None),
            "mean_provider_fee_usd": _mean(fees),
            "energy_known_count": len(energies),
            "energy_known_rate": len(energies) / len(telemetry) if telemetry else 0.0,
            "mean_energy_joules": _mean(energies),
            "energy_measurement_boundaries": [
                {"provenance": provenance, "source": source}
                for provenance, source in sorted(energy_boundaries)
            ],
        }
    return summaries


def _candidate_cells(
    request: WorkflowRequest,
    outcomes_by_key: Mapping[tuple[str, str], ActionOutcome],
    cells_by_key: Mapping[tuple[str, str], ScoredActionCell],
    quality_floor: float,
) -> list[tuple[ActionOutcome, ScoredActionCell]]:
    candidates: list[tuple[ActionOutcome, ScoredActionCell]] = []
    effective_floor = max(quality_floor, request.policy.required_quality)
    for (request_id, _action_id), outcome in outcomes_by_key.items():
        if request_id != request.request_id:
            continue
        cell = cells_by_key[(outcome.request_id, outcome.action_id)]
        if (
            outcome.status is ActionOutcomeStatus.COMPLETED
            and cell.quality_score >= effective_floor
            and _safety_feasible(outcome, request)
        ):
            candidates.append((outcome, cell))
    return candidates


def _oracle_summary(
    objective: str,
    requests: Sequence[WorkflowRequest],
    outcomes_by_key: Mapping[tuple[str, str], ActionOutcome],
    cells_by_key: Mapping[tuple[str, str], ScoredActionCell],
    quality_floor: float,
) -> dict[str, Any]:
    selected: list[tuple[ActionOutcome, ScoredActionCell]] = []
    action_counts: Counter[str] = Counter()
    energy_boundary: tuple[str, str] | None = None
    if objective == "energy":
        boundary_coverage: Counter[tuple[str, str]] = Counter()
        for request in requests:
            boundaries = {
                (cell.energy_provenance.value, cell.energy_measurement_source)
                for _, cell in _candidate_cells(
                    request,
                    outcomes_by_key,
                    cells_by_key,
                    quality_floor,
                )
                if cell.energy_joules is not None
                and cell.energy_measurement_source is not None
                and cell.energy_provenance in {EnergyProvenance.MEASURED, EnergyProvenance.REPLAY}
            }
            boundary_coverage.update(boundaries)
        if boundary_coverage:
            energy_boundary = min(
                boundary_coverage,
                key=lambda boundary: (-boundary_coverage[boundary], boundary),
            )
    for request in requests:
        candidates = _candidate_cells(
            request,
            outcomes_by_key,
            cells_by_key,
            quality_floor,
        )
        if objective == "energy":
            candidates = [
                candidate
                for candidate in candidates
                if candidate[1].energy_joules is not None
                and candidate[1].energy_measurement_source is not None
                and candidate[1].energy_provenance
                in {EnergyProvenance.MEASURED, EnergyProvenance.REPLAY}
                and (
                    candidate[1].energy_provenance.value,
                    candidate[1].energy_measurement_source,
                )
                == energy_boundary
            ]
        if objective == "provider_fee":
            candidates = [
                candidate for candidate in candidates if candidate[1].provider_fee_usd is not None
            ]
        if objective == "latency":
            candidates = [
                candidate for candidate in candidates if candidate[1].wall_latency_ms is not None
            ]
        if not candidates:
            continue

        def candidate_key(
            candidate: tuple[ActionOutcome, ScoredActionCell],
        ) -> tuple[float, float, float, str]:
            outcome, cell = candidate
            fee = cell.provider_fee_usd if cell.provider_fee_usd is not None else math.inf
            latency = cell.wall_latency_ms if cell.wall_latency_ms is not None else math.inf
            energy = cell.energy_joules if cell.energy_joules is not None else math.inf
            if objective == "quality":
                return (-cell.quality_score, fee, latency, outcome.action_id)
            if objective == "provider_fee":
                return (fee, -cell.quality_score, latency, outcome.action_id)
            if objective == "latency":
                return (latency, -cell.quality_score, fee, outcome.action_id)
            return (energy, -cell.quality_score, fee, outcome.action_id)

        chosen = min(candidates, key=candidate_key)
        selected.append(chosen)
        action_counts[chosen[0].action_id] += 1
    quality_total = sum(cell.quality_score for _, cell in selected)
    fees = [cell.provider_fee_usd for _, cell in selected if cell.provider_fee_usd is not None]
    latencies = [cell.wall_latency_ms for _, cell in selected if cell.wall_latency_ms is not None]
    energies = [cell.energy_joules for _, cell in selected if cell.energy_joules is not None]
    energy_boundaries = {
        (cell.energy_provenance.value, cell.energy_measurement_source)
        for _, cell in selected
        if cell.energy_joules is not None
    }
    fee_complete = len(fees) == len(selected)
    energy_comparable = len(energies) == len(selected) and len(energy_boundaries) == 1
    return {
        "objective": objective,
        "quality_floor": quality_floor,
        "request_count": len(requests),
        "feasible_request_count": len(selected),
        "feasible_request_rate": len(selected) / len(requests),
        "mean_quality": quality_total / len(requests),
        "provider_fee_known_count": len(fees),
        "provider_fee_usd": sum(fees) if fee_complete else None,
        "mean_wall_latency_ms": _mean(latencies),
        "mean_energy_joules": _mean(energies) if energy_comparable else None,
        "energy_measurement_boundary": (
            {"provenance": energy_boundary[0], "source": energy_boundary[1]}
            if energy_comparable and energy_boundary is not None
            else None
        ),
        "action_counts": dict(sorted(action_counts.items())),
    }


def _pareto_frontier(
    action_summaries: Mapping[str, Mapping[str, Any]],
    quality_floor: float,
    *,
    include_energy: bool,
    energy_boundary: tuple[str, str] | None = None,
) -> list[str]:
    candidates: dict[
        str,
        tuple[float, float, float, float | None, tuple[str, str] | None],
    ] = {}
    for action_id, summary in action_summaries.items():
        quality = summary["micro_quality"]
        minimum_quality = summary["minimum_quality"]
        fee = summary["mean_provider_fee_usd"]
        latency = summary["mean_wall_latency_ms"]
        completed = int(summary["completed_count"])
        safety_count = int(summary["safety_feasible_completed_count"])
        energy = summary["mean_energy_joules"]
        boundaries = summary["energy_measurement_boundaries"]
        if (
            quality is None
            or minimum_quality is None
            or fee is None
            or latency is None
            or quality < quality_floor
            or float(minimum_quality) < quality_floor
            or completed == 0
            or completed != int(summary["request_count"])
            or safety_count != completed
            or int(summary["hard_constraint_feasible_count"]) != int(summary["request_count"])
            or float(summary["provider_fee_known_rate"]) < 1.0
        ):
            continue
        if include_energy and (
            energy is None or float(summary["energy_known_rate"]) < 1.0 or len(boundaries) != 1
        ):
            continue
        boundary = (
            (str(boundaries[0]["provenance"]), str(boundaries[0]["source"]))
            if include_energy
            else None
        )
        if energy_boundary is not None and boundary != energy_boundary:
            continue
        candidates[action_id] = (
            float(quality),
            float(fee),
            float(latency),
            float(energy) if energy is not None else None,
            boundary,
        )

    frontier: list[str] = []
    for action_id, point in candidates.items():
        dominated = False
        for other_id, other in candidates.items():
            if other_id == action_id:
                continue
            if include_energy and other[4] != point[4]:
                continue
            no_worse = (
                other[0] >= point[0]
                and other[1] <= point[1]
                and other[2] <= point[2]
                and (
                    not include_energy
                    or (other[3] is not None and point[3] is not None and other[3] <= point[3])
                )
            )
            strictly_better = (
                other[0] > point[0]
                or other[1] < point[1]
                or other[2] < point[2]
                or (
                    include_energy
                    and other[3] is not None
                    and point[3] is not None
                    and other[3] < point[3]
                )
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(action_id)
    return sorted(frontier)


def _energy_pareto_frontiers(
    action_summaries: Mapping[str, Mapping[str, Any]],
    quality_floor: float,
) -> dict[str, list[str]]:
    boundaries = {
        (str(boundary["provenance"]), str(boundary["source"]))
        for summary in action_summaries.values()
        for boundary in summary["energy_measurement_boundaries"]
        if len(summary["energy_measurement_boundaries"]) == 1
    }
    return {
        f"{provenance}:{source}": _pareto_frontier(
            action_summaries,
            quality_floor,
            include_energy=True,
            energy_boundary=(provenance, source),
        )
        for provenance, source in sorted(boundaries)
    }


def _baseline_action_id(
    name: str,
    request: WorkflowRequest,
    outcomes_by_key: Mapping[tuple[str, str], ActionOutcome],
) -> str:
    base = "local.qwen-base.v1"
    calculator = "tool.calculator.v1"
    rag = "rag.bm25-qwen-base.v1"
    replay = "remote.strong-replay.reference.v1"
    if name == "always_local_base":
        return base
    if name == "calculator_first_then_local":
        calculator_outcome = outcomes_by_key[(request.request_id, calculator)]
        return (
            calculator
            if calculator_outcome.availability.status is AvailabilityStatus.AVAILABLE
            else base
        )
    if name == "rag_where_eligible_else_local":
        rag_outcome = outcomes_by_key[(request.request_id, rag)]
        return (
            rag
            if isinstance(request.attributes.get("retrieval_query"), str)
            and rag_outcome.availability.status is AvailabilityStatus.AVAILABLE
            else base
        )
    if name == "strong_replay_where_available_else_local":
        replay_outcome = outcomes_by_key[(request.request_id, replay)]
        return (
            replay if replay_outcome.availability.status is AvailabilityStatus.AVAILABLE else base
        )
    if name == "static_tool_rag_local":
        calculator_outcome = outcomes_by_key[(request.request_id, calculator)]
        if calculator_outcome.availability.status is AvailabilityStatus.AVAILABLE:
            return calculator
        rag_outcome = outcomes_by_key[(request.request_id, rag)]
        if (
            isinstance(request.attributes.get("retrieval_query"), str)
            and rag_outcome.availability.status is AvailabilityStatus.AVAILABLE
        ):
            return rag
        return base
    raise ValueError(f"unknown workflow baseline: {name}")


def _baseline_summaries(
    requests: Sequence[WorkflowRequest],
    outcomes_by_key: Mapping[tuple[str, str], ActionOutcome],
    cells_by_key: Mapping[tuple[str, str], ScoredActionCell],
    config: RoutingConfig,
) -> dict[str, dict[str, Any]]:
    names = (
        "always_local_base",
        "calculator_first_then_local",
        "rag_where_eligible_else_local",
        "strong_replay_where_available_else_local",
        "static_tool_rag_local",
    )
    summaries: dict[str, dict[str, Any]] = {}
    oracle_quality_by_request = {
        request.request_id: max(
            (
                cell.quality_score
                for _, cell in _candidate_cells(
                    request,
                    outcomes_by_key,
                    cells_by_key,
                    0,
                )
            ),
            default=0.0,
        )
        for request in requests
    }
    for name in names:
        selected = [
            (
                outcomes_by_key[
                    (
                        request.request_id,
                        _baseline_action_id(name, request, outcomes_by_key),
                    )
                ],
                cells_by_key[
                    (
                        request.request_id,
                        _baseline_action_id(name, request, outcomes_by_key),
                    )
                ],
                request,
            )
            for request in requests
        ]
        quality_values = [cell.quality_score for _, cell, _ in selected]
        completed_count = sum(
            outcome.status is ActionOutcomeStatus.COMPLETED for outcome, _, _ in selected
        )
        safety_feasible_count = sum(
            _safety_feasible(outcome, request) and outcome.status is ActionOutcomeStatus.COMPLETED
            for outcome, _, request in selected
        )
        constrained_quality_values = [
            cell.quality_score
            if outcome.status is ActionOutcomeStatus.COMPLETED
            and _safety_feasible(outcome, request)
            and cell.quality_score >= request.policy.required_quality
            else 0.0
            for outcome, cell, request in selected
        ]
        fees = [
            cell.provider_fee_usd for _, cell, _ in selected if cell.provider_fee_usd is not None
        ]
        latencies = [
            cell.wall_latency_ms for _, cell, _ in selected if cell.wall_latency_ms is not None
        ]
        mean_quality = sum(quality_values) / len(quality_values)
        oracle_gap_values = [
            oracle_quality_by_request[request.request_id] - constrained_quality
            for (_, _, request), constrained_quality in zip(
                selected,
                constrained_quality_values,
                strict=True,
            )
        ]
        summaries[name] = {
            "request_count": len(requests),
            "mean_quality": mean_quality,
            "quality_ci": bootstrap_mean_interval(
                quality_values,
                seed=config.seed,
                resamples=config.analysis.bootstrap_resamples,
                confidence_level=config.analysis.confidence_level,
                label=f"baseline-quality:{name}",
            ).model_dump(mode="json"),
            "completed_count": completed_count,
            "completion_rate": completed_count / len(requests),
            "safety_feasible_completed_count": safety_feasible_count,
            "safety_feasible_completed_rate": safety_feasible_count / len(requests),
            "provider_fee_known_count": len(fees),
            "provider_fee_usd": (sum(fees) if len(fees) == len(selected) else None),
            "mean_wall_latency_ms": _mean(latencies),
            "constrained_mean_quality": (
                sum(constrained_quality_values) / len(constrained_quality_values)
            ),
            "oracle_gap": sum(oracle_gap_values) / len(oracle_gap_values),
            "oracle_gap_ci": bootstrap_mean_interval(
                oracle_gap_values,
                seed=config.seed,
                resamples=config.analysis.bootstrap_resamples,
                confidence_level=config.analysis.confidence_level,
                label=f"baseline-oracle-gap:{name}",
            ).model_dump(mode="json"),
            "action_counts": dict(
                sorted(Counter(outcome.action_id for outcome, _, _ in selected).items())
            ),
        }
    return summaries


def _build_summary(
    config: RoutingConfig,
    registry: ActionRegistry,
    runtime: MatrixRuntime,
    split: RoutingSplit,
    requests: Sequence[WorkflowRequest],
    outcomes: Sequence[ActionOutcome],
    cells: Sequence[ScoredActionCell],
    *,
    complete_split: bool,
    prepared_prompt_catalog_fingerprint: str,
) -> dict[str, Any]:
    outcomes_by_key = {(outcome.request_id, outcome.action_id): outcome for outcome in outcomes}
    cells_by_key = {(cell.request_id, cell.action_id): cell for cell in cells}
    action_summaries = _action_summaries(
        registry,
        outcomes,
        cells,
        requests,
        config,
    )
    floor_summaries: dict[str, Any] = {}
    for floor in config.analysis.quality_floors:
        best_quality = _oracle_summary(
            "quality",
            requests,
            outcomes_by_key,
            cells_by_key,
            floor,
        )
        floor_summaries[str(floor)] = {
            "pareto_frontier": _pareto_frontier(
                action_summaries,
                floor,
                include_energy=False,
            ),
            "energy_pareto_frontier": _pareto_frontier(
                action_summaries,
                floor,
                include_energy=True,
            ),
            "energy_pareto_frontiers": _energy_pareto_frontiers(
                action_summaries,
                floor,
            ),
            "best_quality_oracle": best_quality,
            "cheapest_feasible_oracle": _oracle_summary(
                "provider_fee",
                requests,
                outcomes_by_key,
                cells_by_key,
                floor,
            ),
            "fastest_feasible_oracle": _oracle_summary(
                "latency",
                requests,
                outcomes_by_key,
                cells_by_key,
                floor,
            ),
            "energy_aware_oracle": _oracle_summary(
                "energy",
                requests,
                outcomes_by_key,
                cells_by_key,
                floor,
            ),
        }
    retrieval = runtime.retrieval_metrics(
        outcomes,
        frozenset(request.request_id for request in requests),
    )
    return {
        "schema_version": 1,
        "experiment": "candidate_workflow_full_information",
        "split": split.value,
        "inference_role": (
            "confirmatory_untouched_test"
            if split is RoutingSplit.TEST and complete_split
            else (
                "exploratory_partial_test"
                if split is RoutingSplit.TEST
                else "exploratory_development"
            )
        ),
        "complete_split": complete_split,
        "prepared_prompt_catalog_fingerprint": prepared_prompt_catalog_fingerprint,
        "selected_request_set_sha256": hashlib.sha256(
            canonical_json(
                [
                    {
                        "request_id": request.request_id,
                        "request_fingerprint": request.request_fingerprint,
                    }
                    for request in requests
                ]
            ).encode("utf-8")
        ).hexdigest(),
        "request_count": len(requests),
        "action_count": len(registry.actions),
        "cell_count": len(outcomes),
        "full_information_complete": len(outcomes) == len(requests) * len(registry.actions),
        "action_order": sorted(registry.actions),
        "actions": action_summaries,
        "retrieval": retrieval.model_dump(mode="json") if retrieval is not None else None,
        "quality_floor_analysis": floor_summaries,
        "baselines": _baseline_summaries(
            requests,
            outcomes_by_key,
            cells_by_key,
            config,
        ),
        "claims": {
            "learned_router": False,
            "router_training_labels_emitted": False,
            "energy_optimization_claimed": False,
        },
    }


def render_workflow_matrix_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Candidate Workflow Matrix",
        "",
        f"Split: **{summary['split']}** ({summary['inference_role']})",
        "",
        "| Action | Complete | Quality | Safe feasible | Mean latency (ms) "
        "| Provider fee (USD) | Energy known |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for action_id in summary["action_order"]:
        action = summary["actions"][action_id]
        quality = action["micro_quality"]
        latency = action["mean_wall_latency_ms"]
        provider_fee = action["provider_fee_usd"]
        provider_fee_cell = f"{provider_fee:.6f}" if provider_fee is not None else "-"
        lines.append(
            f"| {action_id} | {action['completion_coverage']:.3f} | "
            f"{quality if quality is not None else 0:.3f} | "
            f"{action['safety_feasible_completed_rate']:.3f} | "
            f"{latency if latency is not None else 0:.3f} | "
            f"{provider_fee_cell} | "
            f"{action['energy_known_rate']:.3f} |"
        )
    lines.extend(["", "## Quality Floors", ""])
    for floor, analysis in summary["quality_floor_analysis"].items():
        best = analysis["best_quality_oracle"]
        cheapest = analysis["cheapest_feasible_oracle"]
        cheapest_fee = cheapest["provider_fee_usd"]
        lines.extend(
            [
                f"### Floor {floor}",
                "",
                f"- Pareto actions: {', '.join(analysis['pareto_frontier']) or 'none'}",
                f"- Best-quality oracle: {best['mean_quality']:.3f} "
                f"({best['feasible_request_rate']:.3f} feasible)",
                (
                    f"- Cheapest feasible provider fee: ${cheapest_fee:.6f}"
                    if cheapest_fee is not None
                    else "- Cheapest feasible provider fee: unknown"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Baselines",
            "",
            "| Baseline | Quality | Completion | Safety feasible | Oracle gap |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, baseline in summary["baselines"].items():
        lines.append(
            f"| {name} | {baseline['mean_quality']:.3f} | "
            f"{baseline['completion_rate']:.3f} | "
            f"{baseline['safety_feasible_completed_rate']:.3f} | "
            f"{baseline['oracle_gap']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "> This experiment measures candidate workflows and oracle opportunity. "
            "It does not train a router or emit per-request routing labels.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_workflow_matrix_unlocked(
    verified: VerifiedRoutingSplit,
    output_dir: Path,
    config: RoutingConfig,
    registry: ActionRegistry,
    runtime: MatrixRuntime,
    sandbox: CodeSandbox,
    options: WorkflowMatrixOptions,
) -> WorkflowMatrixResult:
    requests, evaluators = _selected_split(verified, options.limit)
    all_outcomes: list[ActionOutcome] = []
    action_paths: dict[str, Path] = {}
    for action_id in sorted(registry.actions):
        outcomes, path = _run_action(
            output_dir,
            registry.actions[action_id],
            requests,
            verified.requests_sha256,
            runtime,
            options,
        )
        all_outcomes.extend(outcomes)
        action_paths[action_id] = path
    all_outcomes.sort(key=lambda outcome: (outcome.request_id, outcome.action_id))
    cells = _score_outcomes(all_outcomes, evaluators, sandbox)
    summary = _build_summary(
        config,
        registry,
        runtime,
        options.split,
        requests,
        all_outcomes,
        cells,
        complete_split=len(requests) == len(verified.requests),
        prepared_prompt_catalog_fingerprint=verified.prompt_catalog_fingerprint,
    )
    outcomes_path = output_dir / "action_outcomes.jsonl"
    summary_path = output_dir / "workflow_matrix_summary.json"
    report_path = output_dir / "workflow_matrix_report.md"
    _write_outcomes(outcomes_path, all_outcomes)
    _write_atomic(
        summary_path,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _write_atomic(
        report_path,
        render_workflow_matrix_report(summary).encode("utf-8"),
    )
    return WorkflowMatrixResult(
        outcomes_path=outcomes_path,
        summary_path=summary_path,
        report_path=report_path,
        summary=summary,
        action_outcome_paths=action_paths,
    )


def run_workflow_matrix(
    requests_path: Path,
    evaluator_path: Path,
    data_manifest_path: Path,
    output_dir: Path,
    config: RoutingConfig,
    registry: ActionRegistry,
    runtime: MatrixRuntime,
    sandbox: CodeSandbox,
    options: WorkflowMatrixOptions | None = None,
) -> WorkflowMatrixResult:
    resolved = options or WorkflowMatrixOptions()
    lock_target = output_dir / "action_outcomes.jsonl"
    with acquire_run_lock(lock_target):
        _check_aggregate_collision(output_dir, resolved)
        _invalidate_aggregate_artifacts(output_dir)
        verified = load_verified_routing_split(
            requests_path,
            evaluator_path,
            data_manifest_path,
            resolved.split,
        )
        return _run_workflow_matrix_unlocked(
            verified,
            output_dir,
            config,
            registry,
            runtime,
            sandbox,
            resolved,
        )


def run_workflow_matrix_with_runtime_factory(
    requests_path: Path,
    evaluator_path: Path,
    data_manifest_path: Path,
    output_dir: Path,
    config: RoutingConfig,
    registry: ActionRegistry,
    runtime_factory: RuntimeFactory,
    sandbox: CodeSandbox,
    options: WorkflowMatrixOptions | None = None,
    plan_callback: PlanCallback | None = None,
) -> WorkflowMatrixResult:
    """Invalidate, plan, and execute one CLI matrix attempt under one lock."""

    resolved = options or WorkflowMatrixOptions()
    lock_target = output_dir / "action_outcomes.jsonl"
    with acquire_run_lock(lock_target):
        _check_aggregate_collision(output_dir, resolved)
        _invalidate_aggregate_artifacts(output_dir)
        verified = load_verified_routing_split(
            requests_path,
            evaluator_path,
            data_manifest_path,
            resolved.split,
        )
        runtime = runtime_factory(verified)
        plan = _workflow_matrix_plan_unlocked(
            verified,
            output_dir,
            registry,
            runtime,
            resolved,
        )
        if plan_callback is not None:
            plan_callback(plan)
        return _run_workflow_matrix_unlocked(
            verified,
            output_dir,
            config,
            registry,
            runtime,
            sandbox,
            resolved,
        )
