from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from small_models_society.cli import main
from small_models_society.data.prepare import (
    canonical_json,
    load_benchmark,
    sha256_bytes,
)
from small_models_society.experiments.workflow_matrix import (
    CandidateWorkflowRuntime,
    RetrievalMatrixAnalysis,
    ScoredActionCell,
    WorkflowMatrixOptions,
    _candidate_cells,
    _pareto_frontier,
    bootstrap_mean_interval,
    inspect_workflow_matrix,
    run_workflow_matrix,
)
from small_models_society.inference.contracts import ChatMessage
from small_models_society.inference.hardware import HardwareReport
from small_models_society.inference.prompts import load_prompt_catalog
from small_models_society.inference.runner import ResumeMismatchError, acquire_run_lock
from small_models_society.retrieval.contracts import (
    RetrievalRelevanceRecord,
    create_retrieval_document,
)
from small_models_society.routing.artifacts import write_workflow_requests
from small_models_society.routing.config import RoutingConfig, load_routing_config
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
    WorkflowRequest,
    create_workflow_request,
)
from small_models_society.routing.data import (
    RoutingEvaluatorRecord,
    RoutingSplit,
)
from small_models_society.routing.registry import (
    RegisteredAction,
    build_action_registry,
)
from small_models_society.sandbox import SandboxResult, SandboxStatus
from small_models_society.schemas import BenchmarkExample, Domain
from small_models_society.training.prepare import normalized_content_sha256

ROOT = Path(__file__).parents[1]
ROUTING_CONFIG = ROOT / "configs" / "routing.yaml"
PROMPT_CONFIG = ROOT / "configs" / "prompt_profiles.yaml"
FIXTURE_BENCHMARK = Path(__file__).parent / "fixtures" / "benchmark.jsonl"


def _config() -> RoutingConfig:
    config = load_routing_config(ROUTING_CONFIG)
    analysis = config.analysis.model_copy(update={"bootstrap_resamples": 50})
    return config.model_copy(update={"analysis": analysis})


def _output_contract(domain: Domain) -> OutputContract:
    return {
        Domain.MATH: OutputContract.NUMERIC,
        Domain.CODE: OutputContract.PYTHON_SOURCE,
        Domain.LOGIC: OutputContract.CHOICE_LABEL,
        Domain.KNOWLEDGE: OutputContract.SHORT_ANSWER,
    }[domain]


def _request(example: BenchmarkExample) -> WorkflowRequest:
    defaults = _config().policy_defaults
    attributes = (
        {"retrieval_query": example.input.question} if example.domain is Domain.KNOWLEDGE else {}
    )
    user_content = "4 + 6" if example.domain is Domain.MATH else example.input.model_dump_json()
    return create_workflow_request(
        request_id=example.id,
        messages=(
            ChatMessage(role="system", content="Return only the answer."),
            ChatMessage(role="user", content=user_content),
        ),
        output_contract=_output_contract(example.domain),
        policy=RequestPolicyContext(
            data_classification=defaults.data_classification,
            network_allowed=defaults.network_allowed,
            allowed_corpus_ids=defaults.allowed_corpus_ids,
            allowed_tool_ids=defaults.allowed_tool_ids,
            required_quality=defaults.required_quality,
            allow_unknown_output_safety=True,
        ),
        attributes=attributes,
    )


def _routing_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    examples = load_benchmark(FIXTURE_BENCHMARK)
    requests = [_request(example) for example in examples]
    requests_path = tmp_path / "development.requests.jsonl"
    request_artifact = write_workflow_requests(requests_path, requests)
    evaluators = [
        RoutingEvaluatorRecord(
            request_id=example.id,
            split=RoutingSplit.DEVELOPMENT,
            source_id=f"fixture::{example.id}",
            source_content_sha256=normalized_content_sha256(example),
            example=example,
        )
        for example in examples
    ]
    evaluator_path = tmp_path / "development.evaluator.jsonl"
    evaluator_content = (
        "\n".join(canonical_json(record.model_dump(mode="json")) for record in evaluators) + "\n"
    ).encode("utf-8")
    evaluator_path.write_bytes(evaluator_content)
    test_requests_path = tmp_path / "test.requests.jsonl"
    test_request_artifact = write_workflow_requests(test_requests_path, requests)
    test_evaluator_path = tmp_path / "test.evaluator.jsonl"
    test_evaluator_content = (
        "\n".join(
            canonical_json(
                record.model_copy(update={"split": RoutingSplit.TEST}).model_dump(mode="json")
            )
            for record in evaluators
        )
        + "\n"
    ).encode("utf-8")
    test_evaluator_path.write_bytes(test_evaluator_content)
    manifest = {
        "schema_version": 1,
        "routing_config_fingerprint": _config().fingerprint(),
        "prompt_catalog_fingerprint": load_prompt_catalog(PROMPT_CONFIG).fingerprint(),
        "files": {
            "development_requests": {
                "path": requests_path.name,
                "sha256": request_artifact.sha256,
                "row_count": len(requests),
            },
            "development_evaluator": {
                "path": evaluator_path.name,
                "sha256": sha256_bytes(evaluator_content),
                "row_count": len(evaluators),
            },
            "test_requests": {
                "path": test_requests_path.name,
                "sha256": test_request_artifact.sha256,
                "row_count": len(requests),
            },
            "test_evaluator": {
                "path": test_evaluator_path.name,
                "sha256": sha256_bytes(test_evaluator_content),
                "row_count": len(evaluators),
            },
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return requests_path, evaluator_path, manifest_path


CORRECT = {
    Domain.MATH: "10",
    Domain.CODE: "def add(a, b):\n    return a + b",
    Domain.LOGIC: "A",
    Domain.KNOWLEDGE: "Paris",
}
WRONG = {
    Domain.MATH: "11",
    Domain.CODE: "def add(a, b):\n    return 0",
    Domain.LOGIC: "B",
    Domain.KNOWLEDGE: "London",
}


class FakeRuntime:
    def __init__(
        self,
        *,
        unavailable_actions: set[str] | None = None,
        runtime_version: str = "v1",
        interrupt_after: int | None = None,
    ) -> None:
        self.unavailable_actions = unavailable_actions or set()
        self.runtime_version = runtime_version
        self.interrupt_after = interrupt_after
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def _domain(request: WorkflowRequest) -> Domain:
        return {
            "fixture-math-1": Domain.MATH,
            "fixture-code-1": Domain.CODE,
            "fixture-logic-1": Domain.LOGIC,
            "fixture-knowledge-1": Domain.KNOWLEDGE,
        }[request.request_id]

    def availability(
        self,
        request: WorkflowRequest,
        registered: RegisteredAction,
    ) -> ActionAvailability:
        action = registered.action
        domain = self._domain(request)
        unavailable = action.action_id in self.unavailable_actions
        reason = "synthetic_runtime_missing"
        if action.kind.value == "tool" and domain is not Domain.MATH:
            unavailable = True
            reason = "calculator_input_unsupported"
        if action.kind.value == "retrieval" and domain is not Domain.KNOWLEDGE:
            unavailable = True
            reason = "retrieval_query_missing"
        if action.kind.value == "strong_replay" and domain is not Domain.LOGIC:
            unavailable = True
            reason = "replay_row_missing"
        return ActionAvailability(
            action_id=action.action_id,
            action_fingerprint=action.action_fingerprint,
            status=(
                AvailabilityStatus.UNAVAILABLE if unavailable else AvailabilityStatus.AVAILABLE
            ),
            reason_code=reason if unavailable else None,
            rule_ids=("runtime.synthetic.v1",) if unavailable else (),
        )

    def execute(
        self,
        request: WorkflowRequest,
        registered: RegisteredAction,
        availability: ActionAvailability,
    ) -> ActionOutcome:
        if self.interrupt_after is not None and len(self.calls) >= self.interrupt_after:
            raise KeyboardInterrupt
        action = registered.action
        domain = self._domain(request)
        self.calls.append((request.request_id, action.action_id))
        own_specialist = action.adapter_id == domain.value
        correct = (
            own_specialist
            or (action.kind.value == "tool" and domain is Domain.MATH)
            or (action.kind.value == "retrieval" and domain is Domain.KNOWLEDGE)
            or (action.kind.value == "strong_replay" and domain is Domain.LOGIC)
        )
        provider_fee = 0.01 if action.kind.value == "strong_replay" else 0.0
        known_energy = action.kind.value == "strong_replay"
        return ActionOutcome(
            request_id=request.request_id,
            request_fingerprint=request.request_fingerprint,
            action_id=action.action_id,
            action_fingerprint=action.action_fingerprint,
            status=ActionOutcomeStatus.COMPLETED,
            availability=availability,
            response=(CORRECT if correct else WRONG)[domain],
            safety=SafetyAssessment(
                status=SafetyStatus.SAFE,
                source="synthetic-safe-v1",
            ),
            telemetry=ActionTelemetry(
                wall_latency_ms=float(len(action.action_id)),
                prompt_tokens=10,
                completion_tokens=2,
                provider_fee_usd=provider_fee,
                compute_cost_usd=None,
                total_cost_usd=None,
                energy_joules=5.0 if known_energy else None,
                energy_provenance=(
                    EnergyProvenance.REPLAY if known_energy else EnergyProvenance.UNAVAILABLE
                ),
                energy_measurement_source=("synthetic-meter-v1" if known_energy else None),
                device="synthetic",
            ),
            metadata={"runtime_version": self.runtime_version},
        )

    def outcome_is_current(
        self,
        outcome: ActionOutcome,
        request: WorkflowRequest,
        registered: RegisteredAction,
        availability: ActionAvailability,
    ) -> bool:
        del request, registered
        if outcome.status in {
            ActionOutcomeStatus.BLOCKED,
            ActionOutcomeStatus.UNAVAILABLE,
        }:
            return outcome.availability == availability
        return (
            availability.status is AvailabilityStatus.AVAILABLE
            and outcome.metadata.get("runtime_version") == self.runtime_version
        )

    def retrieval_metrics(
        self,
        outcomes: Sequence[ActionOutcome],
        selected_request_ids: frozenset[str],
    ) -> RetrievalMatrixAnalysis | None:
        del outcomes, selected_request_ids
        return None


class FakeSandbox:
    def run(self, candidate: str, test_setup: str, tests: list[str]) -> SandboxResult:
        del test_setup, tests
        return SandboxResult(
            status=(
                SandboxStatus.PASSED
                if "return a + b" in candidate
                else SandboxStatus.ASSERTION_FAILURE
            ),
            duration_ms=1,
        )


class InfrastructureErrorSandbox:
    def run(self, candidate: str, test_setup: str, tests: list[str]) -> SandboxResult:
        del candidate, test_setup, tests
        return SandboxResult(
            status=SandboxStatus.INFRASTRUCTURE_ERROR,
            duration_ms=1,
            stderr="sandbox unavailable",
        )


def _hardware() -> HardwareReport:
    return HardwareReport(
        ready=True,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        python_version="3.11.9",
        package_versions={"torch": "test", "transformers": "test"},
        selected_device="cpu",
        selected_dtype="float32",
        cuda_available=False,
        system_ram_gb=16,
        model_cache_path="C:/cache/model",
        model_cached=True,
    )


def _run(
    tmp_path: Path,
    runtime: FakeRuntime,
    *,
    output_dir: Path | None = None,
    options: WorkflowMatrixOptions | None = None,
    sandbox: FakeSandbox | InfrastructureErrorSandbox | None = None,
):  # type: ignore[no-untyped-def]
    data_dir = tmp_path / "data"
    requests, evaluator, manifest = _routing_artifacts(data_dir)
    resolved_options = options or WorkflowMatrixOptions()
    if resolved_options.split is RoutingSplit.TEST:
        requests = data_dir / "test.requests.jsonl"
        evaluator = data_dir / "test.evaluator.jsonl"
    config = _config()
    return run_workflow_matrix(
        requests,
        evaluator,
        manifest,
        output_dir or tmp_path / "matrix",
        config,
        build_action_registry(config),
        runtime,
        sandbox or FakeSandbox(),
        resolved_options,
    )


def test_builds_complete_reference_free_action_matrix_and_analysis(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    result = _run(tmp_path, runtime)

    assert result.summary["request_count"] == 4
    assert result.summary["action_count"] == 8
    assert result.summary["cell_count"] == 32
    assert result.summary["full_information_complete"] is True
    assert result.summary["claims"] == {
        "learned_router": False,
        "router_training_labels_emitted": False,
        "energy_optimization_claimed": False,
    }
    assert result.summary["prepared_prompt_catalog_fingerprint"] == (
        load_prompt_catalog(PROMPT_CONFIG).fingerprint()
    )
    assert len(result.outcomes_path.read_text(encoding="utf-8").splitlines()) == 32
    outcome_text = result.outcomes_path.read_text(encoding="utf-8")
    assert '"domain"' not in outcome_text
    assert '"reference"' not in outcome_text
    assert '"supporting_facts"' not in outcome_text
    assert len(result.action_outcome_paths) == 8
    assert all(path.is_file() for path in result.action_outcome_paths.values())

    actions = result.summary["actions"]
    assert actions["local.qwen-base.v1"]["micro_quality"] == 0
    assert actions["local.qwen-lora-math.v1"]["micro_quality"] == 0.25
    assert actions["tool.calculator.v1"]["completion_coverage"] == 0.25
    assert actions["remote.strong-replay.reference.v1"]["energy_known_rate"] == 1
    floor_zero = result.summary["quality_floor_analysis"]["0.0"]
    assert floor_zero["best_quality_oracle"]["mean_quality"] == 1
    assert floor_zero["best_quality_oracle"]["feasible_request_rate"] == 1
    assert floor_zero["energy_aware_oracle"]["feasible_request_count"] == 1
    assert result.summary["baselines"]["always_local_base"]["mean_quality"] == 0
    assert result.summary["baselines"]["always_local_base"]["oracle_gap_ci"]["estimate"] == 1
    assert result.summary["baselines"]["static_tool_rag_local"]["mean_quality"] == 0.5
    assert "does not train a router" in result.report_path.read_text(encoding="utf-8")


def test_completed_resume_executes_no_actions_and_is_byte_identical(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    first_runtime = FakeRuntime()
    first = _run(tmp_path, first_runtime, output_dir=output_dir)
    first_summary = first.summary_path.read_bytes()
    first_outcomes = first.outcomes_path.read_bytes()
    second_runtime = FakeRuntime()

    second = _run(
        tmp_path,
        second_runtime,
        output_dir=output_dir,
        options=WorkflowMatrixOptions(resume=True),
    )

    assert first_runtime.calls
    assert second_runtime.calls == []
    assert second.summary_path.read_bytes() == first_summary
    assert second.outcomes_path.read_bytes() == first_outcomes


def test_newly_available_lora_replaces_only_its_unavailable_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    action_id = "local.qwen-lora-math.v1"
    first = _run(
        tmp_path,
        FakeRuntime(unavailable_actions={action_id}),
        output_dir=output_dir,
    )
    before = {name: path.read_bytes() for name, path in first.action_outcome_paths.items()}
    resumed_runtime = FakeRuntime()

    second = _run(
        tmp_path,
        resumed_runtime,
        output_dir=output_dir,
        options=WorkflowMatrixOptions(resume=True),
    )

    assert resumed_runtime.calls == [
        (request_id, action_id)
        for request_id in (
            "fixture-math-1",
            "fixture-code-1",
            "fixture-logic-1",
            "fixture-knowledge-1",
        )
    ]
    assert second.action_outcome_paths[action_id].read_bytes() != before[action_id]
    for other_id, path in second.action_outcome_paths.items():
        if other_id != action_id:
            assert path.read_bytes() == before[other_id]


def test_interrupted_action_checkpoints_and_resume_runs_only_pending_cells(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "matrix"
    with pytest.raises(KeyboardInterrupt):
        _run(
            tmp_path,
            FakeRuntime(interrupt_after=3),
            output_dir=output_dir,
            options=WorkflowMatrixOptions(checkpoint_interval=1),
        )
    requests, evaluator, manifest = _routing_artifacts(tmp_path / "data")
    config = _config()
    plan = inspect_workflow_matrix(
        requests,
        evaluator,
        manifest,
        output_dir,
        build_action_registry(config),
        FakeRuntime(),
        WorkflowMatrixOptions(resume=True, checkpoint_interval=1),
    )
    assert plan.completed_cell_count == 3
    assert plan.pending_cell_count == 29
    resumed_runtime = FakeRuntime()

    result = run_workflow_matrix(
        requests,
        evaluator,
        manifest,
        output_dir,
        config,
        build_action_registry(config),
        resumed_runtime,
        FakeSandbox(),
        WorkflowMatrixOptions(resume=True, checkpoint_interval=1),
    )

    assert len(resumed_runtime.calls) == 20
    assert result.summary["full_information_complete"] is True


def test_rejects_stale_action_manifest_on_resume(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    result = _run(tmp_path, FakeRuntime(), output_dir=output_dir)
    action_path = result.action_outcome_paths["local.qwen-base.v1"]
    manifest_path = action_path.parent / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["action_fingerprint"] = "a" * 64
    manifest_path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(ResumeMismatchError, match="invalid action run manifest"):
        _run(
            tmp_path,
            FakeRuntime(),
            output_dir=output_dir,
            options=WorkflowMatrixOptions(resume=True),
        )


def test_requires_explicit_collision_policy_and_rejects_concurrent_writer(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "matrix"
    _run(tmp_path, FakeRuntime(), output_dir=output_dir)
    with pytest.raises(FileExistsError, match="resume or overwrite"):
        _run(tmp_path, FakeRuntime(), output_dir=output_dir)

    other_output = tmp_path / "locked"
    with (
        acquire_run_lock(other_output / "action_outcomes.jsonl"),
        pytest.raises(
            FileExistsError,
            match="another process",
        ),
    ):
        _run(tmp_path, FakeRuntime(), output_dir=other_output)


def test_sandbox_infrastructure_error_aborts_aggregates_but_keeps_action_rows(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "matrix"
    with pytest.raises(RuntimeError, match="sandbox infrastructure error"):
        _run(
            tmp_path,
            FakeRuntime(),
            output_dir=output_dir,
            sandbox=InfrastructureErrorSandbox(),
        )

    assert not (output_dir / "workflow_matrix_summary.json").exists()
    assert len(list((output_dir / "actions").glob("*/outcomes.jsonl"))) == 8
    resumed_runtime = FakeRuntime()
    result = _run(
        tmp_path,
        resumed_runtime,
        output_dir=output_dir,
        options=WorkflowMatrixOptions(resume=True),
    )
    assert resumed_runtime.calls == []
    assert result.summary_path.is_file()


def test_bootstrap_intervals_are_deterministic_and_label_scoped() -> None:
    first = bootstrap_mean_interval(
        [0, 1, 1, 0],
        seed=42,
        resamples=100,
        confidence_level=0.95,
        label="first",
    )
    second = bootstrap_mean_interval(
        [0, 1, 1, 0],
        seed=42,
        resamples=100,
        confidence_level=0.95,
        label="first",
    )
    other = bootstrap_mean_interval(
        [0, 1, 1, 0],
        seed=42,
        resamples=100,
        confidence_level=0.95,
        label="other",
    )

    assert first == second
    assert first.estimate == 0.5
    assert first.lower <= first.estimate <= first.upper
    assert other.resamples == first.resamples


def test_request_quality_requirement_overrides_lower_analysis_floor() -> None:
    request = _request(load_benchmark(FIXTURE_BENCHMARK)[0])
    action = build_action_registry(_config()).actions["local.qwen-base.v1"].action
    availability = ActionAvailability(
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=AvailabilityStatus.AVAILABLE,
    )
    outcome = ActionOutcome(
        request_id=request.request_id,
        request_fingerprint=request.request_fingerprint,
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=ActionOutcomeStatus.COMPLETED,
        availability=availability,
        response="partial",
        safety=SafetyAssessment(status=SafetyStatus.SAFE, source="test"),
        telemetry=ActionTelemetry(wall_latency_ms=1, provider_fee_usd=0),
    )
    cell = ScoredActionCell(
        request_id=request.request_id,
        action_id=action.action_id,
        domain=Domain.MATH,
        outcome_status=ActionOutcomeStatus.COMPLETED,
        quality_metric="exact_match",
        quality_score=0.5,
        safety_status=SafetyStatus.SAFE,
        wall_latency_ms=1,
        provider_fee_usd=0,
        energy_provenance=EnergyProvenance.UNAVAILABLE,
    )

    assert request.policy.required_quality == 0.8
    assert (
        _candidate_cells(
            request,
            {(request.request_id, action.action_id): outcome},
            {(request.request_id, action.action_id): cell},
            0.0,
        )
        == []
    )


def test_pareto_excludes_sparse_and_incompletely_priced_actions() -> None:
    full = {
        "micro_quality": 0.9,
        "minimum_quality": 0.9,
        "mean_provider_fee_usd": 0.01,
        "mean_wall_latency_ms": 10.0,
        "completed_count": 2,
        "request_count": 2,
        "safety_feasible_completed_count": 2,
        "hard_constraint_feasible_count": 2,
        "provider_fee_known_rate": 1.0,
        "mean_energy_joules": None,
        "energy_known_rate": 0.0,
        "energy_measurement_boundaries": [],
    }
    sparse = {**full, "completed_count": 1, "safety_feasible_completed_count": 1}
    unknown_fee = {**full, "provider_fee_known_rate": 0.5}

    assert _pareto_frontier(
        {"full": full, "sparse": sparse, "unknown": unknown_fee},
        0.8,
        include_energy=False,
    ) == ["full"]


def test_energy_pareto_does_not_compare_different_measurement_boundaries() -> None:
    base = {
        "micro_quality": 1.0,
        "minimum_quality": 1.0,
        "mean_provider_fee_usd": 0.0,
        "mean_wall_latency_ms": 1.0,
        "completed_count": 1,
        "request_count": 1,
        "safety_feasible_completed_count": 1,
        "hard_constraint_feasible_count": 1,
        "provider_fee_known_rate": 1.0,
        "energy_known_rate": 1.0,
    }
    meter_a = {
        **base,
        "mean_energy_joules": 5.0,
        "energy_measurement_boundaries": [{"provenance": "measured", "source": "meter-a"}],
    }
    meter_b = {
        **base,
        "mean_energy_joules": 6.0,
        "energy_measurement_boundaries": [{"provenance": "measured", "source": "meter-b"}],
    }

    assert _pareto_frontier(
        {"meter-a": meter_a, "meter-b": meter_b},
        0.8,
        include_energy=True,
    ) == ["meter-a", "meter-b"]


class UnsafeCorrectBaseRuntime(FakeRuntime):
    def execute(
        self,
        request: WorkflowRequest,
        registered: RegisteredAction,
        availability: ActionAvailability,
    ) -> ActionOutcome:
        outcome = super().execute(request, registered, availability)
        if registered.action.action_id != "local.qwen-base.v1":
            return outcome
        value = outcome.model_dump(mode="json")
        value["response"] = CORRECT[self._domain(request)]
        value["safety"] = SafetyAssessment(
            status=SafetyStatus.BLOCKED,
            source="synthetic-policy",
            rule_ids=("synthetic.block-v1",),
        ).model_dump(mode="json")
        return ActionOutcome.model_validate(value)


class ErrorOutcomeRuntime(FakeRuntime):
    def execute(
        self,
        request: WorkflowRequest,
        registered: RegisteredAction,
        availability: ActionAvailability,
    ) -> ActionOutcome:
        if registered.action.action_id != "local.qwen-base.v1":
            return super().execute(request, registered, availability)
        self.calls.append((request.request_id, registered.action.action_id))
        action = registered.action
        return ActionOutcome(
            request_id=request.request_id,
            request_fingerprint=request.request_fingerprint,
            action_id=action.action_id,
            action_fingerprint=action.action_fingerprint,
            status=ActionOutcomeStatus.ERROR,
            availability=availability,
            safety=SafetyAssessment(
                status=SafetyStatus.NOT_ASSESSED,
                source="synthetic-runtime",
            ),
            telemetry=ActionTelemetry(wall_latency_ms=1, provider_fee_usd=0),
            error_type="SyntheticInfrastructureError",
            error_message="runtime unavailable",
            metadata={"runtime_version": self.runtime_version},
        )


def test_unsafe_baseline_quality_is_not_credited_against_safe_oracle(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, UnsafeCorrectBaseRuntime())
    baseline = result.summary["baselines"]["always_local_base"]

    assert baseline["mean_quality"] == 1
    assert baseline["constrained_mean_quality"] == 0
    assert baseline["oracle_gap"] == 1
    assert baseline["oracle_gap_ci"]["estimate"] == 1


def test_executor_error_outcome_aborts_aggregates(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    with pytest.raises(RuntimeError, match="action infrastructure error"):
        _run(tmp_path, ErrorOutcomeRuntime(), output_dir=output_dir)

    assert not (output_dir / "workflow_matrix_summary.json").exists()
    assert (output_dir / "actions" / "local.qwen-base.v1" / "outcomes.jsonl").exists()


def test_failed_resume_removes_stale_aggregate_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    first = _run(tmp_path, FakeRuntime(), output_dir=output_dir)
    assert first.summary_path.exists()

    with pytest.raises(KeyboardInterrupt):
        _run(
            tmp_path,
            FakeRuntime(runtime_version="v2", interrupt_after=0),
            output_dir=output_dir,
            options=WorkflowMatrixOptions(resume=True),
        )

    assert not first.outcomes_path.exists()
    assert not first.summary_path.exists()
    assert not first.report_path.exists()


def test_partial_test_split_is_not_labeled_confirmatory(tmp_path: Path) -> None:
    partial = _run(
        tmp_path,
        FakeRuntime(),
        options=WorkflowMatrixOptions(split=RoutingSplit.TEST, limit=1),
    )
    full = _run(
        tmp_path / "full",
        FakeRuntime(),
        options=WorkflowMatrixOptions(split=RoutingSplit.TEST),
    )

    assert partial.summary["inference_role"] == "exploratory_partial_test"
    assert partial.summary["complete_split"] is False
    assert full.summary["inference_role"] == "confirmatory_untouched_test"
    assert full.summary["complete_split"] is True


def test_unverifiable_historical_local_outcome_is_rejected() -> None:
    config = _config()
    registry = build_action_registry(config)
    runtime = CandidateWorkflowRuntime(config, registry)
    request = _request(load_benchmark(FIXTURE_BENCHMARK)[0])
    registered = registry.actions["local.qwen-base.v1"]
    current_availability = runtime.availability(request, registered)
    historical_availability = ActionAvailability(
        action_id=registered.action.action_id,
        action_fingerprint=registered.action.action_fingerprint,
        status=AvailabilityStatus.AVAILABLE,
    )
    outcome = ActionOutcome(
        request_id=request.request_id,
        request_fingerprint=request.request_fingerprint,
        action_id=registered.action.action_id,
        action_fingerprint=registered.action.action_fingerprint,
        status=ActionOutcomeStatus.COMPLETED,
        availability=historical_availability,
        response="10",
        safety=SafetyAssessment(status=SafetyStatus.SAFE, source="historical"),
        telemetry=ActionTelemetry(wall_latency_ms=1, provider_fee_usd=0),
        metadata={"execution_fingerprint": "a" * 64},
    )

    with pytest.raises(ResumeMismatchError, match="cannot verify historical local"):
        runtime.outcome_is_current(
            outcome,
            request,
            registered,
            current_availability,
        )


def test_candidate_runtime_applies_availability_before_calculator_dispatch() -> None:
    config = _config()
    registry = build_action_registry(config)
    runtime = CandidateWorkflowRuntime(config, registry)
    request = _request(load_benchmark(FIXTURE_BENCHMARK)[0])
    calculator = registry.actions["tool.calculator.v1"]
    local = registry.actions["local.qwen-base.v1"]

    calculator_availability = runtime.availability(request, calculator)
    local_availability = runtime.availability(request, local)
    outcome = runtime.execute(request, calculator, calculator_availability)

    assert calculator_availability.status is AvailabilityStatus.AVAILABLE
    assert local_availability.status is AvailabilityStatus.UNAVAILABLE
    assert local_availability.reason_code == "local_model_not_ready"
    assert outcome.status is ActionOutcomeStatus.COMPLETED
    assert outcome.response == "10"
    assert runtime.outcome_is_current(
        outcome,
        request,
        calculator,
        calculator_availability,
    )


def test_candidate_runtime_reconstructs_retrieval_metrics_from_outcome_metadata() -> None:
    config = _config()
    registry = build_action_registry(config)
    document = create_retrieval_document(
        "France",
        "Paris is the capital of France.",
    )
    relevance = RetrievalRelevanceRecord(
        request_id="fixture-knowledge-1",
        split="development",
        relevant_document_ids=(document.document_id,),
    )
    runtime = CandidateWorkflowRuntime(
        config,
        registry,
        retrieval_relevance=[relevance],
    )
    request = _request(load_benchmark(FIXTURE_BENCHMARK)[3])
    action = registry.actions["rag.bm25-qwen-base.v1"].action
    availability = ActionAvailability(
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=AvailabilityStatus.AVAILABLE,
    )
    outcome = ActionOutcome(
        request_id=request.request_id,
        request_fingerprint=request.request_fingerprint,
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=ActionOutcomeStatus.COMPLETED,
        availability=availability,
        response="Paris",
        safety=SafetyAssessment(status=SafetyStatus.UNKNOWN, source="not_assessed"),
        telemetry=ActionTelemetry(
            wall_latency_ms=10,
            provider_fee_usd=0,
            compute_cost_usd=None,
            total_cost_usd=None,
        ),
        metadata={
            "retrieval_status": "completed",
            "retrieval_top_k": 10,
            "retrieval_latency_ms": 2.5,
            "hit_document_ids": [document.document_id],
        },
    )

    analysis = runtime.retrieval_metrics([outcome], frozenset({request.request_id}))

    assert analysis is not None
    assert analysis.observation_coverage == 1
    metrics = analysis.metrics
    assert metrics is not None
    assert metrics.recall_at_k == {1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0}
    assert metrics.mean_reciprocal_rank == 1
    assert metrics.corpus_coverage == 1

    malformed = outcome.model_copy(update={"metadata": {}})
    with pytest.raises(RuntimeError, match="missing persisted retrieval metadata"):
        runtime.retrieval_metrics(
            [malformed],
            frozenset({request.request_id}),
        )

    blocked_relevance = RetrievalRelevanceRecord(
        request_id="blocked-knowledge",
        split="development",
        relevant_document_ids=(document.document_id,),
    )
    mixed_runtime = CandidateWorkflowRuntime(
        config,
        registry,
        retrieval_relevance=[relevance, blocked_relevance],
    )
    blocked_availability = ActionAvailability(
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=AvailabilityStatus.BLOCKED,
        reason_code="corpus_not_authorized",
        rule_ids=("policy.corpus-authorization.v1",),
    )
    blocked_outcome = ActionOutcome(
        request_id="blocked-knowledge",
        request_fingerprint="b" * 64,
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=ActionOutcomeStatus.BLOCKED,
        availability=blocked_availability,
        safety=SafetyAssessment(
            status=SafetyStatus.BLOCKED,
            source="policy",
            rule_ids=("policy.corpus-authorization.v1",),
        ),
    )
    mixed = mixed_runtime.retrieval_metrics(
        [outcome, blocked_outcome],
        frozenset({request.request_id, "blocked-knowledge"}),
    )
    assert mixed is not None
    assert mixed.candidate_request_count == 2
    assert mixed.observed_request_count == 1
    assert mixed.observation_coverage == 0.5
    assert mixed.metrics is not None


def test_routing_inspect_cli_verifies_both_prepared_splits(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    _routing_artifacts(data_dir)

    assert (
        main(
            [
                "routing",
                "inspect",
                "--config",
                str(ROUTING_CONFIG),
                "--data-dir",
                str(data_dir),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["actions"]["remote.strong-replay.reference.v1"]["kind"] == ("strong_replay")
    assert len(output["actions"]["remote.strong-replay.reference.v1"]["action_fingerprint"]) == 64
    assert output["splits"]["development"]["request_count"] == 4
    assert output["splits"]["test"]["request_count"] == 4
    assert output["splits"]["development"]["retrieval"] is None


def test_routing_inspect_cli_rejects_corpus_from_another_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    _routing_artifacts(data_dir)
    corpus_dir = data_dir / "retrieval" / "development"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    swapped = SimpleNamespace(
        manifest=SimpleNamespace(
            split="test",
            source_evaluator_sha256="a" * 64,
        )
    )
    monkeypatch.setattr(
        "small_models_society.cli.load_retrieval_corpus",
        lambda _directory: swapped,
    )

    assert (
        main(
            [
                "routing",
                "inspect",
                "--config",
                str(ROUTING_CONFIG),
                "--data-dir",
                str(data_dir),
            ]
        )
        == 1
    )
    assert "corpus split does not match" in capsys.readouterr().err


def test_workflow_matrix_cli_plans_then_runs_without_real_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    _routing_artifacts(data_dir)
    runtime = FakeRuntime()
    monkeypatch.setattr(
        "small_models_society.cli._workflow_runtime",
        lambda *_args: (runtime, _hardware()),
    )
    monkeypatch.setattr("small_models_society.cli.docker_available", lambda: True)
    monkeypatch.setattr(
        "small_models_society.cli.sandbox_image_available",
        lambda _image: True,
    )
    monkeypatch.setattr(
        "small_models_society.cli.DockerSandbox",
        lambda **_kwargs: FakeSandbox(),
    )

    assert (
        main(
            [
                "experiment",
                "workflow-matrix",
                "--config",
                str(ROUTING_CONFIG),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(tmp_path / "matrix"),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["best_quality_oracle"] == 1
    assert Path(output["outcomes"]).is_file()
    assert Path(output["summary"]).is_file()
    assert "workflow matrix plan:" in captured.err
    assert len(runtime.calls) == 23


def test_cli_resume_runtime_failure_invalidates_old_aggregates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "matrix"
    _routing_artifacts(data_dir)
    monkeypatch.setattr(
        "small_models_society.cli._workflow_runtime",
        lambda *_args: (FakeRuntime(), _hardware()),
    )
    monkeypatch.setattr("small_models_society.cli.docker_available", lambda: True)
    monkeypatch.setattr(
        "small_models_society.cli.sandbox_image_available",
        lambda _image: True,
    )
    monkeypatch.setattr(
        "small_models_society.cli.DockerSandbox",
        lambda **_kwargs: FakeSandbox(),
    )
    arguments = [
        "experiment",
        "workflow-matrix",
        "--config",
        str(ROUTING_CONFIG),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
    ]
    assert main(arguments) == 0
    capsys.readouterr()
    assert (output_dir / "workflow_matrix_summary.json").exists()

    def fail_runtime(*_args: object) -> None:
        raise RuntimeError("synthetic runtime construction failure")

    monkeypatch.setattr("small_models_society.cli._workflow_runtime", fail_runtime)
    assert main([*arguments, "--resume"]) == 1

    assert "synthetic runtime construction failure" in capsys.readouterr().err
    assert not (output_dir / "action_outcomes.jsonl").exists()
    assert not (output_dir / "workflow_matrix_summary.json").exists()
    assert not (output_dir / "workflow_matrix_report.md").exists()


def test_cli_resume_rejects_prompt_catalog_different_from_prepared_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "matrix"
    result = _run(tmp_path, FakeRuntime(), output_dir=output_dir)
    modified_prompts = tmp_path / "modified-prompts.yaml"
    modified_prompts.write_text(
        PROMPT_CONFIG.read_text(encoding="utf-8").replace(
            "You are a careful general-purpose assistant.",
            "You are a changed general-purpose assistant.",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("small_models_society.cli.docker_available", lambda: True)
    monkeypatch.setattr(
        "small_models_society.cli.sandbox_image_available",
        lambda _image: True,
    )

    assert (
        main(
            [
                "experiment",
                "workflow-matrix",
                "--config",
                str(ROUTING_CONFIG),
                "--data-dir",
                str(tmp_path / "data"),
                "--prompts",
                str(modified_prompts),
                "--output-dir",
                str(output_dir),
                "--resume",
            ]
        )
        == 1
    )
    assert "does not match the catalog used to prepare" in capsys.readouterr().err
    assert not result.outcomes_path.exists()
    assert not result.summary_path.exists()
    assert not result.report_path.exists()


def test_routing_prepare_cli_dispatches_both_retrieval_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[RoutingSplit, bool, bool]] = []
    data_dir = tmp_path / "routing-data"
    prepared = SimpleNamespace(
        manifest_path=data_dir / "manifest.json",
        development_requests_path=data_dir / "development.requests.jsonl",
        development_evaluator_path=data_dir / "development.evaluator.jsonl",
        development_row_count=200,
        test_requests_path=data_dir / "test.requests.jsonl",
        test_evaluator_path=data_dir / "test.evaluator.jsonl",
        test_row_count=200,
    )

    def prepare_data(config, _prompts, output_dir, *, overwrite):  # type: ignore[no-untyped-def]
        assert Path(config.data.output_dir) == data_dir
        assert config.data.local_files_only is True
        assert output_dir == data_dir
        assert overwrite is True
        return prepared

    def prepare_corpus(
        _config,
        _evaluator,
        _manifest,
        split,
        _output,
        *,
        overwrite,
    ):  # type: ignore[no-untyped-def]
        calls.append((split, overwrite, True))
        return SimpleNamespace(manifest_path=data_dir / "retrieval" / split / "manifest.json")

    monkeypatch.setattr("small_models_society.cli.prepare_routing_data", prepare_data)
    monkeypatch.setattr("small_models_society.cli.prepare_retrieval_corpus", prepare_corpus)

    assert (
        main(
            [
                "routing",
                "prepare",
                "--config",
                str(ROUTING_CONFIG),
                "--data-dir",
                str(data_dir),
                "--local-files-only",
                "--overwrite",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["development_rows"] == 200
    assert output["test_rows"] == 200
    assert calls == [
        (RoutingSplit.DEVELOPMENT, True, True),
        (RoutingSplit.TEST, True, True),
    ]


def test_workflow_matrix_cli_rejects_explicit_missing_adapter_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    _routing_artifacts(data_dir)
    monkeypatch.setattr("small_models_society.cli.detect_hardware", lambda _config: _hardware())
    monkeypatch.setattr("small_models_society.cli.docker_available", lambda: True)
    monkeypatch.setattr(
        "small_models_society.cli.sandbox_image_available",
        lambda _image: True,
    )

    assert (
        main(
            [
                "experiment",
                "workflow-matrix",
                "--config",
                str(ROUTING_CONFIG),
                "--data-dir",
                str(data_dir),
                "--adapter-root",
                str(tmp_path / "missing-adapters"),
                "--output-dir",
                str(tmp_path / "matrix"),
            ]
        )
        == 1
    )
    assert "explicit adapter root is unavailable" in capsys.readouterr().err
