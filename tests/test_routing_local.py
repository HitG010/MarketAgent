from __future__ import annotations

from pathlib import Path

import pytest

from small_models_society.inference.adapters import AdapterCatalog, AdapterSpec
from small_models_society.inference.config import InferenceConfig, load_inference_config
from small_models_society.inference.contracts import (
    ChatMessage,
    GenerationOutput,
    GenerationRequest,
    TextGenerationBackend,
)
from small_models_society.inference.hardware import HardwareReport
from small_models_society.inference.huggingface import InferenceOutOfMemoryError
from small_models_society.inference.prompts import PromptCatalog, load_prompt_catalog
from small_models_society.retrieval.bm25 import BM25Retriever
from small_models_society.retrieval.contracts import create_retrieval_document
from small_models_society.retrieval.rag import execute_rag
from small_models_society.routing.config import RoutingConfig, load_routing_config
from small_models_society.routing.contracts import (
    ActionAvailability,
    ActionOutcomeStatus,
    AvailabilityStatus,
    EnergyProvenance,
    OutputContract,
    RequestPolicyContext,
    SafetyStatus,
    WorkflowAction,
    WorkflowRequest,
    create_workflow_request,
)
from small_models_society.routing.local import LocalModelExecutor, hardware_fingerprint
from small_models_society.routing.policy import (
    ActionRuntimeContext,
    evaluate_action_availability,
)
from small_models_society.routing.registry import build_action_registry
from small_models_society.schemas import Domain

ROOT = Path(__file__).parents[1]
ROUTING_CONFIG = ROOT / "configs" / "routing.yaml"
INFERENCE_CONFIG = ROOT / "configs" / "inference.yaml"
PROMPT_CONFIG = ROOT / "configs" / "prompt_profiles.yaml"


def _routing_config(*approved: Domain) -> RoutingConfig:
    config = load_routing_config(ROUTING_CONFIG)
    approved_domains = set(approved)
    actions = {
        action_id: (
            configured.model_copy(
                update={
                    "approved": configured.adapter in approved_domains
                    if hasattr(configured, "adapter") and configured.adapter is not None
                    else configured.approved
                }
            )
            if hasattr(configured, "approved")
            else configured
        )
        for action_id, configured in config.actions.items()
    }
    return config.model_copy(update={"actions": actions})


def _inference_config() -> InferenceConfig:
    return load_inference_config(INFERENCE_CONFIG)


def _prompts() -> PromptCatalog:
    return load_prompt_catalog(PROMPT_CONFIG)


def _hardware(**updates: object) -> HardwareReport:
    values: dict[str, object] = {
        "ready": True,
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "python_version": "3.11.9",
        "package_versions": {
            "torch": "2.13.0",
            "transformers": "4.57.6",
            "safetensors": "0.8.0",
            "peft": "0.19.1",
        },
        "selected_device": "cpu",
        "selected_dtype": "float32",
        "cuda_available": False,
        "system_ram_gb": 16,
        "model_cache_path": "C:/cache/first",
        "model_cached": True,
    }
    values.update(updates)
    return HardwareReport.model_validate(values)


def _adapters(tmp_path: Path) -> AdapterCatalog:
    return AdapterCatalog(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        model_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        adapters={
            domain: AdapterSpec(
                name=domain,
                path=tmp_path / domain.value,
                sha256=f"{index + 1}" * 64,
                run_fingerprint=f"{index + 5}" * 64,
            )
            for index, domain in enumerate(Domain)
        },
    )


def _request(request_id: str = "routing-local") -> WorkflowRequest:
    defaults = _routing_config().policy_defaults
    return create_workflow_request(
        request_id=request_id,
        messages=(
            ChatMessage(role="system", content="Be precise."),
            ChatMessage(role="user", content="What is one plus two?"),
        ),
        output_contract=OutputContract.NUMERIC,
        policy=RequestPolicyContext(
            data_classification=defaults.data_classification,
            network_allowed=defaults.network_allowed,
            allowed_corpus_ids=defaults.allowed_corpus_ids,
            allowed_tool_ids=defaults.allowed_tool_ids,
            required_quality=defaults.required_quality,
            allow_unknown_output_safety=defaults.allow_unknown_output_safety,
        ),
    )


def _knowledge_request() -> WorkflowRequest:
    defaults = _routing_config().policy_defaults
    return create_workflow_request(
        request_id="routing-knowledge",
        messages=(
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="Question only."),
        ),
        output_contract=OutputContract.SHORT_ANSWER,
        policy=RequestPolicyContext(
            data_classification=defaults.data_classification,
            network_allowed=defaults.network_allowed,
            allowed_corpus_ids=defaults.allowed_corpus_ids,
            allowed_tool_ids=defaults.allowed_tool_ids,
            required_quality=defaults.required_quality,
            allow_unknown_output_safety=defaults.allow_unknown_output_safety,
        ),
        attributes={"retrieval_query": "What is the capital of France?"},
    )


def _available(action: WorkflowAction) -> ActionAvailability:
    return ActionAvailability(
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=AvailabilityStatus.AVAILABLE,
    )


class FakeBackend:
    def __init__(
        self,
        *,
        text: str = "3",
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return GenerationOutput(
            text=self.text,
            prompt_tokens=12,
            completion_tokens=2,
            latency_ms=4,
            metadata={"stop_reason": "eos", "input_truncated": False},
        )


class RecordingFactory:
    def __init__(self, backend: TextGenerationBackend) -> None:
        self.backend = backend
        self.calls: list[tuple[InferenceConfig, HardwareReport, AdapterCatalog | None]] = []

    def __call__(
        self,
        inference_config: InferenceConfig,
        hardware: HardwareReport,
        adapters: AdapterCatalog | None,
    ) -> TextGenerationBackend:
        self.calls.append((inference_config, hardware, adapters))
        return self.backend


def test_loads_one_adapter_backend_and_sends_base_adapter_base_sequence(
    tmp_path: Path,
) -> None:
    routing = _routing_config(Domain.MATH)
    catalog = _adapters(tmp_path)
    backend = FakeBackend()
    factory = RecordingFactory(backend)
    times = iter([1.0, 1.005, 2.0, 2.006, 3.0, 3.007])
    executor = LocalModelExecutor(
        routing,
        _inference_config(),
        _prompts(),
        _hardware(),
        catalog,
        backend_factory=factory,
        clock=lambda: next(times),
    )
    registry = build_action_registry(routing)
    base = registry.actions["local.qwen-base.v1"].action
    math = registry.actions["local.qwen-lora-math.v1"].action

    outcomes = [
        executor.execute(_request("base-one"), base, _available(base)),
        executor.execute(_request("math"), math, _available(math)),
        executor.execute(_request("base-two"), base, _available(base)),
    ]

    assert executor.backend_loaded is True
    assert executor.uses_adapter_backend is True
    assert len(factory.calls) == 1
    assert factory.calls[0][2] is catalog
    assert [request.adapter for request in backend.requests] == [None, "math", None]
    assert [request.max_new_tokens for request in backend.requests] == [512, 128, 512]
    assert all(outcome.status is ActionOutcomeStatus.COMPLETED for outcome in outcomes)
    assert [outcome.telemetry.wall_latency_ms for outcome in outcomes if outcome.telemetry] == (
        pytest.approx([5, 6, 7])
    )
    adapted = outcomes[1]
    assert adapted.metadata["adapter_id"] == "math"
    assert adapted.metadata["adapter_sha256"] == catalog.adapters[Domain.MATH].sha256
    assert adapted.metadata["adapter_run_fingerprint"] == (
        catalog.adapters[Domain.MATH].run_fingerprint
    )
    assert len(str(adapted.metadata["routing_config_fingerprint"])) == 64
    assert len(str(adapted.metadata["inference_config_fingerprint"])) == 64
    assert len(str(adapted.metadata["prompt_catalog_fingerprint"])) == 64
    assert len(str(adapted.metadata["hardware_fingerprint"])) == 64
    assert adapted.metadata["execution_fingerprint"] == executor.execution_fingerprint(math)
    assert executor.execution_fingerprint(base) != executor.execution_fingerprint(math)


def test_unapproved_catalog_does_not_force_peft_backend(tmp_path: Path) -> None:
    catalog = _adapters(tmp_path)
    backend = FakeBackend()
    factory = RecordingFactory(backend)
    routing = _routing_config()
    executor = LocalModelExecutor(
        routing,
        _inference_config(),
        _prompts(),
        _hardware(),
        catalog,
        backend_factory=factory,
    )
    base = build_action_registry(routing).actions["local.qwen-base.v1"].action

    outcome = executor.execute(_request(), base, _available(base))

    assert outcome.status is ActionOutcomeStatus.COMPLETED
    assert executor.verified_adapter_ids == ("code", "knowledge", "logic", "math")
    assert executor.uses_adapter_backend is False
    assert factory.calls[0][2] is None


def test_base_execution_fingerprint_survives_unrelated_adapter_approval(
    tmp_path: Path,
) -> None:
    base_routing = _routing_config()
    adapter_routing = _routing_config(Domain.MATH)
    base_action = build_action_registry(base_routing).actions["local.qwen-base.v1"].action
    adapter_base_action = (
        build_action_registry(adapter_routing).actions["local.qwen-base.v1"].action
    )
    base_executor = LocalModelExecutor(
        base_routing,
        _inference_config(),
        _prompts(),
        _hardware(),
        backend_factory=RecordingFactory(FakeBackend()),
    )
    adapter_executor = LocalModelExecutor(
        adapter_routing,
        _inference_config(),
        _prompts(),
        _hardware(),
        _adapters(tmp_path),
        backend_factory=RecordingFactory(FakeBackend()),
    )

    assert base_action.action_fingerprint == adapter_base_action.action_fingerprint
    assert base_executor.execution_fingerprint(base_action) == (
        adapter_executor.execution_fingerprint(adapter_base_action)
    )


def test_lora_and_rag_share_one_backend_with_adapter_to_base_transition(
    tmp_path: Path,
) -> None:
    routing = _routing_config(Domain.MATH)
    backend = FakeBackend(text="Paris")
    factory = RecordingFactory(backend)
    executor = LocalModelExecutor(
        routing,
        _inference_config(),
        _prompts(),
        _hardware(),
        _adapters(tmp_path),
        backend_factory=factory,
    )
    registry = build_action_registry(routing)
    math = registry.actions["local.qwen-lora-math.v1"].action
    rag = registry.actions["rag.bm25-qwen-base.v1"].action
    request = _knowledge_request()
    retriever = BM25Retriever(
        (create_retrieval_document("Paris", "Paris is the capital of France."),),
        routing.retrieval,
        "a" * 64,
    )

    executor.execute(_request("math-first"), math, _available(math))
    rag_outcome = execute_rag(
        request,
        rag,
        _available(rag),
        routing,
        retriever,
        executor.generation_backend(),
    )

    assert rag_outcome.status is ActionOutcomeStatus.COMPLETED
    assert len(factory.calls) == 1
    assert [generation.adapter for generation in backend.requests] == ["math", None]


def test_missing_adapter_catalog_stays_unavailable_and_never_loads_backend() -> None:
    routing = _routing_config(Domain.MATH)
    factory = RecordingFactory(FakeBackend())
    executor = LocalModelExecutor(
        routing,
        _inference_config(),
        _prompts(),
        _hardware(),
        backend_factory=factory,
    )
    registered = build_action_registry(routing).actions["local.qwen-lora-math.v1"]
    runtime = ActionRuntimeContext(
        local_model_ready=True,
        verified_adapter_ids=executor.verified_adapter_ids,
        available_corpus_ids=(),
        replay_action_ids=(),
        calculator_supported=False,
    )
    availability = evaluate_action_availability(_request(), registered, runtime)

    assert availability.status is AvailabilityStatus.UNAVAILABLE
    assert availability.reason_code == "adapter_artifact_missing"
    with pytest.raises(ValueError, match="must not be executed"):
        executor.execute(_request(), registered.action, availability)
    assert executor.backend_loaded is False
    assert factory.calls == []


def test_blocked_or_completed_only_work_never_loads_model() -> None:
    routing = _routing_config()
    factory = RecordingFactory(FakeBackend())
    executor = LocalModelExecutor(
        routing,
        _inference_config(),
        _prompts(),
        _hardware(),
        backend_factory=factory,
    )
    action = build_action_registry(routing).actions["local.qwen-base.v1"].action
    blocked = ActionAvailability(
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=AvailabilityStatus.BLOCKED,
        reason_code="already_completed_or_policy_blocked",
        rule_ids=("matrix.pending.v1",),
    )

    assert executor.backend_loaded is False
    with pytest.raises(ValueError, match="must not be executed"):
        executor.execute(_request(), action, blocked)
    assert executor.backend_loaded is False
    assert factory.calls == []


def test_completed_outcome_keeps_local_cost_energy_and_safety_unknown() -> None:
    routing = _routing_config()
    executor = LocalModelExecutor(
        routing,
        _inference_config(),
        _prompts(),
        _hardware(selected_device="mps", selected_dtype="float16"),
        backend_factory=RecordingFactory(FakeBackend(text="  3  ")),
    )
    action = build_action_registry(routing).actions["local.qwen-base.v1"].action

    outcome = executor.execute(_request(), action, _available(action))

    assert outcome.response == "3"
    assert outcome.safety.status is SafetyStatus.UNKNOWN
    assert outcome.telemetry is not None
    assert outcome.telemetry.provider_fee_usd == 0
    assert outcome.telemetry.compute_cost_usd is None
    assert outcome.telemetry.total_cost_usd is None
    assert outcome.telemetry.energy_provenance is EnergyProvenance.UNAVAILABLE
    assert outcome.telemetry.device == "mps"
    assert outcome.metadata["stop_reason"] == "eos"
    assert outcome.metadata["input_truncated"] is False


@pytest.mark.parametrize(
    ("backend", "error_type"),
    [
        (FakeBackend(text="   "), "EmptyGenerationError"),
        (FakeBackend(error=RuntimeError("temporary\nbackend failure")), "RuntimeError"),
    ],
)
def test_empty_or_failed_generation_becomes_error_outcome(
    backend: FakeBackend,
    error_type: str,
) -> None:
    routing = _routing_config()
    executor = LocalModelExecutor(
        routing,
        _inference_config(),
        _prompts(),
        _hardware(),
        backend_factory=RecordingFactory(backend),
    )
    action = build_action_registry(routing).actions["local.qwen-base.v1"].action

    outcome = executor.execute(_request(), action, _available(action))

    assert outcome.status is ActionOutcomeStatus.ERROR
    assert outcome.response is None
    assert outcome.error_type == error_type
    assert outcome.telemetry is not None
    assert outcome.safety.status is SafetyStatus.NOT_ASSESSED
    if backend.error is not None:
        assert outcome.error_message == "temporary backend failure"


def test_out_of_memory_remains_fatal() -> None:
    routing = _routing_config()
    executor = LocalModelExecutor(
        routing,
        _inference_config(),
        _prompts(),
        _hardware(),
        backend_factory=RecordingFactory(
            FakeBackend(error=InferenceOutOfMemoryError("out of memory"))
        ),
    )
    action = build_action_registry(routing).actions["local.qwen-base.v1"].action

    with pytest.raises(InferenceOutOfMemoryError, match="out of memory"):
        executor.execute(_request(), action, _available(action))


def test_hardware_fingerprint_excludes_cache_path_but_tracks_execution_device() -> None:
    first = _hardware(model_cache_path="C:/cache/first")
    relocated = _hardware(model_cache_path="D:/cache/second")
    mps = _hardware(
        model_cache_path="D:/cache/second",
        selected_device="mps",
        selected_dtype="float16",
    )

    assert hardware_fingerprint(first) == hardware_fingerprint(relocated)
    assert hardware_fingerprint(first) != hardware_fingerprint(mps)


@pytest.mark.parametrize(
    ("routing_update", "inference_update", "hardware_update", "message"),
    [
        ({"model_id": "different/model"}, {}, {}, "model IDs"),
        ({"revision": "a" * 40}, {}, {}, "model revisions"),
        ({}, {"seed": 7}, {}, "generation seeds"),
        ({}, {}, {"ready": False}, "not ready"),
        ({}, {}, {"model_id": "different/model"}, "hardware report model ID"),
        ({}, {}, {"revision": "a" * 40}, "hardware report revision"),
    ],
)
def test_rejects_runtime_identity_drift_before_model_load(
    routing_update: dict[str, object],
    inference_update: dict[str, object],
    hardware_update: dict[str, object],
    message: str,
) -> None:
    routing = _routing_config()
    if routing_update:
        routing = routing.model_copy(
            update={"model": routing.model.model_copy(update=routing_update)}
        )
    inference = _inference_config()
    if inference_update:
        inference = inference.model_copy(
            update={"generation": inference.generation.model_copy(update=inference_update)}
        )
    factory = RecordingFactory(FakeBackend())

    with pytest.raises(ValueError, match=message):
        LocalModelExecutor(
            routing,
            inference,
            _prompts(),
            _hardware(**hardware_update),
            backend_factory=factory,
        )

    assert factory.calls == []
