"""Lazy local-model workflow execution over the existing inference backends."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from time import perf_counter

from small_models_society.data.prepare import canonical_json
from small_models_society.inference.adapters import (
    AdapterCatalog,
    AdapterSpec,
    PeftHuggingFaceBackend,
)
from small_models_society.inference.config import InferenceConfig
from small_models_society.inference.contracts import (
    GenerationOutput,
    GenerationRequest,
    TextGenerationBackend,
)
from small_models_society.inference.hardware import HardwareReport
from small_models_society.inference.huggingface import (
    HuggingFaceBackend,
    InferenceOutOfMemoryError,
)
from small_models_society.inference.prompts import PromptCatalog
from small_models_society.routing.config import (
    ActionKind,
    LocalModelActionConfig,
    RoutingConfig,
)
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
from small_models_society.schemas import Domain

LOCAL_EXECUTOR_ID = "huggingface.qwen.v1"
Clock = Callable[[], float]
BackendFactory = Callable[
    [InferenceConfig, HardwareReport, AdapterCatalog | None],
    TextGenerationBackend,
]


def hardware_fingerprint(hardware: HardwareReport) -> str:
    """Hash behavior-relevant hardware/runtime fields without cache paths."""

    payload = {
        "model_id": hardware.model_id,
        "model_revision": hardware.revision,
        "python_version": hardware.python_version,
        "package_versions": hardware.package_versions,
        "selected_device": hardware.selected_device,
        "selected_dtype": hardware.selected_dtype,
        "cuda_available": hardware.cuda_available,
        "cuda_device_name": hardware.cuda_device_name,
        "cuda_runtime_version": hardware.cuda_runtime_version,
        "cuda_vram_gb": hardware.cuda_vram_gb,
        "mps_built": hardware.mps_built,
        "mps_available": hardware.mps_available,
        "system_ram_gb": hardware.system_ram_gb,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_local_execution_context(
    routing_config: RoutingConfig,
    inference_config: InferenceConfig,
    prompts: PromptCatalog,
    hardware: HardwareReport,
    adapters: AdapterCatalog | None = None,
    *,
    require_ready: bool,
) -> None:
    routing_model = routing_config.model
    inference_model = inference_config.model
    if routing_model.model_id != inference_model.model_id:
        raise ValueError("routing and inference model IDs do not match")
    if routing_model.revision != inference_model.revision:
        raise ValueError("routing and inference model revisions do not match")
    if routing_config.seed != inference_config.generation.seed:
        raise ValueError("routing and inference generation seeds do not match")
    prompts.get(routing_model.prompt_profile)
    if require_ready and not hardware.ready:
        raise ValueError("hardware report is not ready for local workflow execution")
    if hardware.model_id != routing_model.model_id:
        raise ValueError("hardware report model ID does not match routing configuration")
    if hardware.revision != routing_model.revision:
        raise ValueError("hardware report revision does not match routing configuration")
    if adapters is not None:
        if adapters.model_id != routing_model.model_id:
            raise ValueError("adapter catalog model ID does not match routing configuration")
        if adapters.model_revision != routing_model.revision:
            raise ValueError("adapter catalog revision does not match routing configuration")


def local_execution_fingerprint(
    action: WorkflowAction,
    inference_config: InferenceConfig,
    prompts: PromptCatalog,
    hardware: HardwareReport,
    adapter: AdapterSpec | None,
) -> str:
    if action.kind is not ActionKind.LOCAL_MODEL:
        raise ValueError("local execution fingerprint requires a local-model action")
    expected_adapter_id = adapter.name.value if adapter is not None else None
    if action.adapter_id != expected_adapter_id:
        raise ValueError("local execution fingerprint adapter does not match action")
    payload = {
        "action_fingerprint": action.action_fingerprint,
        "inference_config_fingerprint": inference_config.fingerprint(),
        "prompt_catalog_fingerprint": prompts.fingerprint(),
        "hardware_fingerprint": hardware_fingerprint(hardware),
        "adapter_id": expected_adapter_id,
        "adapter_sha256": adapter.sha256 if adapter is not None else None,
        "adapter_run_fingerprint": adapter.run_fingerprint if adapter is not None else None,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _default_backend_factory(
    inference_config: InferenceConfig,
    hardware: HardwareReport,
    adapters: AdapterCatalog | None,
) -> TextGenerationBackend:
    if adapters is None:
        return HuggingFaceBackend(inference_config, hardware)
    return PeftHuggingFaceBackend(inference_config, hardware, adapters)


class LocalModelExecutor:
    """Load one backend lazily and execute base or verified adapter actions."""

    def __init__(
        self,
        routing_config: RoutingConfig,
        inference_config: InferenceConfig,
        prompts: PromptCatalog,
        hardware: HardwareReport,
        adapters: AdapterCatalog | None = None,
        *,
        backend_factory: BackendFactory = _default_backend_factory,
        clock: Clock = perf_counter,
    ) -> None:
        self.routing_config = routing_config
        self.inference_config = inference_config
        self.prompts = prompts
        self.hardware = hardware
        self.adapters = adapters
        self.backend_factory = backend_factory
        self.clock = clock
        self._backend: TextGenerationBackend | None = None
        validate_local_execution_context(
            routing_config,
            inference_config,
            prompts,
            hardware,
            adapters,
            require_ready=True,
        )
        self._approved_adapter_ids = frozenset(
            configured.adapter.value
            for configured in routing_config.actions.values()
            if isinstance(configured, LocalModelActionConfig)
            and configured.approved
            and configured.adapter is not None
        )

    @property
    def backend_loaded(self) -> bool:
        return self._backend is not None

    @property
    def verified_adapter_ids(self) -> tuple[str, ...]:
        if self.adapters is None:
            return ()
        return tuple(sorted(domain.value for domain in self.adapters.adapters))

    @property
    def uses_adapter_backend(self) -> bool:
        return bool(self.adapters is not None and self._approved_adapter_ids)

    def generation_backend(self) -> TextGenerationBackend:
        """Return the shared backend, loading it only on first execution demand."""

        if self._backend is None:
            adapter_catalog = self.adapters if self.uses_adapter_backend else None
            self._backend = self.backend_factory(
                self.inference_config,
                self.hardware,
                adapter_catalog,
            )
        return self._backend

    def _configured_action(self, action: WorkflowAction) -> LocalModelActionConfig:
        configured = self.routing_config.actions.get(action.action_id)
        if not isinstance(configured, LocalModelActionConfig):
            raise ValueError("local executor received an unconfigured local action")
        expected_adapter = configured.adapter.value if configured.adapter is not None else None
        if action.adapter_id != expected_adapter:
            raise ValueError("local action adapter does not match routing configuration")
        if action.max_new_tokens != configured.max_new_tokens:
            raise ValueError("local action token budget does not match routing configuration")
        if action.adapter_id is not None and not configured.approved:
            raise ValueError("unapproved adapter action cannot be executed")
        return configured

    def _adapter_spec(self, action: WorkflowAction) -> AdapterSpec | None:
        if action.adapter_id is None:
            return None
        if self.adapters is None:
            raise ValueError("adapter action requires a verified adapter catalog")
        try:
            return self.adapters.adapters[Domain(action.adapter_id)]
        except (KeyError, ValueError) as error:
            raise ValueError(f"verified adapter is unavailable: {action.adapter_id}") from error

    def _validate_execution(
        self,
        action: WorkflowAction,
        availability: ActionAvailability,
    ) -> AdapterSpec | None:
        if (
            action.kind is not ActionKind.LOCAL_MODEL
            or action.executor_id != LOCAL_EXECUTOR_ID
            or action.model_id != self.routing_config.model.model_id
            or action.model_revision != self.routing_config.model.revision
            or action.max_new_tokens is None
        ):
            raise ValueError("local executor received an incompatible workflow action")
        if (
            availability.action_id != action.action_id
            or availability.action_fingerprint != action.action_fingerprint
        ):
            raise ValueError("local availability does not match the workflow action")
        if availability.status is not AvailabilityStatus.AVAILABLE:
            raise ValueError("blocked or unavailable local actions must not be executed")
        self._configured_action(action)
        return self._adapter_spec(action)

    def _generation_request(
        self,
        request: WorkflowRequest,
        action: WorkflowAction,
    ) -> GenerationRequest:
        assert action.max_new_tokens is not None
        return GenerationRequest(
            request_id=request.request_id,
            profile=self.routing_config.model.prompt_profile,
            adapter=action.adapter_id,
            messages=list(request.messages),
            max_new_tokens=action.max_new_tokens,
        )

    def execution_fingerprint(self, action: WorkflowAction) -> str:
        """Bind an action artifact to model, prompt, hardware, and adapter runtime identity."""

        self._configured_action(action)
        adapter = self._adapter_spec(action)
        return local_execution_fingerprint(
            action,
            self.inference_config,
            self.prompts,
            self.hardware,
            adapter,
        )

    def _metadata(
        self,
        action: WorkflowAction,
        adapter: AdapterSpec | None,
        generation: GenerationOutput | None = None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "executor_id": LOCAL_EXECUTOR_ID,
            "routing_config_fingerprint": self.routing_config.fingerprint(),
            "inference_config_fingerprint": self.inference_config.fingerprint(),
            "prompt_catalog_fingerprint": self.prompts.fingerprint(),
            "hardware_fingerprint": hardware_fingerprint(self.hardware),
            "execution_fingerprint": self.execution_fingerprint(action),
            "model_id": self.routing_config.model.model_id,
            "model_revision": self.routing_config.model.revision,
            "prompt_profile": self.routing_config.model.prompt_profile,
            "adapter_id": adapter.name.value if adapter is not None else None,
            "adapter_sha256": adapter.sha256 if adapter is not None else None,
            "adapter_run_fingerprint": (adapter.run_fingerprint if adapter is not None else None),
        }
        if generation is not None:
            stop_reason = generation.metadata.get("stop_reason")
            input_truncated = generation.metadata.get("input_truncated")
            if isinstance(stop_reason, str):
                metadata["stop_reason"] = stop_reason
            if isinstance(input_truncated, bool):
                metadata["input_truncated"] = input_truncated
        return metadata

    def _telemetry(
        self,
        wall_latency_ms: float,
        generation: GenerationOutput | None = None,
    ) -> ActionTelemetry:
        return ActionTelemetry(
            wall_latency_ms=wall_latency_ms,
            prompt_tokens=(generation.prompt_tokens if generation is not None else None),
            completion_tokens=(generation.completion_tokens if generation is not None else None),
            provider_fee_usd=0,
            compute_cost_usd=None,
            total_cost_usd=None,
            energy_provenance=EnergyProvenance.UNAVAILABLE,
            device=self.hardware.selected_device,
        )

    def execute(
        self,
        request: WorkflowRequest,
        action: WorkflowAction,
        availability: ActionAvailability,
    ) -> ActionOutcome:
        adapter = self._validate_execution(action, availability)
        generation_request = self._generation_request(request, action)
        started = self.clock()
        try:
            generation = self.generation_backend().generate(generation_request)
        except InferenceOutOfMemoryError:
            raise
        except Exception as error:
            wall_latency_ms = (self.clock() - started) * 1000
            message = " ".join(str(error).split())[:500]
            return ActionOutcome(
                request_id=request.request_id,
                request_fingerprint=request.request_fingerprint,
                action_id=action.action_id,
                action_fingerprint=action.action_fingerprint,
                status=ActionOutcomeStatus.ERROR,
                availability=availability,
                safety=SafetyAssessment(
                    status=SafetyStatus.NOT_ASSESSED,
                    source=LOCAL_EXECUTOR_ID,
                ),
                telemetry=self._telemetry(wall_latency_ms),
                error_type=type(error).__name__,
                error_message=message or "local generation failed without an error message",
                metadata=self._metadata(action, adapter),
            )

        wall_latency_ms = (self.clock() - started) * 1000
        response = generation.text.strip()
        if not response:
            return ActionOutcome(
                request_id=request.request_id,
                request_fingerprint=request.request_fingerprint,
                action_id=action.action_id,
                action_fingerprint=action.action_fingerprint,
                status=ActionOutcomeStatus.ERROR,
                availability=availability,
                safety=SafetyAssessment(
                    status=SafetyStatus.NOT_ASSESSED,
                    source=LOCAL_EXECUTOR_ID,
                ),
                telemetry=self._telemetry(wall_latency_ms, generation),
                error_type="EmptyGenerationError",
                error_message="local generator returned an empty response",
                metadata=self._metadata(action, adapter, generation),
            )
        return ActionOutcome(
            request_id=request.request_id,
            request_fingerprint=request.request_fingerprint,
            action_id=action.action_id,
            action_fingerprint=action.action_fingerprint,
            status=ActionOutcomeStatus.COMPLETED,
            availability=availability,
            response=response,
            safety=SafetyAssessment(
                status=SafetyStatus.UNKNOWN,
                source="not_assessed",
            ),
            telemetry=self._telemetry(wall_latency_ms, generation),
            metadata=self._metadata(action, adapter, generation),
        )
