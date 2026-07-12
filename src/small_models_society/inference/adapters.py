"""Verified LoRA adapter catalog and switchable PEFT inference backend."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import Field

from small_models_society.data.prepare import sha256_bytes
from small_models_society.inference.config import InferenceConfig
from small_models_society.inference.contracts import AdapterReference, GenerationRequest
from small_models_society.inference.hardware import HardwareReport
from small_models_society.inference.huggingface import (
    HuggingFaceBackend,
    InferenceDependencyError,
    InferenceModules,
    load_inference_modules,
)
from small_models_society.schemas import Domain, StrictModel
from small_models_society.training.config import TrainingConfig
from small_models_society.training.runner import AdapterRunManifest


class AdapterSpec(StrictModel):
    name: Domain
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    def reference(self) -> AdapterReference:
        return AdapterReference(
            name=self.name.value,
            sha256=self.sha256,
            run_fingerprint=self.run_fingerprint,
        )


class AdapterCatalog(StrictModel):
    model_id: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapters: dict[Domain, AdapterSpec]


@dataclass(frozen=True)
class AdapterInferenceModules:
    inference: InferenceModules
    peft_model: Any
    peft_version: str


def load_adapter_inference_modules() -> AdapterInferenceModules:
    """Import PEFT in addition to the base inference dependencies."""

    inference = load_inference_modules()
    try:
        peft_module = importlib.import_module("peft")
    except (ImportError, OSError) as error:
        raise InferenceDependencyError(
            "LoRA inference dependencies are unavailable. Install requirements-training.lock."
        ) from error
    return AdapterInferenceModules(
        inference=inference,
        peft_model=peft_module.PeftModel,
        peft_version=str(peft_module.__version__),
    )


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return cast(Mapping[str, Any], value)


def _load_json(path: Path, description: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), description)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description}: {path}") from error


def _validate_adapter_configuration(
    path: Path,
    training_config: TrainingConfig,
) -> None:
    config = _load_json(path / "adapter_config.json", "adapter configuration")
    expected_targets = {module.value for module in training_config.lora.target_modules}
    actual_targets = config.get("target_modules")
    if not isinstance(actual_targets, list) or set(actual_targets) != expected_targets:
        raise ValueError(f"adapter target modules do not match training config: {path}")
    expected: dict[str, object] = {
        "base_model_name_or_path": training_config.model.model_id,
        "revision": training_config.model.revision,
        "r": training_config.lora.rank,
        "lora_alpha": training_config.lora.alpha,
        "lora_dropout": training_config.lora.dropout,
        "bias": training_config.lora.bias,
        "task_type": training_config.lora.task_type,
    }
    mismatches = {
        field: {"expected": expected_value, "actual": config.get(field)}
        for field, expected_value in expected.items()
        if config.get(field) != expected_value
    }
    if mismatches:
        raise ValueError(f"adapter configuration mismatch at {path}: {mismatches}")


def load_adapter_catalog(
    root: Path,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
) -> AdapterCatalog:
    """Verify four completed adapters before loading the base model."""

    if inference_config.model.model_id != training_config.model.model_id:
        raise ValueError("inference and training model IDs do not match")
    if inference_config.model.revision != training_config.model.revision:
        raise ValueError("inference and training model revisions do not match")
    adapters: dict[Domain, AdapterSpec] = {}
    for domain in Domain:
        path = root / domain.value
        try:
            manifest = AdapterRunManifest.model_validate_json(
                (path / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid completed adapter manifest: {path}") from error
        if manifest.status != "completed":
            raise ValueError(f"adapter manifest is not completed: {path}")
        if manifest.specialist is not domain:
            raise ValueError(f"adapter specialist does not match directory: {path}")
        if manifest.model_id != training_config.model.model_id:
            raise ValueError(f"adapter base model ID does not match: {path}")
        if manifest.model_revision != training_config.model.revision:
            raise ValueError(f"adapter base revision does not match: {path}")
        if manifest.training_config_fingerprint != training_config.fingerprint():
            raise ValueError(f"adapter training configuration does not match: {path}")
        weights_path = path / "adapter_model.safetensors"
        if not weights_path.is_file() or weights_path.stat().st_size <= 0:
            raise ValueError(f"adapter weights are missing: {path}")
        weights_sha256 = sha256_bytes(weights_path.read_bytes())
        if weights_sha256 != manifest.adapter_sha256:
            raise ValueError(f"adapter weights hash does not match manifest: {path}")
        _validate_adapter_configuration(path, training_config)
        adapters[domain] = AdapterSpec(
            name=domain,
            path=path,
            sha256=weights_sha256,
            run_fingerprint=manifest.run_fingerprint,
        )
    return AdapterCatalog(
        model_id=training_config.model.model_id,
        model_revision=training_config.model.revision,
        adapters=adapters,
    )


class PeftHuggingFaceBackend(HuggingFaceBackend):
    """Load one base model, attach verified adapters, and switch per request."""

    def __init__(
        self,
        config: InferenceConfig,
        hardware: HardwareReport,
        catalog: AdapterCatalog,
        modules: AdapterInferenceModules | None = None,
        **kwargs: Any,
    ) -> None:
        if config.model.model_id != catalog.model_id:
            raise ValueError("adapter catalog model ID does not match inference config")
        if config.model.revision != catalog.model_revision:
            raise ValueError("adapter catalog revision does not match inference config")
        adapter_modules = modules or load_adapter_inference_modules()
        self.adapter_modules = adapter_modules
        self.adapter_catalog = catalog
        super().__init__(
            config,
            hardware,
            modules=adapter_modules.inference,
            **kwargs,
        )
        ordered = [catalog.adapters[domain] for domain in Domain]
        first = ordered[0]
        self.model = adapter_modules.peft_model.from_pretrained(
            self.model,
            first.path,
            adapter_name=first.name.value,
            is_trainable=False,
            low_cpu_mem_usage=True,
        )
        for adapter in ordered[1:]:
            self.model.load_adapter(
                adapter.path,
                adapter_name=adapter.name.value,
                is_trainable=False,
                low_cpu_mem_usage=True,
            )
        self.model.eval()

    def _activate_request(
        self,
        request: GenerationRequest,
    ) -> AbstractContextManager[None]:
        if request.adapter is None:
            return cast(AbstractContextManager[None], self.model.disable_adapter())
        try:
            adapter = self.adapter_catalog.adapters[Domain(request.adapter)]
        except (ValueError, KeyError) as error:
            raise ValueError(f"unknown LoRA adapter: {request.adapter}") from error
        self.model.set_adapter(adapter.name.value)
        return nullcontext()

    def _request_metadata(self, request: GenerationRequest) -> dict[str, object]:
        if request.adapter is None:
            return {
                "adapter": None,
                "adapter_sha256": None,
                "peft_version": self.adapter_modules.peft_version,
            }
        adapter = self.adapter_catalog.adapters[Domain(request.adapter)]
        return {
            "adapter": adapter.name.value,
            "adapter_sha256": adapter.sha256,
            "adapter_run_fingerprint": adapter.run_fingerprint,
            "peft_version": self.adapter_modules.peft_version,
        }
