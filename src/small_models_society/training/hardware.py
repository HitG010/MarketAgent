"""Training-stack probing and explicit accelerator readiness selection."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from small_models_society.inference.hardware import (
    model_snapshot_is_complete,
    model_snapshot_path,
)
from small_models_society.schemas import StrictModel
from small_models_society.training.config import (
    TrainingConfig,
    TrainingDevicePreference,
    TrainingDTypePreference,
)

MINIMUM_ACCELERATOR_MEMORY_GB = 12.0
RECOMMENDED_ACCELERATOR_MEMORY_GB = 16.0
REQUIRED_TRAINING_PACKAGES = (
    "torch",
    "transformers",
    "safetensors",
    "accelerate",
    "peft",
    "trl",
)


class TrainingRuntimeCapabilities(StrictModel):
    python_version: str
    platform_system: str
    platform_machine: str
    package_versions: dict[str, str | None]
    import_errors: dict[str, str] = Field(default_factory=dict)
    cuda_available: bool = False
    cuda_bfloat16_supported: bool = False
    cuda_device_name: str | None = None
    cuda_runtime_version: str | None = None
    cuda_vram_gb: float | None = Field(default=None, ge=0)
    mps_built: bool = False
    mps_available: bool = False
    system_ram_gb: float | None = Field(default=None, ge=0)
    model_cache_path: str
    model_cached: bool = False
    artifact_root: str
    artifact_root_writable: bool = False
    artifact_error: str | None = None
    mps_fallback_enabled: bool = False


class TrainingHardwareReport(StrictModel):
    ready: bool
    model_id: str
    revision: str
    python_version: str
    platform_system: str
    platform_machine: str
    package_versions: dict[str, str | None]
    selected_device: Literal["cpu", "cuda", "mps"]
    selected_dtype: Literal["float32", "float16", "bfloat16"]
    estimated_fit: Literal["likely", "tight", "insufficient", "unknown", "cpu_debug_only"]
    cpu_debug_only: bool
    cuda_available: bool
    cuda_device_name: str | None = None
    cuda_runtime_version: str | None = None
    cuda_vram_gb: float | None = Field(default=None, ge=0)
    mps_built: bool
    mps_available: bool
    mps_fallback_enabled: bool
    system_ram_gb: float | None = Field(default=None, ge=0)
    model_cache_path: str
    model_cached: bool
    artifact_root: str
    artifact_root_writable: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _optional_import(module_name: str) -> tuple[Any | None, str | None]:
    if importlib.util.find_spec(module_name) is None:
        return None, None
    try:
        return importlib.import_module(module_name), None
    except (ImportError, OSError) as error:
        message = " ".join(str(error).split())
        return None, f"{type(error).__name__}: {message}"


def _artifact_writable(path: Path) -> tuple[bool, str | None]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".sms-write-test-", delete=True):
            pass
    except OSError as error:
        return False, " ".join(str(error).split())
    return True, None


def collect_training_capabilities(
    config: TrainingConfig,
    cache_dir: Path | None = None,
) -> TrainingRuntimeCapabilities:
    """Inspect the optional training stack and hardware without loading model weights."""

    modules: dict[str, Any | None] = {}
    import_errors: dict[str, str] = {}
    for package in (*REQUIRED_TRAINING_PACKAGES, "psutil"):
        module, error = _optional_import(package)
        modules[package] = module
        if error is not None:
            import_errors[package] = error

    torch_module = modules["torch"]
    cuda_available = False
    cuda_bfloat16_supported = False
    cuda_device_name: str | None = None
    cuda_runtime_version: str | None = None
    cuda_vram_gb: float | None = None
    mps_built = False
    mps_available = False
    if torch_module is not None:
        cuda_runtime_version = getattr(getattr(torch_module, "version", None), "cuda", None)
        cuda_available = bool(torch_module.cuda.is_available())
        cuda_bfloat16_supported = bool(cuda_available and torch_module.cuda.is_bf16_supported())
        if cuda_available:
            cuda_device_name = str(torch_module.cuda.get_device_name(0))
            total_memory = int(torch_module.cuda.get_device_properties(0).total_memory)
            cuda_vram_gb = total_memory / 1024**3
        mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
        if mps_backend is not None:
            mps_built = bool(mps_backend.is_built())
            mps_available = bool(mps_backend.is_available())

    system_ram_gb: float | None = None
    psutil_module = modules["psutil"]
    if psutil_module is not None:
        system_ram_gb = int(psutil_module.virtual_memory().total) / 1024**3

    snapshot_path = model_snapshot_path(
        config.model.model_id,
        config.model.revision,
        cache_dir,
    )
    artifact_root = Path(config.output.adapter_root)
    artifact_root_writable, artifact_error = _artifact_writable(artifact_root)
    package_versions = {
        package: _package_version(package) for package in (*REQUIRED_TRAINING_PACKAGES, "psutil")
    }
    return TrainingRuntimeCapabilities(
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        package_versions=package_versions,
        import_errors=import_errors,
        cuda_available=cuda_available,
        cuda_bfloat16_supported=cuda_bfloat16_supported,
        cuda_device_name=cuda_device_name,
        cuda_runtime_version=cuda_runtime_version,
        cuda_vram_gb=cuda_vram_gb,
        mps_built=mps_built,
        mps_available=mps_available,
        system_ram_gb=system_ram_gb,
        model_cache_path=str(snapshot_path),
        model_cached=model_snapshot_is_complete(snapshot_path),
        artifact_root=str(artifact_root),
        artifact_root_writable=artifact_root_writable,
        artifact_error=artifact_error,
        mps_fallback_enabled=os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
    )


def _select_device(
    preference: TrainingDevicePreference,
    capabilities: TrainingRuntimeCapabilities,
    errors: list[str],
) -> Literal["cpu", "cuda", "mps"]:
    if preference is TrainingDevicePreference.CPU:
        return "cpu"
    if preference is TrainingDevicePreference.CUDA:
        if not capabilities.cuda_available:
            errors.append("CUDA training was requested but no CUDA device is available.")
        return "cuda"
    if preference is TrainingDevicePreference.MPS:
        if not capabilities.mps_available:
            errors.append("MPS training was requested but no MPS device is available.")
        return "mps"
    if capabilities.cuda_available:
        return "cuda"
    if capabilities.mps_available:
        return "mps"
    return "cpu"


def _select_dtype(
    preference: TrainingDTypePreference,
    device: Literal["cpu", "cuda", "mps"],
    capabilities: TrainingRuntimeCapabilities,
    errors: list[str],
) -> Literal["float32", "float16", "bfloat16"]:
    if preference is TrainingDTypePreference.AUTO:
        if device == "cpu":
            return "float32"
        if device == "mps":
            return "float16"
        return "bfloat16" if capabilities.cuda_bfloat16_supported else "float16"
    selected = preference.value
    if device == "cpu" and selected != "float32":
        errors.append("CPU training supports only float32 in this project configuration.")
    if device == "mps" and selected == "bfloat16":
        errors.append("MPS training does not use bfloat16 in this project configuration.")
    if selected == "bfloat16" and device == "cuda" and not capabilities.cuda_bfloat16_supported:
        errors.append("bfloat16 was requested but the CUDA device does not support it.")
    return selected  # type: ignore[return-value]


def _accelerator_fit(
    device: Literal["cpu", "cuda", "mps"],
    capabilities: TrainingRuntimeCapabilities,
    warnings: list[str],
    errors: list[str],
) -> Literal["likely", "tight", "insufficient", "unknown", "cpu_debug_only"]:
    if device == "cpu":
        return "cpu_debug_only"
    memory_gb = capabilities.cuda_vram_gb if device == "cuda" else capabilities.system_ram_gb
    memory_name = "CUDA VRAM" if device == "cuda" else "Apple unified memory"
    if memory_gb is None:
        warnings.append(f"{memory_name} could not be detected; training fit is unknown.")
        return "unknown"
    if memory_gb < MINIMUM_ACCELERATOR_MEMORY_GB:
        errors.append(
            f"{memory_name} is {memory_gb:.1f} GB; standard LoRA requires at least "
            f"{MINIMUM_ACCELERATOR_MEMORY_GB:.1f} GB for this pilot."
        )
        return "insufficient"
    if memory_gb < RECOMMENDED_ACCELERATOR_MEMORY_GB:
        warnings.append(
            f"{memory_name} is {memory_gb:.1f} GB; {RECOMMENDED_ACCELERATOR_MEMORY_GB:.1f} "
            "GB is recommended and the run may require the explicit 384-token fallback."
        )
        return "tight"
    return "likely"


def select_training_hardware(
    config: TrainingConfig,
    capabilities: TrainingRuntimeCapabilities,
    *,
    allow_cpu: bool = False,
) -> TrainingHardwareReport:
    """Resolve training device policy without silently falling back to CPU."""

    warnings: list[str] = []
    errors: list[str] = []
    missing_packages = [
        package
        for package in REQUIRED_TRAINING_PACKAGES
        if capabilities.package_versions.get(package) is None
    ]
    if missing_packages:
        errors.append(
            "Missing training packages: "
            f"{', '.join(missing_packages)}. Install requirements-training.lock."
        )
    failed_imports = {
        package: error
        for package, error in capabilities.import_errors.items()
        if package in REQUIRED_TRAINING_PACKAGES
    }
    if failed_imports:
        details = "; ".join(
            f"{package}: {error}" for package, error in sorted(failed_imports.items())
        )
        errors.append(f"Training packages are installed but failed to import: {details}")

    device = _select_device(config.model.device, capabilities, errors)
    dtype = _select_dtype(config.model.dtype, device, capabilities, errors)
    cpu_debug_only = device == "cpu"
    if cpu_debug_only:
        message = (
            "No supported accelerator was selected. CPU is a tiny-smoke/debug path only; "
            "pass the explicit CPU override to proceed."
        )
        if allow_cpu:
            warnings.append(message)
        else:
            errors.append(message)
    estimated_fit = _accelerator_fit(device, capabilities, warnings, errors)

    if device == "mps":
        warnings.append("The PyTorch MPS backend is beta; run the one-step smoke first.")
        if capabilities.mps_fallback_enabled:
            warnings.append(
                "PYTORCH_ENABLE_MPS_FALLBACK=1 is set; unsupported operations may run on CPU."
            )
    if capabilities.system_ram_gb is None:
        detail = capabilities.import_errors.get("psutil")
        suffix = f" ({detail})" if detail else ""
        warnings.append(f"System memory could not be detected; install or repair psutil.{suffix}")
    if not capabilities.artifact_root_writable:
        detail = f" ({capabilities.artifact_error})" if capabilities.artifact_error else ""
        errors.append(
            f"Training artifact root is not writable: {capabilities.artifact_root}.{detail}"
        )
    if not capabilities.model_cached:
        message = f"Model snapshot is not cached at {capabilities.model_cache_path}."
        if config.model.local_files_only:
            errors.append(f"{message} Disable local_files_only to allow the first download.")
        else:
            warnings.append(f"{message} The first training run will download model files.")

    return TrainingHardwareReport(
        ready=not errors,
        model_id=config.model.model_id,
        revision=config.model.revision,
        python_version=capabilities.python_version,
        platform_system=capabilities.platform_system,
        platform_machine=capabilities.platform_machine,
        package_versions=capabilities.package_versions,
        selected_device=device,
        selected_dtype=dtype,
        estimated_fit=estimated_fit,
        cpu_debug_only=cpu_debug_only,
        cuda_available=capabilities.cuda_available,
        cuda_device_name=capabilities.cuda_device_name,
        cuda_runtime_version=capabilities.cuda_runtime_version,
        cuda_vram_gb=capabilities.cuda_vram_gb,
        mps_built=capabilities.mps_built,
        mps_available=capabilities.mps_available,
        mps_fallback_enabled=capabilities.mps_fallback_enabled,
        system_ram_gb=capabilities.system_ram_gb,
        model_cache_path=capabilities.model_cache_path,
        model_cached=capabilities.model_cached,
        artifact_root=capabilities.artifact_root,
        artifact_root_writable=capabilities.artifact_root_writable,
        warnings=warnings,
        errors=errors,
    )


def detect_training_hardware(
    config: TrainingConfig,
    cache_dir: Path | None = None,
    *,
    allow_cpu: bool = False,
) -> TrainingHardwareReport:
    capabilities = collect_training_capabilities(config, cache_dir)
    return select_training_hardware(config, capabilities, allow_cpu=allow_cpu)
