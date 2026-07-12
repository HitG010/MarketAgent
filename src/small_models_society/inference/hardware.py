"""Optional-runtime probing and deterministic device selection."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from small_models_society.inference.config import (
    DevicePreference,
    DTypePreference,
    InferenceConfig,
)
from small_models_society.schemas import StrictModel

MINIMUM_CUDA_VRAM_GB = 4.5
MINIMUM_CPU_RAM_GB = 8.0


class RuntimeCapabilities(StrictModel):
    python_version: str
    torch_version: str | None = None
    transformers_version: str | None = None
    safetensors_version: str | None = None
    psutil_version: str | None = None
    import_errors: dict[str, str] = Field(default_factory=dict)
    cuda_available: bool = False
    cuda_bfloat16_supported: bool = False
    cuda_device_name: str | None = None
    cuda_runtime_version: str | None = None
    cuda_vram_gb: float | None = Field(default=None, ge=0)
    system_ram_gb: float | None = Field(default=None, ge=0)
    model_cache_path: str
    model_cached: bool = False


class HardwareReport(StrictModel):
    ready: bool
    model_id: str
    revision: str
    python_version: str
    package_versions: dict[str, str | None]
    selected_device: Literal["cpu", "cuda"]
    selected_dtype: Literal["float32", "float16", "bfloat16"]
    cuda_available: bool
    cuda_device_name: str | None = None
    cuda_runtime_version: str | None = None
    cuda_vram_gb: float | None = Field(default=None, ge=0)
    system_ram_gb: float | None = Field(default=None, ge=0)
    model_cache_path: str
    model_cached: bool
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


def model_snapshot_path(
    model_id: str,
    revision: str,
    cache_dir: Path | None = None,
) -> Path:
    if cache_dir is not None:
        hub_cache = cache_dir
    elif os.getenv("HF_HUB_CACHE"):
        hub_cache = Path(os.environ["HF_HUB_CACHE"])
    elif os.getenv("HUGGINGFACE_HUB_CACHE"):
        hub_cache = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    else:
        default_cache_home = (
            Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface"
        )
        huggingface_home = Path(os.getenv("HF_HOME", default_cache_home))
        hub_cache = huggingface_home / "hub"
    model_directory = f"models--{model_id.replace('/', '--')}"
    return hub_cache / model_directory / "snapshots" / revision


def model_snapshot_is_complete(snapshot_path: Path) -> bool:
    """Return whether config, tokenizer, and every Safetensors shard are present."""

    required_files = (
        snapshot_path / "config.json",
        snapshot_path / "tokenizer_config.json",
    )
    if not snapshot_path.is_dir() or not all(path.is_file() for path in required_files):
        return False
    has_tokenizer = (snapshot_path / "tokenizer.json").is_file() or all(
        (snapshot_path / filename).is_file() for filename in ("vocab.json", "merges.txt")
    )
    if not has_tokenizer:
        return False
    single_weights = snapshot_path / "model.safetensors"
    if single_weights.is_file() and single_weights.stat().st_size > 0:
        return True
    index_path = snapshot_path / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        shard_names = set(weight_map.values())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return False
    return bool(shard_names) and all(
        isinstance(name, str)
        and name.endswith(".safetensors")
        and (snapshot_path / name).is_file()
        and (snapshot_path / name).stat().st_size > 0
        for name in shard_names
    )


def collect_runtime_capabilities(
    config: InferenceConfig,
    cache_dir: Path | None = None,
) -> RuntimeCapabilities:
    """Inspect installed packages and hardware without downloading model files."""

    torch_module, torch_error = _optional_import("torch")
    _transformers_module, transformers_error = _optional_import("transformers")
    _safetensors_module, safetensors_error = _optional_import("safetensors")
    psutil_module, psutil_error = _optional_import("psutil")
    import_errors = {
        name: error
        for name, error in {
            "torch": torch_error,
            "transformers": transformers_error,
            "safetensors": safetensors_error,
            "psutil": psutil_error,
        }.items()
        if error is not None
    }
    cuda_available = False
    cuda_bfloat16_supported = False
    cuda_device_name: str | None = None
    cuda_runtime_version: str | None = None
    cuda_vram_gb: float | None = None
    if torch_module is not None:
        cuda_runtime_version = getattr(torch_module.version, "cuda", None)
        cuda_available = bool(torch_module.cuda.is_available())
        cuda_bfloat16_supported = bool(cuda_available and torch_module.cuda.is_bf16_supported())
        if cuda_available:
            cuda_device_name = str(torch_module.cuda.get_device_name(0))
            total_memory = int(torch_module.cuda.get_device_properties(0).total_memory)
            cuda_vram_gb = total_memory / 1024**3

    system_ram_gb: float | None = None
    if psutil_module is not None:
        system_ram_gb = int(psutil_module.virtual_memory().total) / 1024**3

    snapshot_path = model_snapshot_path(
        config.model.model_id,
        config.model.revision,
        cache_dir,
    )
    return RuntimeCapabilities(
        python_version=platform.python_version(),
        torch_version=_package_version("torch"),
        transformers_version=_package_version("transformers"),
        safetensors_version=_package_version("safetensors"),
        psutil_version=_package_version("psutil"),
        import_errors=import_errors,
        cuda_available=cuda_available,
        cuda_bfloat16_supported=cuda_bfloat16_supported,
        cuda_device_name=cuda_device_name,
        cuda_runtime_version=cuda_runtime_version,
        cuda_vram_gb=cuda_vram_gb,
        system_ram_gb=system_ram_gb,
        model_cache_path=str(snapshot_path),
        model_cached=model_snapshot_is_complete(snapshot_path),
    )


def _select_device(
    preference: DevicePreference,
    capabilities: RuntimeCapabilities,
    errors: list[str],
) -> Literal["cpu", "cuda"]:
    if preference is DevicePreference.CPU:
        return "cpu"
    if preference is DevicePreference.CUDA:
        if not capabilities.cuda_available:
            errors.append("CUDA was requested but no CUDA device is available.")
        return "cuda"
    return "cuda" if capabilities.cuda_available else "cpu"


def _select_dtype(
    preference: DTypePreference,
    device: Literal["cpu", "cuda"],
    capabilities: RuntimeCapabilities,
    errors: list[str],
) -> Literal["float32", "float16", "bfloat16"]:
    if preference is DTypePreference.AUTO:
        if device == "cpu":
            return "float32"
        return "bfloat16" if capabilities.cuda_bfloat16_supported else "float16"
    selected = preference.value
    if device == "cpu" and selected != "float32":
        errors.append("CPU inference supports only float32 in this project configuration.")
    if selected == "bfloat16" and device == "cuda" and not capabilities.cuda_bfloat16_supported:
        errors.append("bfloat16 was requested but the CUDA device does not support it.")
    return selected  # type: ignore[return-value]


def select_hardware(
    config: InferenceConfig,
    capabilities: RuntimeCapabilities,
) -> HardwareReport:
    """Resolve configured preferences and return actionable readiness diagnostics."""

    warnings: list[str] = []
    errors: list[str] = []
    required_packages = {
        "torch": capabilities.torch_version,
        "transformers": capabilities.transformers_version,
        "safetensors": capabilities.safetensors_version,
    }
    missing_packages = [name for name, version in required_packages.items() if version is None]
    if missing_packages:
        errors.append(
            "Missing inference packages: "
            f"{', '.join(missing_packages)}. Install requirements-inference.lock."
        )
    failed_imports = {
        name: error
        for name, error in capabilities.import_errors.items()
        if name in required_packages
    }
    if failed_imports:
        details = "; ".join(f"{name}: {error}" for name, error in sorted(failed_imports.items()))
        errors.append(f"Inference packages are installed but failed to import: {details}")

    device = _select_device(config.model.device, capabilities, errors)
    dtype = _select_dtype(config.model.dtype, device, capabilities, errors)
    if device == "cuda" and (
        capabilities.cuda_vram_gb is not None and capabilities.cuda_vram_gb < MINIMUM_CUDA_VRAM_GB
    ):
        warnings.append(
            f"CUDA VRAM is {capabilities.cuda_vram_gb:.1f} GB; "
            f"at least {MINIMUM_CUDA_VRAM_GB:.1f} GB is recommended."
        )
    if device == "cpu" and (
        capabilities.system_ram_gb is not None and capabilities.system_ram_gb < MINIMUM_CPU_RAM_GB
    ):
        warnings.append(
            f"System RAM is {capabilities.system_ram_gb:.1f} GB; "
            f"at least {MINIMUM_CPU_RAM_GB:.1f} GB is recommended for CPU inference."
        )
    if capabilities.system_ram_gb is None:
        psutil_detail = capabilities.import_errors.get("psutil")
        suffix = f" ({psutil_detail})" if psutil_detail else ""
        warnings.append(
            "System RAM could not be detected; install or repair psutil for memory diagnostics."
            f"{suffix}"
        )
    if not capabilities.model_cached:
        cache_message = f"Model snapshot is not cached at {capabilities.model_cache_path}."
        if config.model.local_files_only:
            errors.append(f"{cache_message} Disable local_files_only to allow the first download.")
        else:
            warnings.append(f"{cache_message} The first inference run will download model files.")

    package_versions = {
        "torch": capabilities.torch_version,
        "transformers": capabilities.transformers_version,
        "safetensors": capabilities.safetensors_version,
        "psutil": capabilities.psutil_version,
    }
    return HardwareReport(
        ready=not errors,
        model_id=config.model.model_id,
        revision=config.model.revision,
        python_version=capabilities.python_version,
        package_versions=package_versions,
        selected_device=device,
        selected_dtype=dtype,
        cuda_available=capabilities.cuda_available,
        cuda_device_name=capabilities.cuda_device_name,
        cuda_runtime_version=capabilities.cuda_runtime_version,
        cuda_vram_gb=capabilities.cuda_vram_gb,
        system_ram_gb=capabilities.system_ram_gb,
        model_cache_path=capabilities.model_cache_path,
        model_cached=capabilities.model_cached,
        warnings=warnings,
        errors=errors,
    )


def detect_hardware(
    config: InferenceConfig,
    cache_dir: Path | None = None,
) -> HardwareReport:
    capabilities = collect_runtime_capabilities(config, cache_dir)
    return select_hardware(config, capabilities)
