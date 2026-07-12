from __future__ import annotations

from pathlib import Path

from small_models_society.training.config import (
    TrainingConfig,
    TrainingDevicePreference,
    TrainingDTypePreference,
    load_training_config,
)
from small_models_society.training.hardware import (
    REQUIRED_TRAINING_PACKAGES,
    TrainingRuntimeCapabilities,
    select_training_hardware,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "training.yaml"


def _config(
    device: TrainingDevicePreference = TrainingDevicePreference.AUTO,
    dtype: TrainingDTypePreference = TrainingDTypePreference.AUTO,
    local_files_only: bool = False,
) -> TrainingConfig:
    config = load_training_config(CONFIG_PATH)
    model = config.model.model_copy(
        update={
            "device": device,
            "dtype": dtype,
            "local_files_only": local_files_only,
        }
    )
    return config.model_copy(update={"model": model})


def _capabilities(**updates: object) -> TrainingRuntimeCapabilities:
    values: dict[str, object] = {
        "python_version": "3.11.9",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "package_versions": {
            "torch": "2.13.0",
            "transformers": "4.57.6",
            "safetensors": "0.8.0",
            "accelerate": "1.14.0",
            "peft": "0.19.1",
            "trl": "1.8.0",
            "psutil": "7.2.2",
        },
        "cuda_available": False,
        "cuda_bfloat16_supported": False,
        "cuda_device_name": None,
        "cuda_runtime_version": None,
        "cuda_vram_gb": None,
        "mps_built": False,
        "mps_available": False,
        "system_ram_gb": 16.0,
        "model_cache_path": "C:/cache/model",
        "model_cached": True,
        "artifact_root": "C:/artifacts/adapters",
        "artifact_root_writable": True,
        "mps_fallback_enabled": False,
    }
    values.update(updates)
    return TrainingRuntimeCapabilities.model_validate(values)


def test_auto_prefers_cuda_bfloat16_when_supported() -> None:
    report = select_training_hardware(
        _config(),
        _capabilities(
            cuda_available=True,
            cuda_bfloat16_supported=True,
            cuda_device_name="Test GPU",
            cuda_vram_gb=16.0,
            mps_built=True,
            mps_available=True,
        ),
    )

    assert report.ready is True
    assert report.selected_device == "cuda"
    assert report.selected_dtype == "bfloat16"
    assert report.estimated_fit == "likely"


def test_auto_selects_mps_float16_without_cuda() -> None:
    report = select_training_hardware(
        _config(),
        _capabilities(
            platform_system="Darwin",
            platform_machine="arm64",
            mps_built=True,
            mps_available=True,
            system_ram_gb=16.0,
        ),
    )

    assert report.ready is True
    assert report.selected_device == "mps"
    assert report.selected_dtype == "float16"
    assert report.estimated_fit == "likely"
    assert any("MPS backend is beta" in warning for warning in report.warnings)


def test_explicit_unavailable_accelerators_are_errors() -> None:
    cuda = select_training_hardware(
        _config(device=TrainingDevicePreference.CUDA),
        _capabilities(),
    )
    mps = select_training_hardware(
        _config(device=TrainingDevicePreference.MPS),
        _capabilities(),
    )

    assert cuda.ready is False
    assert any("CUDA training was requested" in error for error in cuda.errors)
    assert mps.ready is False
    assert any("MPS training was requested" in error for error in mps.errors)


def test_cpu_requires_explicit_debug_override() -> None:
    blocked = select_training_hardware(_config(), _capabilities())
    allowed = select_training_hardware(_config(), _capabilities(), allow_cpu=True)

    assert blocked.ready is False
    assert blocked.cpu_debug_only is True
    assert allowed.ready is True
    assert allowed.selected_device == "cpu"
    assert allowed.selected_dtype == "float32"
    assert allowed.estimated_fit == "cpu_debug_only"


def test_low_mps_memory_is_an_error() -> None:
    report = select_training_hardware(
        _config(),
        _capabilities(
            platform_system="Darwin",
            platform_machine="arm64",
            mps_built=True,
            mps_available=True,
            system_ram_gb=8.0,
        ),
    )

    assert report.ready is False
    assert report.estimated_fit == "insufficient"
    assert any("Apple unified memory is 8.0 GB" in error for error in report.errors)


def test_tight_accelerator_memory_recommends_explicit_fallback() -> None:
    report = select_training_hardware(
        _config(),
        _capabilities(cuda_available=True, cuda_vram_gb=14.0),
    )

    assert report.ready is True
    assert report.estimated_fit == "tight"
    assert any("384-token fallback" in warning for warning in report.warnings)


def test_missing_or_broken_training_packages_are_errors() -> None:
    versions = _capabilities().package_versions.copy()
    versions["peft"] = None
    missing = select_training_hardware(
        _config(),
        _capabilities(package_versions=versions),
        allow_cpu=True,
    )
    broken = select_training_hardware(
        _config(),
        _capabilities(import_errors={"trl": "OSError: missing shared library"}),
        allow_cpu=True,
    )

    assert missing.ready is False
    assert "requirements-training.lock" in missing.errors[0]
    assert broken.ready is False
    assert any("missing shared library" in error for error in broken.errors)
    assert set(REQUIRED_TRAINING_PACKAGES) <= set(missing.package_versions)


def test_mps_fallback_environment_is_reported_not_enabled() -> None:
    report = select_training_hardware(
        _config(),
        _capabilities(
            platform_system="Darwin",
            platform_machine="arm64",
            mps_built=True,
            mps_available=True,
            system_ram_gb=16.0,
            mps_fallback_enabled=True,
        ),
    )

    assert report.mps_fallback_enabled is True
    assert any("unsupported operations may run on CPU" in warning for warning in report.warnings)


def test_unwritable_artifact_root_is_an_error() -> None:
    report = select_training_hardware(
        _config(),
        _capabilities(
            cuda_available=True,
            cuda_vram_gb=16.0,
            artifact_root_writable=False,
            artifact_error="access denied",
        ),
    )

    assert report.ready is False
    assert any("access denied" in error for error in report.errors)


def test_uncached_offline_model_is_an_error() -> None:
    report = select_training_hardware(
        _config(local_files_only=True),
        _capabilities(cuda_available=True, cuda_vram_gb=16.0, model_cached=False),
    )

    assert report.ready is False
    assert any("local_files_only" in error for error in report.errors)


def test_unsupported_explicit_dtype_is_an_error() -> None:
    report = select_training_hardware(
        _config(
            device=TrainingDevicePreference.MPS,
            dtype=TrainingDTypePreference.BFLOAT16,
        ),
        _capabilities(
            platform_system="Darwin",
            platform_machine="arm64",
            mps_built=True,
            mps_available=True,
            system_ram_gb=16.0,
        ),
    )

    assert report.ready is False
    assert any("does not use bfloat16" in error for error in report.errors)
