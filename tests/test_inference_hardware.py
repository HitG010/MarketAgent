from __future__ import annotations

import json
from pathlib import Path

import pytest

from small_models_society.cli import main
from small_models_society.inference.config import (
    DevicePreference,
    DTypePreference,
    InferenceConfig,
    load_inference_config,
)
from small_models_society.inference.hardware import (
    HardwareReport,
    RuntimeCapabilities,
    model_snapshot_is_complete,
    model_snapshot_path,
    select_hardware,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "inference.yaml"


def _config(
    device: DevicePreference = DevicePreference.AUTO,
    dtype: DTypePreference = DTypePreference.AUTO,
    local_files_only: bool = False,
) -> InferenceConfig:
    config = load_inference_config(CONFIG_PATH)
    model = config.model.model_copy(
        update={
            "device": device,
            "dtype": dtype,
            "local_files_only": local_files_only,
        }
    )
    return config.model_copy(update={"model": model})


def _capabilities(**updates: object) -> RuntimeCapabilities:
    values: dict[str, object] = {
        "python_version": "3.11.9",
        "torch_version": "2.13.0",
        "transformers_version": "4.57.6",
        "safetensors_version": "0.8.0",
        "psutil_version": "7.2.2",
        "cuda_available": False,
        "cuda_bfloat16_supported": False,
        "cuda_device_name": None,
        "cuda_runtime_version": None,
        "cuda_vram_gb": None,
        "system_ram_gb": 16.0,
        "model_cache_path": "C:/cache/model",
        "model_cached": True,
    }
    values.update(updates)
    return RuntimeCapabilities.model_validate(values)


def test_auto_selects_cpu_float32_without_cuda() -> None:
    report = select_hardware(_config(), _capabilities())

    assert report.ready is True
    assert report.selected_device == "cpu"
    assert report.selected_dtype == "float32"


def test_auto_selects_cuda_bfloat16_when_supported() -> None:
    report = select_hardware(
        _config(),
        _capabilities(
            cuda_available=True,
            cuda_bfloat16_supported=True,
            cuda_device_name="Test GPU",
            cuda_vram_gb=12.0,
        ),
    )

    assert report.ready is True
    assert report.selected_device == "cuda"
    assert report.selected_dtype == "bfloat16"
    assert report.cuda_device_name == "Test GPU"


def test_auto_selects_cuda_float16_without_bfloat16() -> None:
    report = select_hardware(
        _config(),
        _capabilities(cuda_available=True, cuda_vram_gb=8.0),
    )

    assert report.selected_device == "cuda"
    assert report.selected_dtype == "float16"


def test_auto_selects_mps_float16_without_cuda() -> None:
    report = select_hardware(
        _config(),
        _capabilities(mps_built=True, mps_available=True),
    )

    assert report.ready is True
    assert report.selected_device == "mps"
    assert report.selected_dtype == "float16"


def test_explicit_unavailable_mps_is_an_error() -> None:
    report = select_hardware(_config(device=DevicePreference.MPS), _capabilities())

    assert report.ready is False
    assert any("MPS was requested" in error for error in report.errors)


def test_missing_inference_packages_make_report_not_ready() -> None:
    report = select_hardware(
        _config(),
        _capabilities(
            torch_version=None,
            transformers_version=None,
            safetensors_version=None,
        ),
    )

    assert report.ready is False
    assert "Missing inference packages" in report.errors[0]


def test_installed_but_broken_inference_package_is_not_ready() -> None:
    report = select_hardware(
        _config(),
        _capabilities(import_errors={"torch": "OSError: missing CUDA DLL"}),
    )

    assert report.ready is False
    assert any("failed to import" in error for error in report.errors)
    assert any("missing CUDA DLL" in error for error in report.errors)


def test_explicit_unavailable_cuda_is_an_error() -> None:
    report = select_hardware(_config(device=DevicePreference.CUDA), _capabilities())

    assert report.ready is False
    assert any("CUDA was requested" in error for error in report.errors)


def test_explicit_unsupported_dtype_is_an_error() -> None:
    report = select_hardware(
        _config(device=DevicePreference.CUDA, dtype=DTypePreference.BFLOAT16),
        _capabilities(cuda_available=True, cuda_vram_gb=8.0),
    )

    assert report.ready is False
    assert any("does not support" in error for error in report.errors)


def test_low_memory_and_uncached_model_produce_warnings() -> None:
    report = select_hardware(
        _config(),
        _capabilities(system_ram_gb=6.0, model_cached=False),
    )

    assert report.ready is True
    assert any("System RAM" in warning for warning in report.warnings)
    assert any("first inference run" in warning for warning in report.warnings)


def test_uncached_offline_model_is_an_error() -> None:
    report = select_hardware(
        _config(local_files_only=True),
        _capabilities(model_cached=False),
    )

    assert report.ready is False
    assert any("local_files_only" in error for error in report.errors)


def test_model_snapshot_path_uses_explicit_cache(tmp_path: Path) -> None:
    snapshot = model_snapshot_path("org/model", "a" * 40, tmp_path)

    assert snapshot == tmp_path / "models--org--model" / "snapshots" / ("a" * 40)


def test_model_snapshot_path_honors_hf_hub_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "legacy"))

    snapshot = model_snapshot_path("org/model", "a" * 40)

    assert snapshot == tmp_path / "models--org--model" / "snapshots" / ("a" * 40)


def test_partial_snapshot_without_weights_is_not_cached(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")

    assert model_snapshot_is_complete(snapshot) is False


def _write_tokenizer_and_config(snapshot: Path) -> None:
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_complete_single_safetensors_snapshot_is_cached(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _write_tokenizer_and_config(snapshot)
    (snapshot / "model.safetensors").write_bytes(b"fixture")

    assert model_snapshot_is_complete(snapshot) is True


def test_bare_or_incomplete_safetensors_index_is_not_cached(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _write_tokenizer_and_config(snapshot)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00002.safetensors"}}),
        encoding="utf-8",
    )

    assert model_snapshot_is_complete(snapshot) is False


def test_complete_sharded_safetensors_snapshot_is_cached(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _write_tokenizer_and_config(snapshot)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"first": shards[0], "second": shards[1]}}),
        encoding="utf-8",
    )
    for shard in shards:
        (snapshot / shard).write_bytes(b"fixture")

    assert model_snapshot_is_complete(snapshot) is True


def test_inference_doctor_prints_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = HardwareReport(
        ready=True,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision="a" * 40,
        python_version="3.11.9",
        package_versions={"torch": "2.13.0"},
        selected_device="cpu",
        selected_dtype="float32",
        cuda_available=False,
        system_ram_gb=16,
        model_cache_path="C:/cache/model",
        model_cached=True,
    )
    monkeypatch.setattr("small_models_society.cli.detect_hardware", lambda _config: report)

    exit_code = main(["inference", "doctor", "--config", str(CONFIG_PATH)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["ready"] is True
    assert output["selected_device"] == "cpu"
