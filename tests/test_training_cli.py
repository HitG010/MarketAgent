from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from small_models_society.cli import main
from small_models_society.schemas import Domain
from small_models_society.training.hardware import TrainingHardwareReport

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "training.yaml"


def _hardware(ready: bool = True) -> TrainingHardwareReport:
    return TrainingHardwareReport(
        ready=ready,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        python_version="3.11.9",
        platform_system="Windows",
        platform_machine="AMD64",
        package_versions={"torch": "test", "peft": "test", "trl": "test"},
        selected_device="cpu",
        selected_dtype="float32",
        estimated_fit="cpu_debug_only",
        cpu_debug_only=True,
        cuda_available=False,
        mps_built=False,
        mps_available=False,
        mps_fallback_enabled=False,
        system_ram_gb=32,
        model_cache_path="C:/cache/model",
        model_cached=True,
        artifact_root="C:/artifacts",
        artifact_root_writable=True,
        errors=[] if ready else ["missing accelerator"],
    )


def test_training_doctor_prints_readiness(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "small_models_society.cli.detect_training_hardware",
        lambda _config, *, allow_cpu: _hardware(allow_cpu),
    )

    exit_code = main(["training", "doctor", "--config", str(CONFIG_PATH), "--allow-cpu"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["selected_device"] == "cpu"
    assert report["ready"] is True


def test_prepare_preflights_collision_before_loading_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "training"
    output_dir.mkdir()
    (output_dir / "train.jsonl").write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        "small_models_society.cli.load_training_modules",
        lambda: pytest.fail("tokenizer stack should not load"),
    )

    exit_code = main(
        [
            "training",
            "prepare",
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 1
    assert "use overwrite explicitly" in capsys.readouterr().err


def test_prepare_prints_source_and_sft_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "training"
    tokenizer = object()
    tokenizer_factory = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(
        "small_models_society.cli.load_training_modules",
        lambda: SimpleNamespace(auto_tokenizer=tokenizer_factory),
    )
    monkeypatch.setattr(
        "small_models_society.cli.build_sft_eligibility_filter",
        lambda _catalog, received, _max_length: lambda _example: received is tokenizer,
    )
    source = SimpleNamespace(
        train_path=output_dir / "train.jsonl",
        validation_path=output_dir / "validation.jsonl",
        manifest_path=output_dir / "manifest.json",
    )
    sft = SimpleNamespace(
        train_path=output_dir / "sft" / "train.jsonl",
        validation_path=output_dir / "sft" / "validation.jsonl",
        manifest_path=output_dir / "sft" / "manifest.json",
        train_row_count=384,
        validation_row_count=96,
        train_sha256="a" * 64,
        validation_sha256="b" * 64,
    )
    monkeypatch.setattr(
        "small_models_society.cli.prepare_training_data",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        "small_models_society.cli.prepare_sft_data",
        lambda *_args, **_kwargs: sft,
    )

    exit_code = main(
        [
            "training",
            "prepare",
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(output_dir),
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["train_row_count"] == 384
    assert summary["validation_row_count"] == 96
    assert summary["train_sha256"] == "a" * 64


def _install_training_run_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ready: bool = True,
    pending: bool = True,
) -> list[object]:
    backends: list[object] = []
    monkeypatch.setattr(
        "small_models_society.cli.detect_training_hardware",
        lambda _config, *, allow_cpu: _hardware(ready),
    )
    plan = SimpleNamespace(
        pending=pending,
        train_row_count=96,
        validation_row_count=24,
    )
    monkeypatch.setattr(
        "small_models_society.cli.inspect_adapter_training",
        lambda *_args, **_kwargs: plan,
    )

    def create_backend(_config: object, _hardware_report: object) -> object:
        backend = object()
        backends.append(backend)
        return backend

    monkeypatch.setattr("small_models_society.cli.LoraTrainerBackend", create_backend)
    manifest = SimpleNamespace(
        adapter_sha256="c" * 64,
        duration_seconds=2.5,
        eval_metrics={"eval_loss": 0.5},
        run_fingerprint="d" * 64,
        status="completed",
        train_metrics={"train_loss": 1.0},
        trainable_parameters=16,
    )

    def run(*args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            adapter_dir=Path("artifacts/adapters/math"),
            manifest_path=Path("artifacts/adapters/math/manifest.json"),
            manifest=manifest,
            backend=args[5],
        )

    monkeypatch.setattr("small_models_society.cli.run_adapter_training", run)
    return backends


def _train_arguments(tmp_path: Path) -> list[str]:
    return [
        "training",
        "train",
        "--config",
        str(CONFIG_PATH),
        "--specialist",
        "math",
        "--adapter-root",
        str(tmp_path / "adapters"),
        "--allow-cpu",
    ]


def test_train_preflights_then_loads_one_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backends = _install_training_run_fakes(monkeypatch)

    exit_code = main(_train_arguments(tmp_path))
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert len(backends) == 1
    assert summary["specialist"] == "math"
    assert summary["adapter_sha256"] == "c" * 64
    assert "training plan:" in captured.err


def test_completed_resume_skips_backend_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backends = _install_training_run_fakes(monkeypatch, pending=False)

    assert main([*_train_arguments(tmp_path), "--resume"]) == 0
    capsys.readouterr()

    assert backends == []


def test_unready_training_environment_skips_inspection_and_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backends = _install_training_run_fakes(monkeypatch, ready=False)
    monkeypatch.setattr(
        "small_models_society.cli.inspect_adapter_training",
        lambda *_args, **_kwargs: pytest.fail("inspection should not run"),
    )

    exit_code = main(_train_arguments(tmp_path))

    assert exit_code == 1
    assert "training prerequisites are not ready" in capsys.readouterr().err
    assert backends == []


def test_train_all_launches_four_sequential_specialist_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("small_models_society.cli.subprocess.run", run)

    exit_code = main(
        [
            "training",
            "train-all",
            "--config",
            str(CONFIG_PATH),
            "--adapter-root",
            str(tmp_path / "adapters"),
            "--local-files-only",
            "--allow-cpu",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(calls) == 4
    assert [call[call.index("--specialist") + 1] for call in calls] == [
        domain.value for domain in Domain
    ]
    assert all("--local-files-only" in call and "--allow-cpu" in call for call in calls)
    assert summary["completed_specialists"] == [domain.value for domain in Domain]
