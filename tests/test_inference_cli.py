from __future__ import annotations

import json
from pathlib import Path

import pytest

from small_models_society.cli import main
from small_models_society.evaluation import load_predictions
from small_models_society.inference.config import InferenceConfig
from small_models_society.inference.contracts import GenerationOutput, GenerationRequest
from small_models_society.inference.hardware import HardwareReport
from small_models_society.schemas import Domain

FIXTURE_BENCHMARK = Path(__file__).parent / "fixtures" / "benchmark.jsonl"


class FakeBackend:
    def __init__(self, config: InferenceConfig, hardware: HardwareReport) -> None:
        self.config = config
        self.hardware = hardware
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.requests.append(request)
        responses = {
            Domain.MATH: "10",
            Domain.CODE: "def add(a, b):\n    return a + b",
            Domain.LOGIC: "A",
            Domain.KNOWLEDGE: "Paris",
        }
        return GenerationOutput(
            text=responses[request.example.domain],
            prompt_tokens=10,
            completion_tokens=2,
            latency_ms=4,
            metadata={"backend": "fake"},
        )


def _hardware(ready: bool = True) -> HardwareReport:
    return HardwareReport(
        ready=ready,
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
        errors=[] if ready else ["missing inference packages"],
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    ready: bool = True,
) -> list[FakeBackend]:
    backends: list[FakeBackend] = []
    monkeypatch.setattr(
        "small_models_society.cli.detect_hardware", lambda _config: _hardware(ready)
    )

    def create_backend(config: InferenceConfig, hardware: HardwareReport) -> FakeBackend:
        backend = FakeBackend(config, hardware)
        backends.append(backend)
        return backend

    monkeypatch.setattr("small_models_society.cli.HuggingFaceBackend", create_backend)
    return backends


def _predict_arguments(output: Path) -> list[str]:
    return [
        "inference",
        "predict",
        "--benchmark",
        str(FIXTURE_BENCHMARK),
        "--output",
        str(output),
    ]


def test_predict_command_filters_domain_and_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backends = _install_fakes(monkeypatch)
    output = tmp_path / "predictions.jsonl"

    exit_code = main(
        [
            *_predict_arguments(output),
            "--profile",
            "math",
            "--domain",
            "math",
            "--limit",
            "1",
        ]
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["prediction_count"] == 1
    assert summary["status_counts"] == {"ok": 1}
    assert summary["prompt_tokens"] == 10
    assert summary["completion_tokens"] == 2
    assert summary["mean_latency_ms"] == 4
    assert summary["profile"] == "math"
    assert "inference plan:" in captured.err
    assert len(backends) == 1
    assert len(backends[0].requests) == 1
    request_json = backends[0].requests[0].model_dump_json()
    assert '"reference"' not in request_json
    assert load_predictions(output)[0].domain is Domain.MATH


def test_predict_refuses_existing_output_without_explicit_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backends = _install_fakes(monkeypatch)
    output = tmp_path / "predictions.jsonl"
    assert main(_predict_arguments(output)) == 0
    capsys.readouterr()

    exit_code = main(_predict_arguments(output))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "resume or overwrite" in captured.err
    assert len(backends) == 1


def test_predict_resume_skips_completed_examples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backends = _install_fakes(monkeypatch)
    output = tmp_path / "predictions.jsonl"
    assert main(_predict_arguments(output)) == 0
    first_bytes = output.read_bytes()
    capsys.readouterr()

    assert main([*_predict_arguments(output), "--resume"]) == 0
    capsys.readouterr()

    assert output.read_bytes() == first_bytes
    assert len(backends) == 1


def test_local_files_only_override_reaches_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backends = _install_fakes(monkeypatch)

    assert main([*_predict_arguments(tmp_path / "predictions.jsonl"), "--local-files-only"]) == 0
    capsys.readouterr()

    assert backends[0].config.model.local_files_only is True


def test_missing_inference_dependencies_return_nonzero_without_loading_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backends = _install_fakes(monkeypatch, ready=False)

    exit_code = main(_predict_arguments(tmp_path / "predictions.jsonl"))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "inference prerequisites are not ready" in captured.err
    assert backends == []


def test_invalid_profile_is_rejected_by_argument_parser(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        main([*_predict_arguments(tmp_path / "predictions.jsonl"), "--profile", "unknown"])

    assert error.value.code == 2
