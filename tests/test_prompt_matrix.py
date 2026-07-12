from __future__ import annotations

import json
from pathlib import Path

import pytest

from small_models_society.cli import main
from small_models_society.experiments.prompt_matrix import (
    PromptMatrixOptions,
    run_prompt_matrix,
)
from small_models_society.inference.config import InferenceConfig, load_inference_config
from small_models_society.inference.contracts import GenerationOutput, GenerationRequest
from small_models_society.inference.hardware import HardwareReport
from small_models_society.inference.prompts import PromptProfileName, load_prompt_catalog
from small_models_society.inference.runner import acquire_run_lock
from small_models_society.sandbox import SandboxResult, SandboxStatus
from small_models_society.schemas import Domain

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "inference.yaml"
PROMPT_CONFIG = Path(__file__).parents[1] / "configs" / "prompt_profiles.yaml"
FIXTURE_BENCHMARK = Path(__file__).parent / "fixtures" / "benchmark.jsonl"


class ProfileAwareBackend:
    def __init__(
        self,
        config: InferenceConfig | None = None,
        hardware: HardwareReport | None = None,
    ) -> None:
        self.config = config
        self.hardware = hardware
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.requests.append(request)
        is_own_specialty = request.profile == request.example.domain.value
        correct = {
            Domain.MATH: "10",
            Domain.CODE: "def add(a, b):\n    return a + b",
            Domain.LOGIC: "A",
            Domain.KNOWLEDGE: "Paris",
        }
        wrong = {
            Domain.MATH: "11",
            Domain.CODE: "def add(a, b):\n    return 0",
            Domain.LOGIC: "B",
            Domain.KNOWLEDGE: "London",
        }
        return GenerationOutput(
            text=(correct if is_own_specialty else wrong)[request.example.domain],
            prompt_tokens=10,
            completion_tokens=2,
            latency_ms=5,
            metadata={"backend": "profile-aware-fake"},
        )


class FakeSandbox:
    def run(self, candidate: str, test_setup: str, tests: list[str]) -> SandboxResult:
        del test_setup, tests
        passed = "return a + b" in candidate
        return SandboxResult(
            status=SandboxStatus.PASSED if passed else SandboxStatus.ASSERTION_FAILURE,
            duration_ms=1,
        )


class InfrastructureErrorSandbox:
    def run(self, candidate: str, test_setup: str, tests: list[str]) -> SandboxResult:
        del candidate, test_setup, tests
        return SandboxResult(
            status=SandboxStatus.INFRASTRUCTURE_ERROR,
            duration_ms=1,
            stderr="sandbox image is unavailable",
        )


class InterruptingBackend(ProfileAwareBackend):
    def generate(self, request: GenerationRequest) -> GenerationOutput:
        if len(self.requests) == 4:
            raise KeyboardInterrupt
        return super().generate(request)


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


def test_prompt_matrix_builds_full_profile_domain_experiment(tmp_path: Path) -> None:
    backend = ProfileAwareBackend()
    output_dir = tmp_path / "matrix"

    result = run_prompt_matrix(
        FIXTURE_BENCHMARK,
        output_dir,
        load_inference_config(CONFIG_PATH),
        load_prompt_catalog(PROMPT_CONFIG),
        _hardware(),
        backend,
        FakeSandbox(),
    )

    assert len(backend.requests) == 20
    assert {(request.profile, request.example.domain.value) for request in backend.requests} == {
        (profile.value, domain.value) for profile in PromptProfileName for domain in Domain
    }
    matrix = result.summary["profile_by_domain"]
    assert matrix["general"] == {
        "math": 0.0,
        "code": 0.0,
        "logic": 0.0,
        "knowledge": 0.0,
    }
    for profile in (
        PromptProfileName.MATH,
        PromptProfileName.CODE,
        PromptProfileName.LOGIC,
        PromptProfileName.KNOWLEDGE,
    ):
        assert matrix[profile.value][profile.value] == 1.0
        assert result.summary["specialist_effects"][profile.value]["own_domain_lift"] == 1.0
    oracle = result.summary["prompt_profile_oracle"]
    assert oracle == {
        "example_count": 4,
        "general_score": 0.0,
        "oracle_score": 1.0,
        "routing_opportunity": 1.0,
    }
    result_lines = result.results_path.read_text(encoding="utf-8").splitlines()
    assert len(result_lines) == 20
    assert '"reference"' not in result.results_path.read_text(encoding="utf-8")
    assert '"reference"' not in result.summary_path.read_text(encoding="utf-8")
    assert "not trained specialists" in result.report_path.read_text(encoding="utf-8")
    for profile in PromptProfileName:
        assert result.prediction_paths[profile].exists()
        assert (output_dir / profile.value / "evaluation" / "summary.json").exists()


def test_prompt_matrix_resume_skips_all_completed_generation(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    config = load_inference_config(CONFIG_PATH)
    catalog = load_prompt_catalog(PROMPT_CONFIG)
    first_backend = ProfileAwareBackend()
    first = run_prompt_matrix(
        FIXTURE_BENCHMARK,
        output_dir,
        config,
        catalog,
        _hardware(),
        first_backend,
        FakeSandbox(),
    )
    first_summary = first.summary_path.read_bytes()
    second_backend = ProfileAwareBackend()

    second = run_prompt_matrix(
        FIXTURE_BENCHMARK,
        output_dir,
        config,
        catalog,
        _hardware(),
        second_backend,
        FakeSandbox(),
        PromptMatrixOptions(resume=True),
    )

    assert len(first_backend.requests) == 20
    assert second_backend.requests == []
    assert second.summary_path.read_bytes() == first_summary


def test_prompt_matrix_resume_starts_profiles_that_have_no_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    config = load_inference_config(CONFIG_PATH)
    catalog = load_prompt_catalog(PROMPT_CONFIG)

    with pytest.raises(KeyboardInterrupt):
        run_prompt_matrix(
            FIXTURE_BENCHMARK,
            output_dir,
            config,
            catalog,
            _hardware(),
            InterruptingBackend(),
            FakeSandbox(),
        )

    resumed_backend = ProfileAwareBackend()
    result = run_prompt_matrix(
        FIXTURE_BENCHMARK,
        output_dir,
        config,
        catalog,
        _hardware(),
        resumed_backend,
        FakeSandbox(),
        PromptMatrixOptions(resume=True),
    )

    assert len(resumed_backend.requests) == 16
    assert result.summary["prompt_profile_oracle"]["example_count"] == 4


def test_prompt_matrix_rejects_sandbox_infrastructure_scores(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="sandbox infrastructure error"):
        run_prompt_matrix(
            FIXTURE_BENCHMARK,
            tmp_path / "matrix",
            load_inference_config(CONFIG_PATH),
            load_prompt_catalog(PROMPT_CONFIG),
            _hardware(),
            ProfileAwareBackend(),
            InfrastructureErrorSandbox(),
        )


def test_filtered_matrix_marks_off_domain_effect_unmeasured(tmp_path: Path) -> None:
    result = run_prompt_matrix(
        FIXTURE_BENCHMARK,
        tmp_path / "matrix",
        load_inference_config(CONFIG_PATH),
        load_prompt_catalog(PROMPT_CONFIG),
        _hardware(),
        ProfileAwareBackend(),
        FakeSandbox(),
        PromptMatrixOptions(domains=[Domain.MATH]),
    )

    math_effect = result.summary["specialist_effects"]["math"]
    assert math_effect["off_domain_delta"] is None
    assert math_effect["off_domain_degradation"] is None
    assert "| math | +1.000 | - | - |" in result.report_path.read_text(encoding="utf-8")


def test_prompt_matrix_requires_explicit_collision_policy(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()
    (output_dir / "specialization_summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="resume or overwrite"):
        run_prompt_matrix(
            FIXTURE_BENCHMARK,
            output_dir,
            load_inference_config(CONFIG_PATH),
            load_prompt_catalog(PROMPT_CONFIG),
            _hardware(),
            ProfileAwareBackend(),
            FakeSandbox(),
        )


def test_prompt_matrix_rejects_concurrent_writer(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    lock_target = output_dir / "profile_results.jsonl"

    with acquire_run_lock(lock_target), pytest.raises(FileExistsError, match="another process"):
        run_prompt_matrix(
            FIXTURE_BENCHMARK,
            output_dir,
            load_inference_config(CONFIG_PATH),
            load_prompt_catalog(PROMPT_CONFIG),
            _hardware(),
            ProfileAwareBackend(),
            FakeSandbox(),
        )


def test_prompt_matrix_cli_loads_one_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backends: list[ProfileAwareBackend] = []
    monkeypatch.setattr("small_models_society.cli.detect_hardware", lambda _config: _hardware())
    monkeypatch.setattr("small_models_society.cli.docker_available", lambda: True)
    monkeypatch.setattr("small_models_society.cli.sandbox_image_available", lambda _image: True)
    monkeypatch.setattr("small_models_society.cli.DockerSandbox", lambda **_kwargs: FakeSandbox())

    def create_backend(config: InferenceConfig, hardware: HardwareReport) -> ProfileAwareBackend:
        backend = ProfileAwareBackend(config, hardware)
        backends.append(backend)
        return backend

    monkeypatch.setattr("small_models_society.cli.HuggingFaceBackend", create_backend)

    exit_code = main(
        [
            "experiment",
            "prompt-matrix",
            "--benchmark",
            str(FIXTURE_BENCHMARK),
            "--output-dir",
            str(tmp_path / "matrix"),
        ]
    )
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert exit_code == 0
    assert len(backends) == 1
    assert len(backends[0].requests) == 20
    assert output["oracle_score"] == 1.0
    assert output["routing_opportunity"] == 1.0
    assert "prompt matrix plan:" in captured.err


def test_prompt_matrix_cli_rejects_missing_sandbox_image_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backends: list[ProfileAwareBackend] = []
    monkeypatch.setattr("small_models_society.cli.docker_available", lambda: True)
    monkeypatch.setattr("small_models_society.cli.sandbox_image_available", lambda _image: False)

    def create_backend(config: InferenceConfig, hardware: HardwareReport) -> ProfileAwareBackend:
        backend = ProfileAwareBackend(config, hardware)
        backends.append(backend)
        return backend

    monkeypatch.setattr("small_models_society.cli.HuggingFaceBackend", create_backend)

    exit_code = main(
        [
            "experiment",
            "prompt-matrix",
            "--benchmark",
            str(FIXTURE_BENCHMARK),
            "--output-dir",
            str(tmp_path / "matrix"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "sandbox image is unavailable" in captured.err
    assert backends == []


def test_completed_matrix_cli_resume_loads_no_second_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backends: list[ProfileAwareBackend] = []
    monkeypatch.setattr("small_models_society.cli.detect_hardware", lambda _config: _hardware())
    monkeypatch.setattr("small_models_society.cli.docker_available", lambda: True)
    monkeypatch.setattr("small_models_society.cli.sandbox_image_available", lambda _image: True)
    monkeypatch.setattr("small_models_society.cli.DockerSandbox", lambda **_kwargs: FakeSandbox())

    def create_backend(config: InferenceConfig, hardware: HardwareReport) -> ProfileAwareBackend:
        backend = ProfileAwareBackend(config, hardware)
        backends.append(backend)
        return backend

    monkeypatch.setattr("small_models_society.cli.HuggingFaceBackend", create_backend)
    arguments = [
        "experiment",
        "prompt-matrix",
        "--benchmark",
        str(FIXTURE_BENCHMARK),
        "--output-dir",
        str(tmp_path / "matrix"),
    ]
    assert main(arguments) == 0
    capsys.readouterr()

    assert main([*arguments, "--resume"]) == 0
    capsys.readouterr()

    assert len(backends) == 1
