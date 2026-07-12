from __future__ import annotations

import json
from pathlib import Path

import pytest

from small_models_society.cli import main
from small_models_society.experiments.lora_matrix import (
    BASE_VARIANT,
    LoraMatrixOptions,
    inspect_lora_matrix,
    run_lora_matrix,
)
from small_models_society.inference.adapters import AdapterCatalog, AdapterSpec
from small_models_society.inference.config import load_inference_config
from small_models_society.inference.contracts import GenerationOutput, GenerationRequest
from small_models_society.inference.hardware import HardwareReport
from small_models_society.inference.prompts import load_prompt_catalog
from small_models_society.inference.runner import acquire_run_lock
from small_models_society.sandbox import SandboxResult, SandboxStatus
from small_models_society.schemas import Domain

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "inference.yaml"
PROMPT_CONFIG = ROOT / "configs" / "prompt_profiles.yaml"
FIXTURE_BENCHMARK = Path(__file__).parent / "fixtures" / "benchmark.jsonl"


class AdapterAwareBackend:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.requests.append(request)
        is_own_specialty = request.adapter == request.example.domain.value
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
        )


class FakeSandbox:
    def run(self, candidate: str, test_setup: str, tests: list[str]) -> SandboxResult:
        del test_setup, tests
        return SandboxResult(
            status=(
                SandboxStatus.PASSED
                if "return a + b" in candidate
                else SandboxStatus.ASSERTION_FAILURE
            ),
            duration_ms=1,
        )


class InfrastructureErrorSandbox:
    def run(self, candidate: str, test_setup: str, tests: list[str]) -> SandboxResult:
        del candidate, test_setup, tests
        return SandboxResult(
            status=SandboxStatus.INFRASTRUCTURE_ERROR,
            duration_ms=1,
            stderr="sandbox unavailable",
        )


def _hardware() -> HardwareReport:
    return HardwareReport(
        ready=True,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        python_version="3.11.9",
        package_versions={"torch": "test", "peft": "test"},
        selected_device="cpu",
        selected_dtype="float32",
        cuda_available=False,
        system_ram_gb=16,
        model_cache_path="C:/cache/model",
        model_cached=True,
    )


def _adapters(tmp_path: Path) -> AdapterCatalog:
    return AdapterCatalog(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        model_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        adapters={
            domain: AdapterSpec(
                name=domain,
                path=tmp_path / domain.value,
                sha256=(str(index + 1) * 64),
                run_fingerprint=(str(index + 5) * 64),
            )
            for index, domain in enumerate(Domain)
        },
    )


def test_lora_matrix_isolates_weights_under_fixed_general_prompt(tmp_path: Path) -> None:
    backend = AdapterAwareBackend()
    adapters = _adapters(tmp_path / "adapters")
    result = run_lora_matrix(
        FIXTURE_BENCHMARK,
        tmp_path / "matrix",
        load_inference_config(CONFIG_PATH),
        load_prompt_catalog(PROMPT_CONFIG),
        _hardware(),
        adapters,
        backend,
        FakeSandbox(),
    )

    assert len(backend.requests) == 20
    assert {request.profile for request in backend.requests} == {"general"}
    assert {request.adapter for request in backend.requests} == {
        None,
        *[domain.value for domain in Domain],
    }
    matrix = result.summary["adapter_by_domain"]
    assert matrix[BASE_VARIANT] == {
        "math": 0.0,
        "code": 0.0,
        "logic": 0.0,
        "knowledge": 0.0,
    }
    for domain in Domain:
        assert matrix[domain.value][domain.value] == 1.0
        assert result.summary["specialist_effects"][domain.value]["own_domain_lift"] == 1.0
    aggregate = result.summary["aggregate_differentiation"]
    assert aggregate["positive_own_domain_lift_count"] == 4
    assert aggregate["mean_own_domain_lift"] == 1.0
    assert result.summary["adapter_oracle"] == {
        "example_count": 4,
        "base_score": 0.0,
        "oracle_score": 1.0,
        "routing_opportunity": 1.0,
    }
    assert len(result.results_path.read_text(encoding="utf-8").splitlines()) == 20
    report = result.report_path.read_text(encoding="utf-8")
    assert "same fixed general prompt" in report
    assert "does not emit router labels" in report


def test_completed_matrix_resume_requires_no_backend(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    config = load_inference_config(CONFIG_PATH)
    prompts = load_prompt_catalog(PROMPT_CONFIG)
    adapters = _adapters(tmp_path / "adapters")
    first_backend = AdapterAwareBackend()
    first = run_lora_matrix(
        FIXTURE_BENCHMARK,
        output_dir,
        config,
        prompts,
        _hardware(),
        adapters,
        first_backend,
        FakeSandbox(),
    )
    summary_bytes = first.summary_path.read_bytes()

    plan = inspect_lora_matrix(
        FIXTURE_BENCHMARK,
        output_dir,
        config,
        prompts,
        _hardware(),
        adapters,
        LoraMatrixOptions(resume=True),
    )
    resumed = run_lora_matrix(
        FIXTURE_BENCHMARK,
        output_dir,
        config,
        prompts,
        _hardware(),
        adapters,
        None,
        FakeSandbox(),
        LoraMatrixOptions(resume=True),
    )

    assert plan.pending_generation_count == 0
    assert resumed.summary_path.read_bytes() == summary_bytes
    assert len(first_backend.requests) == 20


def test_partial_resume_runs_only_missing_weight_variants(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    config = load_inference_config(CONFIG_PATH)
    prompts = load_prompt_catalog(PROMPT_CONFIG)
    adapters = _adapters(tmp_path / "adapters")
    interrupting = AdapterAwareBackend()

    class InterruptingBackend(AdapterAwareBackend):
        def generate(self, request: GenerationRequest) -> GenerationOutput:
            if len(self.requests) == 4:
                raise KeyboardInterrupt
            return super().generate(request)

    with pytest.raises(KeyboardInterrupt):
        run_lora_matrix(
            FIXTURE_BENCHMARK,
            output_dir,
            config,
            prompts,
            _hardware(),
            adapters,
            InterruptingBackend(),
            FakeSandbox(),
        )

    resumed = run_lora_matrix(
        FIXTURE_BENCHMARK,
        output_dir,
        config,
        prompts,
        _hardware(),
        adapters,
        interrupting,
        FakeSandbox(),
        LoraMatrixOptions(resume=True),
    )

    assert len(interrupting.requests) == 16
    assert resumed.summary["adapter_oracle"]["example_count"] == 4


def test_filtered_matrix_marks_off_domain_unmeasured(tmp_path: Path) -> None:
    result = run_lora_matrix(
        FIXTURE_BENCHMARK,
        tmp_path / "matrix",
        load_inference_config(CONFIG_PATH),
        load_prompt_catalog(PROMPT_CONFIG),
        _hardware(),
        _adapters(tmp_path / "adapters"),
        AdapterAwareBackend(),
        FakeSandbox(),
        LoraMatrixOptions(domains=[Domain.MATH]),
    )

    effect = result.summary["specialist_effects"]["math"]
    assert effect["off_domain_delta"] is None
    assert effect["off_domain_degradation"] is None


def test_prompt_matrix_comparator_is_aggregate_only(tmp_path: Path) -> None:
    prompt_summary = tmp_path / "prompt-summary.json"
    prompt_summary.write_text(
        json.dumps({"prompt_profile_oracle": {"routing_opportunity": 0.25}}),
        encoding="utf-8",
    )
    result = run_lora_matrix(
        FIXTURE_BENCHMARK,
        tmp_path / "matrix",
        load_inference_config(CONFIG_PATH),
        load_prompt_catalog(PROMPT_CONFIG),
        _hardware(),
        _adapters(tmp_path / "adapters"),
        AdapterAwareBackend(),
        FakeSandbox(),
        LoraMatrixOptions(prompt_summary_path=prompt_summary),
    )

    assert result.summary["prompt_matrix_comparator"] == {
        "prompt_routing_opportunity": 0.25,
        "lora_minus_prompt_opportunity": 0.75,
    }
    assert "oracle_labels" not in result.summary


def test_sandbox_infrastructure_error_aborts_matrix(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="sandbox infrastructure error"):
        run_lora_matrix(
            FIXTURE_BENCHMARK,
            tmp_path / "matrix",
            load_inference_config(CONFIG_PATH),
            load_prompt_catalog(PROMPT_CONFIG),
            _hardware(),
            _adapters(tmp_path / "adapters"),
            AdapterAwareBackend(),
            InfrastructureErrorSandbox(),
        )


def test_matrix_requires_explicit_collision_policy(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()
    (output_dir / "lora_specialization_summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="resume or overwrite"):
        inspect_lora_matrix(
            FIXTURE_BENCHMARK,
            output_dir,
            load_inference_config(CONFIG_PATH),
            load_prompt_catalog(PROMPT_CONFIG),
            _hardware(),
            _adapters(tmp_path / "adapters"),
        )


def test_matrix_rejects_concurrent_writer(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    lock_target = output_dir / "adapter_results.jsonl"

    with acquire_run_lock(lock_target), pytest.raises(FileExistsError, match="another process"):
        run_lora_matrix(
            FIXTURE_BENCHMARK,
            output_dir,
            load_inference_config(CONFIG_PATH),
            load_prompt_catalog(PROMPT_CONFIG),
            _hardware(),
            _adapters(tmp_path / "adapters"),
            AdapterAwareBackend(),
            FakeSandbox(),
        )


def test_lora_matrix_cli_preflights_and_loads_one_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapters = _adapters(tmp_path / "adapters")
    backends: list[AdapterAwareBackend] = []
    monkeypatch.setattr(
        "small_models_society.cli.load_adapter_catalog",
        lambda *_args: adapters,
    )
    monkeypatch.setattr("small_models_society.cli.detect_hardware", lambda _config: _hardware())
    monkeypatch.setattr("small_models_society.cli.docker_available", lambda: True)
    monkeypatch.setattr("small_models_society.cli.sandbox_image_available", lambda _image: True)
    monkeypatch.setattr("small_models_society.cli.DockerSandbox", lambda **_kwargs: FakeSandbox())

    def create_backend(*_args: object) -> AdapterAwareBackend:
        backend = AdapterAwareBackend()
        backends.append(backend)
        return backend

    monkeypatch.setattr("small_models_society.cli.PeftHuggingFaceBackend", create_backend)

    exit_code = main(
        [
            "experiment",
            "lora-matrix",
            "--benchmark",
            str(FIXTURE_BENCHMARK),
            "--output-dir",
            str(tmp_path / "matrix"),
            "--adapter-root",
            str(tmp_path / "adapters"),
        ]
    )
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert exit_code == 0
    assert len(backends) == 1
    assert len(backends[0].requests) == 20
    assert output["routing_opportunity"] == 1.0
    assert output["positive_own_domain_lift_count"] == 4
    assert "LoRA matrix plan:" in captured.err


def test_lora_matrix_cli_rejects_missing_sandbox_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "small_models_society.cli.load_adapter_catalog",
        lambda *_args: _adapters(tmp_path / "adapters"),
    )
    monkeypatch.setattr("small_models_society.cli.docker_available", lambda: True)
    monkeypatch.setattr("small_models_society.cli.sandbox_image_available", lambda _image: False)
    monkeypatch.setattr(
        "small_models_society.cli.PeftHuggingFaceBackend",
        lambda *_args: pytest.fail("backend should not load"),
    )

    exit_code = main(
        [
            "experiment",
            "lora-matrix",
            "--benchmark",
            str(FIXTURE_BENCHMARK),
            "--output-dir",
            str(tmp_path / "matrix"),
            "--adapter-root",
            str(tmp_path / "adapters"),
        ]
    )

    assert exit_code == 1
    assert "sandbox image is unavailable" in capsys.readouterr().err
