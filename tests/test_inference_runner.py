from __future__ import annotations

from pathlib import Path

import pytest

from small_models_society.evaluation import load_predictions, write_predictions
from small_models_society.inference.config import InferenceConfig, load_inference_config
from small_models_society.inference.contracts import (
    AdapterReference,
    GenerationOutput,
    GenerationRequest,
)
from small_models_society.inference.hardware import HardwareReport
from small_models_society.inference.huggingface import InferenceOutOfMemoryError
from small_models_society.inference.prompts import (
    PromptCatalog,
    PromptProfileName,
    load_prompt_catalog,
)
from small_models_society.inference.runner import (
    PredictionRunOptions,
    ResumeMismatchError,
    acquire_run_lock,
    inspect_prediction_run,
    manifest_path_for,
    run_predictions,
)
from small_models_society.schemas import Domain, PredictionStatus

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "inference.yaml"
PROMPT_CONFIG = Path(__file__).parents[1] / "configs" / "prompt_profiles.yaml"
FIXTURE_BENCHMARK = Path(__file__).parent / "fixtures" / "benchmark.jsonl"
DOMAIN_BY_REQUEST_ID = {
    "fixture-math-1": Domain.MATH,
    "fixture-code-1": Domain.CODE,
    "fixture-logic-1": Domain.LOGIC,
    "fixture-knowledge-1": Domain.KNOWLEDGE,
}


class FakeBackend:
    def __init__(
        self,
        failures: dict[str, BaseException] | None = None,
    ) -> None:
        self.failures = failures or {}
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.requests.append(request)
        failure = self.failures.get(request.request_id)
        if failure is not None:
            raise failure
        domain = DOMAIN_BY_REQUEST_ID[request.request_id]
        responses = {
            Domain.MATH: "10",
            Domain.CODE: "```python\ndef add(a, b):\n    return a + b\n```",
            Domain.LOGIC: "A",
            Domain.KNOWLEDGE: "Paris",
        }
        return GenerationOutput(
            text=responses[domain],
            prompt_tokens=10,
            completion_tokens=2,
            latency_ms=3.5,
            metadata={"backend": "fake"},
        )


def _config(checkpoint_interval: int = 5) -> InferenceConfig:
    config = load_inference_config(CONFIG_PATH)
    generation = config.generation.model_copy(update={"checkpoint_interval": checkpoint_interval})
    return config.model_copy(update={"generation": generation})


def _catalog() -> PromptCatalog:
    return load_prompt_catalog(PROMPT_CONFIG)


def _hardware() -> HardwareReport:
    return HardwareReport(
        ready=True,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        python_version="3.11.9",
        package_versions={
            "torch": "2.13.0",
            "transformers": "4.57.6",
            "safetensors": "0.8.0",
        },
        selected_device="cpu",
        selected_dtype="float32",
        cuda_available=False,
        system_ram_gb=16,
        model_cache_path="C:/cache/model",
        model_cached=True,
    )


def test_writes_ordered_predictions_and_reference_free_manifest(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    backend = FakeBackend()

    result = run_predictions(
        FIXTURE_BENCHMARK,
        output,
        _config(),
        _catalog(),
        _hardware(),
        backend,
    )

    assert [prediction.example_id for prediction in result.predictions] == [
        "fixture-math-1",
        "fixture-code-1",
        "fixture-logic-1",
        "fixture-knowledge-1",
    ]
    assert all(prediction.status is PredictionStatus.OK for prediction in result.predictions)
    assert all(prediction.prompt_tokens == 10 for prediction in result.predictions)
    assert all(prediction.completion_tokens == 2 for prediction in result.predictions)
    assert all(prediction.cost_usd == 0 for prediction in result.predictions)
    assert all(
        prediction.metadata["run_fingerprint"] == result.manifest.run_fingerprint
        for prediction in result.predictions
    )
    assert result.predictions[1].response == "def add(a, b):\n    return a + b"
    assert len(backend.requests) == 4
    for request in backend.requests:
        serialized = request.model_dump_json()
        assert '"reference"' not in serialized
        assert '"metadata"' not in serialized
    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    assert '"reference"' not in manifest_text
    assert not output.with_suffix(".jsonl.tmp").exists()


def test_recoverable_error_becomes_prediction_and_run_continues(tmp_path: Path) -> None:
    backend = FakeBackend({"fixture-code-1": RuntimeError("temporary\n failure")})
    output = tmp_path / "predictions.jsonl"

    times = iter([0.0, 10.0, 10.125, 20.0, 30.0])
    result = run_predictions(
        FIXTURE_BENCHMARK,
        output,
        _config(),
        _catalog(),
        _hardware(),
        backend,
        clock=lambda: next(times),
    )

    assert len(result.predictions) == 4
    error_prediction = result.predictions[1]
    assert error_prediction.status is PredictionStatus.ERROR
    assert error_prediction.response is None
    assert error_prediction.metadata["error_type"] == "RuntimeError"
    assert error_prediction.metadata["error_message"] == "temporary failure"
    assert error_prediction.latency_ms == pytest.approx(125.0)


def test_interrupted_run_resumes_to_uninterrupted_bytes(tmp_path: Path) -> None:
    interrupted_output = tmp_path / "interrupted.jsonl"
    interrupting_backend = FakeBackend({"fixture-code-1": KeyboardInterrupt()})

    with pytest.raises(KeyboardInterrupt):
        run_predictions(
            FIXTURE_BENCHMARK,
            interrupted_output,
            _config(),
            _catalog(),
            _hardware(),
            interrupting_backend,
        )

    assert [prediction.example_id for prediction in load_predictions(interrupted_output)] == [
        "fixture-math-1"
    ]
    resumed = run_predictions(
        FIXTURE_BENCHMARK,
        interrupted_output,
        _config(),
        _catalog(),
        _hardware(),
        FakeBackend(),
        PredictionRunOptions(resume=True),
    )
    uninterrupted_output = tmp_path / "uninterrupted.jsonl"
    uninterrupted = run_predictions(
        FIXTURE_BENCHMARK,
        uninterrupted_output,
        _config(),
        _catalog(),
        _hardware(),
        FakeBackend(),
    )

    assert interrupted_output.read_bytes() == uninterrupted_output.read_bytes()
    assert resumed.predictions == uninterrupted.predictions


def test_inspection_reports_completed_resume_without_backend(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    config = _config()
    catalog = _catalog()
    hardware = _hardware()
    run_predictions(
        FIXTURE_BENCHMARK,
        output,
        config,
        catalog,
        hardware,
        FakeBackend(),
    )
    options = PredictionRunOptions(resume=True)

    plan = inspect_prediction_run(
        FIXTURE_BENCHMARK,
        output,
        config,
        catalog,
        hardware,
        options,
    )
    resumed = run_predictions(
        FIXTURE_BENCHMARK,
        output,
        config,
        catalog,
        hardware,
        None,
        options,
    )

    assert plan.example_count == 4
    assert plan.completed_count == 4
    assert plan.pending_count == 0
    assert len(resumed.predictions) == 4


def test_inspection_rejects_collision_before_backend_creation(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="resume or overwrite"):
        inspect_prediction_run(
            FIXTURE_BENCHMARK,
            output,
            _config(),
            _catalog(),
            _hardware(),
        )


def test_resume_rejects_changed_profile(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    run_predictions(
        FIXTURE_BENCHMARK,
        output,
        _config(),
        _catalog(),
        _hardware(),
        FakeBackend(),
    )

    with pytest.raises(ResumeMismatchError, match="fingerprint"):
        run_predictions(
            FIXTURE_BENCHMARK,
            output,
            _config(),
            _catalog(),
            _hardware(),
            FakeBackend(),
            PredictionRunOptions(profile=PromptProfileName.MATH, resume=True),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "different/model", "model"),
        ("domain", Domain.CODE, "domain"),
    ],
)
def test_resume_rejects_mislabeled_prediction_rows(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    output = tmp_path / "predictions.jsonl"
    result = run_predictions(
        FIXTURE_BENCHMARK,
        output,
        _config(),
        _catalog(),
        _hardware(),
        FakeBackend(),
    )
    predictions = load_predictions(output)
    predictions[0] = predictions[0].model_copy(update={field: value})
    write_predictions(output, predictions)

    with pytest.raises(ResumeMismatchError, match=message):
        run_predictions(
            FIXTURE_BENCHMARK,
            output,
            _config(),
            _catalog(),
            _hardware(),
            FakeBackend(),
            PredictionRunOptions(resume=True),
        )
    assert result.manifest.run_fingerprint


def test_resume_rejects_stale_row_fingerprint(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    run_predictions(
        FIXTURE_BENCHMARK,
        output,
        _config(),
        _catalog(),
        _hardware(),
        FakeBackend(),
    )
    predictions = load_predictions(output)
    metadata = {**predictions[0].metadata, "run_fingerprint": "0" * 64}
    predictions[0] = predictions[0].model_copy(update={"metadata": metadata})
    write_predictions(output, predictions)

    with pytest.raises(ResumeMismatchError, match="fingerprint"):
        run_predictions(
            FIXTURE_BENCHMARK,
            output,
            _config(),
            _catalog(),
            _hardware(),
            FakeBackend(),
            PredictionRunOptions(resume=True),
        )


def test_run_lock_rejects_concurrent_writer(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"

    with acquire_run_lock(output), pytest.raises(FileExistsError, match="another process"):
        run_predictions(
            FIXTURE_BENCHMARK,
            output,
            _config(),
            _catalog(),
            _hardware(),
            FakeBackend(),
        )


def test_overwrite_clears_stale_rows_before_replacing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "predictions.jsonl"
    run_predictions(
        FIXTURE_BENCHMARK,
        output,
        _config(),
        _catalog(),
        _hardware(),
        FakeBackend(),
    )

    def fail_manifest(_path: Path, _manifest: object) -> None:
        raise RuntimeError("injected manifest failure")

    monkeypatch.setattr("small_models_society.inference.runner._write_manifest", fail_manifest)

    with pytest.raises(RuntimeError, match="injected manifest failure"):
        run_predictions(
            FIXTURE_BENCHMARK,
            output,
            _config(),
            _catalog(),
            _hardware(),
            FakeBackend(),
            PredictionRunOptions(profile=PromptProfileName.MATH, overwrite=True),
        )

    assert load_predictions(output) == []


def test_existing_output_requires_explicit_resume_or_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="resume or overwrite"):
        run_predictions(
            FIXTURE_BENCHMARK,
            output,
            _config(),
            _catalog(),
            _hardware(),
            FakeBackend(),
        )


def test_overwrite_replaces_existing_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text("existing", encoding="utf-8")
    manifest_path_for(output).write_text("existing", encoding="utf-8")

    result = run_predictions(
        FIXTURE_BENCHMARK,
        output,
        _config(),
        _catalog(),
        _hardware(),
        FakeBackend(),
        PredictionRunOptions(overwrite=True),
    )

    assert len(result.predictions) == 4
    assert len(load_predictions(output)) == 4


def test_domain_and_limit_filters_preserve_benchmark_order(tmp_path: Path) -> None:
    backend = FakeBackend()
    result = run_predictions(
        FIXTURE_BENCHMARK,
        tmp_path / "filtered.jsonl",
        _config(),
        _catalog(),
        _hardware(),
        backend,
        PredictionRunOptions(domains=[Domain.MATH, Domain.LOGIC], limit=2),
    )

    assert result.manifest.example_ids == ["fixture-math-1", "fixture-logic-1"]
    assert [DOMAIN_BY_REQUEST_ID[request.request_id] for request in backend.requests] == [
        Domain.MATH,
        Domain.LOGIC,
    ]


def test_fail_fast_preserves_completed_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "fail-fast.jsonl"
    backend = FakeBackend({"fixture-code-1": RuntimeError("stop now")})

    with pytest.raises(RuntimeError, match="stop now"):
        run_predictions(
            FIXTURE_BENCHMARK,
            output,
            _config(),
            _catalog(),
            _hardware(),
            backend,
            PredictionRunOptions(fail_fast=True),
        )

    assert [prediction.example_id for prediction in load_predictions(output)] == ["fixture-math-1"]


def test_adapter_identity_enters_requests_manifest_rows_and_resume(tmp_path: Path) -> None:
    output = tmp_path / "adapter.jsonl"
    backend = FakeBackend()
    adapter = AdapterReference(
        name="math",
        sha256="a" * 64,
        run_fingerprint="b" * 64,
    )
    options = PredictionRunOptions(
        domains=[Domain.MATH],
        adapter=adapter,
    )

    result = run_predictions(
        FIXTURE_BENCHMARK,
        output,
        _config(),
        _catalog(),
        _hardware(),
        backend,
        options,
    )

    assert result.manifest.adapter_name == "math"
    assert result.manifest.adapter_sha256 == "a" * 64
    assert backend.requests[0].adapter == "math"
    assert result.predictions[0].metadata["adapter"] == "math"
    assert result.predictions[0].metadata["adapter_sha256"] == "a" * 64

    predictions = load_predictions(output)
    metadata = {**predictions[0].metadata, "adapter_sha256": "c" * 64}
    write_predictions(output, [predictions[0].model_copy(update={"metadata": metadata})])
    with pytest.raises(ResumeMismatchError, match="adapter hash"):
        run_predictions(
            FIXTURE_BENCHMARK,
            output,
            _config(),
            _catalog(),
            _hardware(),
            backend,
            options.model_copy(update={"resume": True}),
        )


def test_out_of_memory_is_always_fatal(tmp_path: Path) -> None:
    output = tmp_path / "oom.jsonl"
    backend = FakeBackend({"fixture-code-1": InferenceOutOfMemoryError("oom")})

    with pytest.raises(InferenceOutOfMemoryError):
        run_predictions(
            FIXTURE_BENCHMARK,
            output,
            _config(),
            _catalog(),
            _hardware(),
            backend,
        )

    assert [prediction.example_id for prediction in load_predictions(output)] == ["fixture-math-1"]
