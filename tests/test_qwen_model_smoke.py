from __future__ import annotations

import os
from pathlib import Path

import pytest

from small_models_society.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    evaluate_to_directory,
)
from small_models_society.inference.config import load_inference_config
from small_models_society.inference.hardware import detect_hardware
from small_models_society.inference.huggingface import HuggingFaceBackend
from small_models_society.inference.prompts import load_prompt_catalog
from small_models_society.inference.runner import PredictionRunOptions, run_predictions
from small_models_society.sandbox import DockerSandbox, build_sandbox_image, docker_available
from small_models_society.schemas import Domain, PredictionStatus

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "inference.yaml"
PROMPT_CONFIG = Path(__file__).parents[1] / "configs" / "prompt_profiles.yaml"
FIXTURE_BENCHMARK = Path(__file__).parent / "fixtures" / "benchmark.jsonl"

pytestmark = [
    pytest.mark.model,
    pytest.mark.filterwarnings(
        r"ignore:hf_xet\.download_files\(\) is deprecated.*:DeprecationWarning:"
        r"huggingface_hub\.file_download"
    ),
]


def test_pinned_qwen_generates_contract_valid_predictions(tmp_path: Path) -> None:
    if os.getenv("SMS_RUN_MODEL_TESTS") != "1":
        pytest.skip("set SMS_RUN_MODEL_TESTS=1 to run the full-weight model smoke test")
    config = load_inference_config(CONFIG_PATH)
    generation = config.generation.model_copy(
        update={
            "max_new_tokens": {
                Domain.MATH: 32,
                Domain.CODE: 64,
                Domain.LOGIC: 16,
                Domain.KNOWLEDGE: 32,
            }
        }
    )
    config = config.model_copy(update={"generation": generation})
    hardware = detect_hardware(config)
    if not hardware.ready:
        pytest.fail("; ".join(hardware.errors))
    if not docker_available():
        pytest.fail("Docker is required to evaluate the generated code prediction")

    backend = HuggingFaceBackend(config, hardware)
    predictions_path = tmp_path / "predictions.jsonl"
    run = run_predictions(
        FIXTURE_BENCHMARK,
        predictions_path,
        config,
        load_prompt_catalog(PROMPT_CONFIG),
        hardware,
        backend,
        PredictionRunOptions(overwrite=True),
    )

    assert len(run.predictions) == 4
    assert all(prediction.status is PredictionStatus.OK for prediction in run.predictions)
    assert all(
        prediction.response and prediction.response.strip() for prediction in run.predictions
    )
    assert all(prediction.prompt_tokens > 0 for prediction in run.predictions)
    assert all(prediction.completion_tokens > 0 for prediction in run.predictions)
    assert all(prediction.latency_ms >= 0 for prediction in run.predictions)
    assert all(
        prediction.metadata["model_revision"] == config.model.revision
        for prediction in run.predictions
    )
    serialized_predictions = predictions_path.read_text(encoding="utf-8")
    assert '"reference"' not in serialized_predictions
    assert '"gold"' not in serialized_predictions

    build_sandbox_image()
    artifacts = evaluate_to_directory(
        FIXTURE_BENCHMARK,
        predictions_path,
        tmp_path / "evaluation",
        DockerSandbox(),
    )
    results = [
        EvaluationResult.model_validate_json(line)
        for line in artifacts.results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(results) == 4
    assert all(result.status is not EvaluationStatus.SANDBOX_ERROR for result in results)
    assert artifacts.summary["example_count"] == 4
