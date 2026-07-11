from __future__ import annotations

import json
from pathlib import Path

import pytest

from small_models_society.data.prepare import load_benchmark
from small_models_society.evaluation import (
    EvaluationStatus,
    evaluate_to_directory,
    write_predictions,
)
from small_models_society.fixtures import oracle_predictions
from small_models_society.sandbox import SandboxResult, SandboxStatus
from small_models_society.schemas import PredictionRecord, PredictionStatus

FIXTURE_BENCHMARK = Path(__file__).parent / "fixtures" / "benchmark.jsonl"


class FixtureSandbox:
    def run(self, candidate: str, test_setup: str, tests: list[str]) -> SandboxResult:
        del test_setup, tests
        passed = "return a + b" in candidate
        return SandboxResult(
            status=SandboxStatus.PASSED if passed else SandboxStatus.ASSERTION_FAILURE,
            duration_ms=1,
            stderr="" if passed else "assertion failed",
        )


def test_oracle_fixture_scores_100_percent_across_all_domains(tmp_path: Path) -> None:
    examples = load_benchmark(FIXTURE_BENCHMARK)
    predictions_path = tmp_path / "oracle.jsonl"
    write_predictions(predictions_path, oracle_predictions(examples))

    artifacts = evaluate_to_directory(
        FIXTURE_BENCHMARK,
        predictions_path,
        tmp_path / "report",
        FixtureSandbox(),
    )

    assert artifacts.summary["macro_primary_score"] == 1.0
    assert artifacts.summary["micro_primary_score"] == 1.0
    assert all(values["primary_score"] == 1.0 for values in artifacts.summary["domains"].values())
    assert artifacts.results_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.report_path.exists()

    serialized_predictions = predictions_path.read_text(encoding="utf-8")
    assert '"reference"' not in serialized_predictions
    assert '"gold"' not in serialized_predictions


def test_corrupted_fixtures_produce_expected_failures(tmp_path: Path) -> None:
    examples = load_benchmark(FIXTURE_BENCHMARK)
    predictions = oracle_predictions(examples, model_id="fixture-corrupted")
    wrong_answers = {
        "fixture-math-1": "11",
        "fixture-code-1": "def add(a, b):\n    return 0",
        "fixture-logic-1": "B",
        "fixture-knowledge-1": "London",
    }
    predictions = [
        prediction.model_copy(update={"response": wrong_answers[prediction.example_id]})
        for prediction in predictions
    ]
    predictions_path = tmp_path / "corrupted.jsonl"
    write_predictions(predictions_path, predictions)

    artifacts = evaluate_to_directory(
        FIXTURE_BENCHMARK,
        predictions_path,
        tmp_path / "report",
        FixtureSandbox(),
    )

    assert artifacts.summary["macro_primary_score"] == 0.0
    assert artifacts.summary["micro_primary_score"] == 0.0
    assert artifacts.summary["status"]["counts"][EvaluationStatus.SCORED] == 4
    results = [
        json.loads(line) for line in artifacts.results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(result["primary_score"] == 0.0 for result in results)


def test_unknown_prediction_ids_are_rejected(tmp_path: Path) -> None:
    examples = load_benchmark(FIXTURE_BENCHMARK)
    predictions = oracle_predictions(examples)
    predictions[0] = predictions[0].model_copy(update={"example_id": "unknown"})
    predictions_path = tmp_path / "unknown.jsonl"
    write_predictions(predictions_path, predictions)

    with pytest.raises(ValueError, match="unknown example IDs"):
        evaluate_to_directory(
            FIXTURE_BENCHMARK,
            predictions_path,
            tmp_path / "report",
            FixtureSandbox(),
        )


def test_abstentions_and_missing_predictions_remain_in_denominator(
    tmp_path: Path,
) -> None:
    examples = load_benchmark(FIXTURE_BENCHMARK)
    predictions = oracle_predictions(examples)
    abstained = PredictionRecord(
        example_id=predictions[0].example_id,
        domain=predictions[0].domain,
        model_id="fixture-abstain",
        status=PredictionStatus.ABSTAINED,
        latency_ms=0,
    )
    predictions = [abstained, *predictions[1:3]]
    predictions_path = tmp_path / "partial.jsonl"
    write_predictions(predictions_path, predictions)

    artifacts = evaluate_to_directory(
        FIXTURE_BENCHMARK,
        predictions_path,
        tmp_path / "report",
        FixtureSandbox(),
    )

    assert artifacts.summary["micro_primary_score"] == 0.5
    assert artifacts.summary["status"]["counts"][EvaluationStatus.ABSTAINED] == 1
    assert artifacts.summary["status"]["counts"][EvaluationStatus.MISSING_PREDICTION] == 1
