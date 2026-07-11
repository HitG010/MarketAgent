"""Prediction evaluation and deterministic report generation."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from small_models_society.data.prepare import canonical_json, load_benchmark
from small_models_society.sandbox import DockerSandbox, SandboxResult, SandboxStatus
from small_models_society.schemas import (
    BenchmarkExample,
    CodeExample,
    Domain,
    KnowledgeExample,
    LogicExample,
    MathExample,
    PredictionRecord,
    PredictionStatus,
    StrictModel,
)
from small_models_society.scoring import score_choice, score_knowledge, score_math


class EvaluationStatus(StrEnum):
    SCORED = "scored"
    ABSTAINED = "abstained"
    PREDICTION_ERROR = "prediction_error"
    MISSING_PREDICTION = "missing_prediction"
    SANDBOX_ERROR = "sandbox_error"


class EvaluationResult(StrictModel):
    example_id: str
    domain: Domain
    model_id: str | None = None
    status: EvaluationStatus
    metrics: dict[str, float]
    primary_metric: str
    primary_score: float = Field(ge=0, le=1)
    latency_ms: float = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    sandbox_status: SandboxStatus | None = None
    sandbox_duration_ms: float | None = Field(default=None, ge=0)
    error: str | None = None


class CodeSandbox(Protocol):
    def run(self, candidate: str, test_setup: str, tests: list[str]) -> SandboxResult: ...


@dataclass(frozen=True)
class EvaluationArtifacts:
    results_path: Path
    summary_path: Path
    report_path: Path
    summary: dict[str, Any]


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def load_predictions(path: Path) -> list[PredictionRecord]:
    predictions: list[PredictionRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                predictions.append(PredictionRecord.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid prediction row at {path}:{line_number}") from error
    if len({prediction.example_id for prediction in predictions}) != len(predictions):
        raise ValueError(f"predictions contain duplicate example IDs: {path}")
    return predictions


def write_predictions(path: Path, predictions: Sequence[PredictionRecord]) -> None:
    lines = [canonical_json(prediction.model_dump(mode="json")) for prediction in predictions]
    content = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
    _write_atomic(path, content)


def _metric_shape(domain: Domain) -> tuple[str, dict[str, float]]:
    if domain is Domain.MATH:
        return "exact_match", {"exact_match": 0.0}
    if domain is Domain.CODE:
        return "pass_at_1", {"pass_at_1": 0.0}
    if domain is Domain.LOGIC:
        return "accuracy", {"accuracy": 0.0}
    return "f1", {"exact_match": 0.0, "f1": 0.0}


def _make_result(
    example: BenchmarkExample,
    prediction: PredictionRecord | None,
    status: EvaluationStatus,
    metrics: dict[str, float] | None = None,
    sandbox_result: SandboxResult | None = None,
    error: str | None = None,
) -> EvaluationResult:
    primary_metric, empty_metrics = _metric_shape(example.domain)
    resolved_metrics = metrics or empty_metrics
    return EvaluationResult(
        example_id=example.id,
        domain=example.domain,
        model_id=prediction.model_id if prediction else None,
        status=status,
        metrics=resolved_metrics,
        primary_metric=primary_metric,
        primary_score=resolved_metrics[primary_metric],
        latency_ms=prediction.latency_ms if prediction else 0,
        prompt_tokens=prediction.prompt_tokens if prediction else 0,
        completion_tokens=prediction.completion_tokens if prediction else 0,
        cost_usd=prediction.cost_usd if prediction else 0,
        sandbox_status=sandbox_result.status if sandbox_result else None,
        sandbox_duration_ms=sandbox_result.duration_ms if sandbox_result else None,
        error=error,
    )


def _score_prediction(
    example: BenchmarkExample,
    prediction: PredictionRecord | None,
    sandbox: CodeSandbox,
) -> EvaluationResult:
    if prediction is None:
        return _make_result(example, None, EvaluationStatus.MISSING_PREDICTION)
    if prediction.domain is not example.domain:
        return _make_result(
            example,
            prediction,
            EvaluationStatus.PREDICTION_ERROR,
            error=f"prediction domain {prediction.domain.value} does not match benchmark",
        )
    if prediction.status is PredictionStatus.ABSTAINED:
        return _make_result(example, prediction, EvaluationStatus.ABSTAINED)
    if prediction.status is PredictionStatus.ERROR:
        return _make_result(example, prediction, EvaluationStatus.PREDICTION_ERROR)

    assert prediction.response is not None
    if isinstance(example, MathExample):
        exact_match = score_math(prediction.response, example.reference.answer)
        return _make_result(
            example, prediction, EvaluationStatus.SCORED, {"exact_match": exact_match}
        )
    if isinstance(example, LogicExample):
        labels = [choice.label for choice in example.input.choices]
        accuracy = score_choice(prediction.response, example.reference.answer_label, labels)
        return _make_result(example, prediction, EvaluationStatus.SCORED, {"accuracy": accuracy})
    if isinstance(example, KnowledgeExample):
        score = score_knowledge(prediction.response, example.reference.answers)
        return _make_result(
            example,
            prediction,
            EvaluationStatus.SCORED,
            {"exact_match": score.exact_match, "f1": score.f1},
        )
    if isinstance(example, CodeExample):
        sandbox_result = sandbox.run(
            prediction.response,
            example.reference.test_setup,
            example.reference.tests,
        )
        status = (
            EvaluationStatus.SANDBOX_ERROR
            if sandbox_result.status is SandboxStatus.INFRASTRUCTURE_ERROR
            else EvaluationStatus.SCORED
        )
        return _make_result(
            example,
            prediction,
            status,
            {"pass_at_1": float(sandbox_result.passed)},
            sandbox_result,
            sandbox_result.stderr or None,
        )
    raise TypeError(f"unsupported example type: {type(example).__name__}")


def evaluate_predictions(
    examples: Sequence[BenchmarkExample],
    predictions: Sequence[PredictionRecord],
    sandbox: CodeSandbox,
) -> list[EvaluationResult]:
    prediction_by_id: dict[str, PredictionRecord] = {}
    for prediction in predictions:
        if prediction.example_id in prediction_by_id:
            raise ValueError(f"duplicate prediction ID: {prediction.example_id}")
        prediction_by_id[prediction.example_id] = prediction

    example_ids = {example.id for example in examples}
    unknown_ids = set(prediction_by_id) - example_ids
    if unknown_ids:
        raise ValueError(f"predictions contain unknown example IDs: {sorted(unknown_ids)}")
    return [
        _score_prediction(example, prediction_by_id.get(example.id), sandbox)
        for example in examples
    ]


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _efficiency(results: Sequence[EvaluationResult]) -> dict[str, Any]:
    observed = [result for result in results if result.model_id is not None]
    latencies = [result.latency_ms for result in observed]
    prompt_tokens = sum(result.prompt_tokens for result in observed)
    completion_tokens = sum(result.completion_tokens for result in observed)
    total_tokens = prompt_tokens + completion_tokens
    count = len(observed)
    total_cost = sum(result.cost_usd for result in observed)
    return {
        "latency": {
            "samples": count,
            "mean_ms": sum(latencies) / count if count else 0.0,
            "p95_ms": _percentile_95(latencies),
            "max_ms": max(latencies, default=0.0),
        },
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
            "mean_total": total_tokens / count if count else 0.0,
        },
        "cost": {
            "total_usd": total_cost,
            "mean_usd": total_cost / count if count else 0.0,
        },
    }


def _status_metrics(results: Sequence[EvaluationResult]) -> dict[str, Any]:
    counts = Counter(result.status.value for result in results)
    total = len(results)
    return {
        "counts": {status.value: counts[status.value] for status in EvaluationStatus},
        "rates": {
            status.value: counts[status.value] / total if total else 0.0
            for status in EvaluationStatus
        },
    }


def build_summary(results: Sequence[EvaluationResult]) -> dict[str, Any]:
    domain_summaries: dict[str, Any] = {}
    domain_primary_scores: list[float] = []
    for domain in Domain:
        domain_results = [result for result in results if result.domain is domain]
        if not domain_results:
            continue
        primary_metric = domain_results[0].primary_metric
        metric_names = sorted({metric for result in domain_results for metric in result.metrics})
        metric_means = {
            metric: sum(result.metrics.get(metric, 0.0) for result in domain_results)
            / len(domain_results)
            for metric in metric_names
        }
        primary_score = sum(result.primary_score for result in domain_results) / len(domain_results)
        domain_primary_scores.append(primary_score)
        domain_summaries[domain.value] = {
            "count": len(domain_results),
            "primary_metric": primary_metric,
            "primary_score": primary_score,
            "metrics": metric_means,
            "status": _status_metrics(domain_results),
            **_efficiency(domain_results),
        }

    total = len(results)
    return {
        "schema_version": 1,
        "example_count": total,
        "macro_primary_score": (
            sum(domain_primary_scores) / len(domain_primary_scores)
            if domain_primary_scores
            else 0.0
        ),
        "micro_primary_score": (
            sum(result.primary_score for result in results) / total if total else 0.0
        ),
        "domains": domain_summaries,
        "status": _status_metrics(results),
        **_efficiency(results),
    }


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Report",
        "",
        "| Domain | Examples | Primary metric | Score |",
        "|---|---:|---|---:|",
    ]
    domains = summary["domains"]
    for domain in Domain:
        if domain.value not in domains:
            continue
        values = domains[domain.value]
        lines.append(
            f"| {domain.value.title()} | {values['count']} | "
            f"{values['primary_metric']} | {values['primary_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"**Macro:** {summary['macro_primary_score']:.3f}  ",
            f"**Micro:** {summary['micro_primary_score']:.3f}",
            "",
            "## Outcomes",
            "",
            "| Status | Count | Rate |",
            "|---|---:|---:|",
        ]
    )
    for status in EvaluationStatus:
        count = summary["status"]["counts"][status.value]
        rate = summary["status"]["rates"][status.value]
        lines.append(f"| {status.value} | {count} | {rate:.1%} |")
    latency = summary["latency"]
    tokens = summary["tokens"]
    cost = summary["cost"]
    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            "| Mean latency | P95 latency | Prompt tokens | Completion tokens | Cost |",
            "|---:|---:|---:|---:|---:|",
            f"| {latency['mean_ms']:.2f} ms | {latency['p95_ms']:.2f} ms | "
            f"{tokens['prompt']} | {tokens['completion']} | ${cost['total_usd']:.6f} |",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_to_directory(
    benchmark_path: Path,
    predictions_path: Path,
    output_dir: Path,
    sandbox: CodeSandbox | None = None,
) -> EvaluationArtifacts:
    examples = load_benchmark(benchmark_path)
    predictions = load_predictions(predictions_path)
    results = evaluate_predictions(examples, predictions, sandbox or DockerSandbox())
    summary = build_summary(results)

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    result_lines = [canonical_json(result.model_dump(mode="json")) for result in results]
    _write_atomic(results_path, ("\n".join(result_lines) + "\n").encode("utf-8"))
    _write_atomic(
        summary_path,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _write_atomic(report_path, render_markdown_report(summary).encode("utf-8"))
    return EvaluationArtifacts(results_path, summary_path, report_path, summary)
