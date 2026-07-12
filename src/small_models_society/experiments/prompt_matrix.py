"""Cross-domain evaluation of general and specialist prompt profiles."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from pydantic import Field, model_validator

from small_models_society.data.prepare import canonical_json, load_benchmark
from small_models_society.evaluation import (
    CodeSandbox,
    EvaluationResult,
    EvaluationStatus,
    evaluate_records_to_directory,
)
from small_models_society.inference.config import InferenceConfig
from small_models_society.inference.contracts import TextGenerationBackend
from small_models_society.inference.hardware import HardwareReport
from small_models_society.inference.prompts import PromptCatalog, PromptProfileName
from small_models_society.inference.runner import (
    PredictionRunOptions,
    ResumeMismatchError,
    acquire_run_lock,
    inspect_prediction_run,
    manifest_path_for,
    run_predictions,
)
from small_models_society.schemas import BenchmarkExample, Domain, StrictModel


class PromptMatrixOptions(StrictModel):
    domains: list[Domain] = Field(default_factory=lambda: list(Domain), min_length=1)
    limit: int | None = Field(default=None, gt=0)
    resume: bool = False
    overwrite: bool = False
    fail_fast: bool = False

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        if len(set(self.domains)) != len(self.domains):
            raise ValueError("domains must not contain duplicates")
        return self


class ProfileResult(StrictModel):
    profile: PromptProfileName
    example_id: str
    domain: Domain
    status: EvaluationStatus
    primary_metric: str
    primary_score: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


@dataclass(frozen=True)
class PromptMatrixResult:
    results_path: Path
    summary_path: Path
    report_path: Path
    summary: dict[str, Any]
    prediction_paths: dict[PromptProfileName, Path]


@dataclass(frozen=True)
class PromptMatrixPlan:
    example_count: int
    profile_count: int
    pending_generation_count: int


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _select_examples(
    examples: list[BenchmarkExample], options: PromptMatrixOptions
) -> list[BenchmarkExample]:
    domains = set(options.domains)
    selected = [example for example in examples if example.domain in domains]
    if options.limit is not None:
        selected = selected[: options.limit]
    if not selected:
        raise ValueError("no benchmark examples match the requested matrix filters")
    return selected


def _check_aggregate_collision(
    output_dir: Path,
    options: PromptMatrixOptions,
) -> None:
    aggregate_paths = (
        output_dir / "profile_results.jsonl",
        output_dir / "specialization_summary.json",
        output_dir / "specialization_report.md",
    )
    if (
        any(path.exists() for path in aggregate_paths)
        and not options.resume
        and not options.overwrite
    ):
        raise FileExistsError(
            "prompt matrix artifacts already exist; use resume or overwrite explicitly"
        )


def _profile_options(
    prediction_path: Path,
    profile: PromptProfileName,
    options: PromptMatrixOptions,
) -> PredictionRunOptions:
    profile_resume = False
    if options.resume:
        prediction_exists = prediction_path.exists()
        manifest_exists = manifest_path_for(prediction_path).exists()
        if prediction_exists != manifest_exists:
            raise ResumeMismatchError(
                f"profile {profile.value} has only one of prediction or manifest artifacts"
            )
        profile_resume = prediction_exists and manifest_exists
    return PredictionRunOptions(
        profile=profile,
        domains=options.domains,
        limit=options.limit,
        resume=profile_resume,
        overwrite=options.overwrite,
        fail_fast=options.fail_fast,
    )


def _inspect_prompt_matrix_unlocked(
    benchmark_path: Path,
    output_dir: Path,
    config: InferenceConfig,
    catalog: PromptCatalog,
    hardware: HardwareReport,
    options: PromptMatrixOptions | None = None,
) -> PromptMatrixPlan:
    """Validate every profile run before a shared model backend is loaded."""

    resolved_options = options or PromptMatrixOptions()
    _check_aggregate_collision(output_dir, resolved_options)
    selected_examples = _select_examples(load_benchmark(benchmark_path), resolved_options)
    pending_count = 0
    for profile in PromptProfileName:
        prediction_path = output_dir / profile.value / "predictions.jsonl"
        plan = inspect_prediction_run(
            benchmark_path,
            prediction_path,
            config,
            catalog,
            hardware,
            _profile_options(prediction_path, profile, resolved_options),
        )
        pending_count += plan.pending_count
    return PromptMatrixPlan(
        example_count=len(selected_examples),
        profile_count=len(PromptProfileName),
        pending_generation_count=pending_count,
    )


def inspect_prompt_matrix(
    benchmark_path: Path,
    output_dir: Path,
    config: InferenceConfig,
    catalog: PromptCatalog,
    hardware: HardwareReport,
    options: PromptMatrixOptions | None = None,
) -> PromptMatrixPlan:
    lock_target = output_dir / "profile_results.jsonl"
    with acquire_run_lock(lock_target):
        return _inspect_prompt_matrix_unlocked(
            benchmark_path,
            output_dir,
            config,
            catalog,
            hardware,
            options,
        )


def _load_evaluation_results(path: Path) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                results.append(EvaluationResult.model_validate_json(line))
    return results


def _profile_rows(
    profile: PromptProfileName,
    results: list[EvaluationResult],
) -> list[ProfileResult]:
    return [
        ProfileResult(
            profile=profile,
            example_id=result.example_id,
            domain=result.domain,
            status=result.status,
            primary_metric=result.primary_metric,
            primary_score=result.primary_score,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
        for result in results
    ]


def _domain_matrix(
    profile_summaries: dict[PromptProfileName, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for profile in PromptProfileName:
        domain_summaries = profile_summaries[profile]["domains"]
        matrix[profile.value] = {
            domain.value: float(domain_summaries[domain.value]["primary_score"])
            for domain in Domain
            if domain.value in domain_summaries
        }
    return matrix


def _observed_oracle(profile_results: list[ProfileResult]) -> dict[str, float | int]:
    scores_by_example: dict[str, list[float]] = defaultdict(list)
    general_scores: dict[str, float] = {}
    for result in profile_results:
        scores_by_example[result.example_id].append(result.primary_score)
        if result.profile is PromptProfileName.GENERAL:
            general_scores[result.example_id] = result.primary_score
    example_count = len(scores_by_example)
    oracle_score = (
        sum(max(scores) for scores in scores_by_example.values()) / example_count
        if example_count
        else 0.0
    )
    general_score = sum(general_scores.values()) / len(general_scores) if general_scores else 0.0
    return {
        "example_count": example_count,
        "general_score": general_score,
        "oracle_score": oracle_score,
        "routing_opportunity": oracle_score - general_score,
    }


def _build_summary(
    profile_summaries: dict[PromptProfileName, dict[str, Any]],
    profile_results: list[ProfileResult],
) -> dict[str, Any]:
    matrix = _domain_matrix(profile_summaries)
    general_summary = profile_summaries[PromptProfileName.GENERAL]
    general_domains = matrix[PromptProfileName.GENERAL.value]
    profiles: dict[str, Any] = {}
    for profile in PromptProfileName:
        summary = profile_summaries[profile]
        domain_scores = matrix[profile.value]
        profiles[profile.value] = {
            "macro_primary_score": summary["macro_primary_score"],
            "micro_primary_score": summary["micro_primary_score"],
            "domains": domain_scores,
            "delta_from_general": {
                "macro": summary["macro_primary_score"] - general_summary["macro_primary_score"],
                "micro": summary["micro_primary_score"] - general_summary["micro_primary_score"],
                "domains": {
                    domain: score - general_domains[domain]
                    for domain, score in domain_scores.items()
                },
            },
            "latency": summary["latency"],
            "tokens": summary["tokens"],
            "status": summary["status"],
        }

    specialist_effects: dict[str, Any] = {}
    for profile in (
        PromptProfileName.MATH,
        PromptProfileName.CODE,
        PromptProfileName.LOGIC,
        PromptProfileName.KNOWLEDGE,
    ):
        own_domain = profile.value
        profile_domains = matrix[profile.value]
        if own_domain not in profile_domains:
            continue
        off_domains = [domain for domain in profile_domains if domain != own_domain]
        specialist_off_score = None
        general_off_score = None
        if off_domains:
            specialist_off_score = sum(profile_domains[domain] for domain in off_domains) / len(
                off_domains
            )
            general_off_score = sum(general_domains[domain] for domain in off_domains) / len(
                off_domains
            )
        specialist_effects[profile.value] = {
            "own_domain": own_domain,
            "own_domain_lift": profile_domains[own_domain] - general_domains[own_domain],
            "off_domain_delta": (
                specialist_off_score - general_off_score
                if specialist_off_score is not None and general_off_score is not None
                else None
            ),
            "off_domain_degradation": (
                general_off_score - specialist_off_score
                if specialist_off_score is not None and general_off_score is not None
                else None
            ),
        }

    return {
        "schema_version": 1,
        "experiment": "prompt_profiles_not_trained_models",
        "profile_order": [profile.value for profile in PromptProfileName],
        "profile_by_domain": matrix,
        "profiles": profiles,
        "specialist_effects": specialist_effects,
        "prompt_profile_oracle": _observed_oracle(profile_results),
    }


def render_specialization_report(summary: dict[str, Any]) -> str:
    matrix = summary["profile_by_domain"]
    profiles = summary["profiles"]
    lines = [
        "# Prompt Profile Specialization Report",
        "",
        "> These are prompt profiles sharing one base model, not trained specialists.",
        "",
        "| Profile | Math | Code | Logic | Knowledge | Macro | Micro | Delta vs general |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PromptProfileName:
        scores = matrix[profile.value]
        profile_summary = profiles[profile.value]
        domain_cells = [
            f"{scores[domain.value]:.3f}" if domain.value in scores else "-" for domain in Domain
        ]
        lines.append(
            f"| {profile.value} | {' | '.join(domain_cells)} | "
            f"{profile_summary['macro_primary_score']:.3f} | "
            f"{profile_summary['micro_primary_score']:.3f} | "
            f"{profile_summary['delta_from_general']['micro']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Specialist Effects",
            "",
            "| Profile | Own-domain lift | Off-domain delta | Off-domain degradation |",
            "|---|---:|---:|---:|",
        ]
    )
    for profile, effect in summary["specialist_effects"].items():
        off_domain_delta = effect["off_domain_delta"]
        off_domain_degradation = effect["off_domain_degradation"]
        delta_cell = f"{off_domain_delta:+.3f}" if off_domain_delta is not None else "-"
        degradation_cell = (
            f"{off_domain_degradation:+.3f}" if off_domain_degradation is not None else "-"
        )
        lines.append(
            f"| {profile} | {effect['own_domain_lift']:+.3f} | {delta_cell} | {degradation_cell} |"
        )
    oracle = summary["prompt_profile_oracle"]
    lines.extend(
        [
            "",
            "## Observed Prompt-Profile Oracle",
            "",
            f"- General profile score: {oracle['general_score']:.3f}",
            f"- Oracle profile score: {oracle['oracle_score']:.3f}",
            f"- Routing opportunity: {oracle['routing_opportunity']:+.3f}",
            "",
        ]
    )
    return "\n".join(lines)


def _run_prompt_matrix_unlocked(
    benchmark_path: Path,
    output_dir: Path,
    config: InferenceConfig,
    catalog: PromptCatalog,
    hardware: HardwareReport,
    backend: TextGenerationBackend | None,
    sandbox: CodeSandbox,
    options: PromptMatrixOptions | None = None,
) -> PromptMatrixResult:
    """Run every prompt profile over the same selected examples and aggregate results."""

    resolved_options = options or PromptMatrixOptions()
    results_path = output_dir / "profile_results.jsonl"
    summary_path = output_dir / "specialization_summary.json"
    report_path = output_dir / "specialization_report.md"
    _check_aggregate_collision(output_dir, resolved_options)

    selected_examples = _select_examples(load_benchmark(benchmark_path), resolved_options)
    selected_ids = [example.id for example in selected_examples]
    profile_summaries: dict[PromptProfileName, dict[str, Any]] = {}
    profile_results: list[ProfileResult] = []
    prediction_paths: dict[PromptProfileName, Path] = {}
    for profile in PromptProfileName:
        profile_dir = output_dir / profile.value
        prediction_path = profile_dir / "predictions.jsonl"
        profile_run_options = _profile_options(
            prediction_path,
            profile,
            resolved_options,
        )
        prediction_run = run_predictions(
            benchmark_path,
            prediction_path,
            config,
            catalog,
            hardware,
            backend,
            profile_run_options,
        )
        if prediction_run.manifest.example_ids != selected_ids:
            raise ValueError(f"profile {profile.value} selected a different benchmark slice")
        artifacts = evaluate_records_to_directory(
            selected_examples,
            prediction_run.predictions,
            profile_dir / "evaluation",
            sandbox,
        )
        evaluation_results = _load_evaluation_results(artifacts.results_path)
        if any(result.status is EvaluationStatus.SANDBOX_ERROR for result in evaluation_results):
            raise RuntimeError(
                f"profile {profile.value} encountered a sandbox infrastructure error"
            )
        profile_summaries[profile] = artifacts.summary
        profile_results.extend(_profile_rows(profile, evaluation_results))
        prediction_paths[profile] = prediction_path

    summary = _build_summary(profile_summaries, profile_results)
    result_lines = [canonical_json(result.model_dump(mode="json")) for result in profile_results]
    _write_atomic(results_path, ("\n".join(result_lines) + "\n").encode("utf-8"))
    _write_atomic(
        summary_path,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _write_atomic(report_path, render_specialization_report(summary).encode("utf-8"))
    return PromptMatrixResult(
        results_path=results_path,
        summary_path=summary_path,
        report_path=report_path,
        summary=summary,
        prediction_paths=prediction_paths,
    )


def run_prompt_matrix(
    benchmark_path: Path,
    output_dir: Path,
    config: InferenceConfig,
    catalog: PromptCatalog,
    hardware: HardwareReport,
    backend: TextGenerationBackend | None,
    sandbox: CodeSandbox,
    options: PromptMatrixOptions | None = None,
) -> PromptMatrixResult:
    lock_target = output_dir / "profile_results.jsonl"
    with acquire_run_lock(lock_target):
        return _run_prompt_matrix_unlocked(
            benchmark_path,
            output_dir,
            config,
            catalog,
            hardware,
            backend,
            sandbox,
            options,
        )
