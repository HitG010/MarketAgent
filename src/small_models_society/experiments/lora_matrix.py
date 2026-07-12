"""Cross-domain evaluation of base and learned LoRA specialist weights."""

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
from small_models_society.inference.adapters import AdapterCatalog, AdapterSpec
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

BASE_VARIANT = "base"


class LoraMatrixOptions(StrictModel):
    domains: list[Domain] = Field(default_factory=lambda: list(Domain), min_length=1)
    limit: int | None = Field(default=None, gt=0)
    resume: bool = False
    overwrite: bool = False
    fail_fast: bool = False
    prompt_summary_path: Path | None = None

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        if len(set(self.domains)) != len(self.domains):
            raise ValueError("domains must not contain duplicates")
        return self


class AdapterEvaluationResult(StrictModel):
    variant: str = Field(min_length=1)
    adapter_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    example_id: str
    domain: Domain
    status: EvaluationStatus
    primary_metric: str
    primary_score: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


@dataclass(frozen=True)
class LoraMatrixResult:
    results_path: Path
    summary_path: Path
    report_path: Path
    summary: dict[str, Any]
    prediction_paths: dict[str, Path]


@dataclass(frozen=True)
class LoraMatrixPlan:
    example_count: int
    variant_count: int
    pending_generation_count: int


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _variants(catalog: AdapterCatalog) -> list[tuple[str, AdapterSpec | None]]:
    return [(BASE_VARIANT, None), *[(domain.value, catalog.adapters[domain]) for domain in Domain]]


def _select_examples(
    examples: list[BenchmarkExample],
    options: LoraMatrixOptions,
) -> list[BenchmarkExample]:
    domains = set(options.domains)
    selected = [example for example in examples if example.domain in domains]
    if options.limit is not None:
        selected = selected[: options.limit]
    if not selected:
        raise ValueError("no benchmark examples match the requested LoRA matrix filters")
    return selected


def _check_aggregate_collision(output_dir: Path, options: LoraMatrixOptions) -> None:
    aggregate_paths = (
        output_dir / "adapter_results.jsonl",
        output_dir / "lora_specialization_summary.json",
        output_dir / "lora_specialization_report.md",
    )
    if (
        any(path.exists() for path in aggregate_paths)
        and not options.resume
        and not options.overwrite
    ):
        raise FileExistsError(
            "LoRA matrix artifacts already exist; use resume or overwrite explicitly"
        )


def _variant_options(
    prediction_path: Path,
    variant: str,
    adapter: AdapterSpec | None,
    options: LoraMatrixOptions,
) -> PredictionRunOptions:
    variant_resume = False
    if options.resume:
        prediction_exists = prediction_path.exists()
        manifest_exists = manifest_path_for(prediction_path).exists()
        if prediction_exists != manifest_exists:
            raise ResumeMismatchError(
                f"weight variant {variant} has only one of prediction or manifest artifacts"
            )
        variant_resume = prediction_exists and manifest_exists
    return PredictionRunOptions(
        profile=PromptProfileName.GENERAL,
        adapter=adapter.reference() if adapter else None,
        domains=options.domains,
        limit=options.limit,
        resume=variant_resume,
        overwrite=options.overwrite,
        fail_fast=options.fail_fast,
    )


def _inspect_lora_matrix_unlocked(
    benchmark_path: Path,
    output_dir: Path,
    config: InferenceConfig,
    prompts: PromptCatalog,
    hardware: HardwareReport,
    adapters: AdapterCatalog,
    options: LoraMatrixOptions,
) -> LoraMatrixPlan:
    _check_aggregate_collision(output_dir, options)
    selected_examples = _select_examples(load_benchmark(benchmark_path), options)
    pending_count = 0
    variants = _variants(adapters)
    for variant, adapter in variants:
        prediction_path = output_dir / variant / "predictions.jsonl"
        plan = inspect_prediction_run(
            benchmark_path,
            prediction_path,
            config,
            prompts,
            hardware,
            _variant_options(prediction_path, variant, adapter, options),
        )
        pending_count += plan.pending_count
    return LoraMatrixPlan(
        example_count=len(selected_examples),
        variant_count=len(variants),
        pending_generation_count=pending_count,
    )


def inspect_lora_matrix(
    benchmark_path: Path,
    output_dir: Path,
    config: InferenceConfig,
    prompts: PromptCatalog,
    hardware: HardwareReport,
    adapters: AdapterCatalog,
    options: LoraMatrixOptions | None = None,
) -> LoraMatrixPlan:
    resolved_options = options or LoraMatrixOptions()
    lock_target = output_dir / "adapter_results.jsonl"
    with acquire_run_lock(lock_target):
        return _inspect_lora_matrix_unlocked(
            benchmark_path,
            output_dir,
            config,
            prompts,
            hardware,
            adapters,
            resolved_options,
        )


def _load_evaluation_results(path: Path) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                results.append(EvaluationResult.model_validate_json(line))
    return results


def _adapter_rows(
    variant: str,
    adapter: AdapterSpec | None,
    results: list[EvaluationResult],
) -> list[AdapterEvaluationResult]:
    return [
        AdapterEvaluationResult(
            variant=variant,
            adapter_sha256=adapter.sha256 if adapter else None,
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
    summaries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    return {
        variant: {
            domain.value: float(summary["domains"][domain.value]["primary_score"])
            for domain in Domain
            if domain.value in summary["domains"]
        }
        for variant, summary in summaries.items()
    }


def _observed_oracle(
    results: list[AdapterEvaluationResult],
) -> dict[str, float | int]:
    scores_by_example: dict[str, list[float]] = defaultdict(list)
    base_scores: dict[str, float] = {}
    for result in results:
        scores_by_example[result.example_id].append(result.primary_score)
        if result.variant == BASE_VARIANT:
            base_scores[result.example_id] = result.primary_score
    example_count = len(scores_by_example)
    oracle_score = (
        sum(max(scores) for scores in scores_by_example.values()) / example_count
        if example_count
        else 0.0
    )
    base_score = sum(base_scores.values()) / len(base_scores) if base_scores else 0.0
    return {
        "example_count": example_count,
        "base_score": base_score,
        "oracle_score": oracle_score,
        "routing_opportunity": oracle_score - base_score,
    }


def _load_prompt_comparator(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        prompt_oracle = summary["prompt_profile_oracle"]
        prompt_opportunity = float(prompt_oracle["routing_opportunity"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid prompt matrix summary: {path}") from error
    return {"prompt_routing_opportunity": prompt_opportunity}


def _build_summary(
    summaries: dict[str, dict[str, Any]],
    results: list[AdapterEvaluationResult],
    adapters: AdapterCatalog,
    prompt_summary_path: Path | None,
) -> dict[str, Any]:
    matrix = _domain_matrix(summaries)
    base_summary = summaries[BASE_VARIANT]
    base_domains = matrix[BASE_VARIANT]
    variant_summaries: dict[str, Any] = {}
    for variant, summary in summaries.items():
        domain_scores = matrix[variant]
        base_micro = float(base_summary["micro_primary_score"])
        variant_micro = float(summary["micro_primary_score"])
        variant_summaries[variant] = {
            "adapter_sha256": (
                adapters.adapters[Domain(variant)].sha256 if variant != BASE_VARIANT else None
            ),
            "macro_primary_score": summary["macro_primary_score"],
            "micro_primary_score": variant_micro,
            "domains": domain_scores,
            "delta_from_base": {
                "macro": summary["macro_primary_score"] - base_summary["macro_primary_score"],
                "micro": variant_micro - base_micro,
                "domains": {
                    domain: score - base_domains[domain] for domain, score in domain_scores.items()
                },
            },
            "general_retention": (variant_micro / base_micro if base_micro > 0 else None),
            "latency": summary["latency"],
            "tokens": summary["tokens"],
            "status": summary["status"],
        }

    specialist_effects: dict[str, Any] = {}
    for domain in Domain:
        variant = domain.value
        domain_scores = matrix[variant]
        if variant not in domain_scores:
            continue
        off_domains = [name for name in domain_scores if name != variant]
        adapter_off_score = None
        base_off_score = None
        if off_domains:
            adapter_off_score = sum(domain_scores[name] for name in off_domains) / len(off_domains)
            base_off_score = sum(base_domains[name] for name in off_domains) / len(off_domains)
        specialist_effects[variant] = {
            "own_domain": variant,
            "own_domain_lift": domain_scores[variant] - base_domains[variant],
            "off_domain_delta": (
                adapter_off_score - base_off_score
                if adapter_off_score is not None and base_off_score is not None
                else None
            ),
            "off_domain_degradation": (
                base_off_score - adapter_off_score
                if adapter_off_score is not None and base_off_score is not None
                else None
            ),
        }

    measured_effects = list(specialist_effects.values())
    own_lifts = [float(effect["own_domain_lift"]) for effect in measured_effects]
    degradations = [
        float(effect["off_domain_degradation"])
        for effect in measured_effects
        if effect["off_domain_degradation"] is not None
    ]
    oracle = _observed_oracle(results)
    comparator = _load_prompt_comparator(prompt_summary_path)
    if comparator is not None:
        comparator["lora_minus_prompt_opportunity"] = (
            float(oracle["routing_opportunity"]) - comparator["prompt_routing_opportunity"]
        )
    return {
        "schema_version": 1,
        "experiment": "learned_lora_weights_fixed_general_prompt",
        "prompt_profile": PromptProfileName.GENERAL.value,
        "variant_order": [BASE_VARIANT, *[domain.value for domain in Domain]],
        "adapter_by_domain": matrix,
        "variants": variant_summaries,
        "specialist_effects": specialist_effects,
        "aggregate_differentiation": {
            "measured_specialists": len(measured_effects),
            "positive_own_domain_lift_count": sum(lift > 0 for lift in own_lifts),
            "mean_own_domain_lift": (sum(own_lifts) / len(own_lifts) if own_lifts else None),
            "mean_off_domain_degradation": (
                sum(degradations) / len(degradations) if degradations else None
            ),
        },
        "adapter_oracle": oracle,
        "prompt_matrix_comparator": comparator,
    }


def render_lora_specialization_report(summary: dict[str, Any]) -> str:
    matrix = summary["adapter_by_domain"]
    variants = summary["variants"]
    lines = [
        "# LoRA Specialist Weight Report",
        "",
        "> Every weight variant used the same fixed general prompt.",
        "",
        "| Weight variant | Math | Code | Logic | Knowledge | Macro | Micro | Delta vs base |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in summary["variant_order"]:
        scores = matrix[variant]
        values = variants[variant]
        domain_cells = [
            f"{scores[domain.value]:.3f}" if domain.value in scores else "-" for domain in Domain
        ]
        lines.append(
            f"| {variant} | {' | '.join(domain_cells)} | "
            f"{values['macro_primary_score']:.3f} | "
            f"{values['micro_primary_score']:.3f} | "
            f"{values['delta_from_base']['micro']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Specialist Effects",
            "",
            "| Adapter | Own-domain lift | Off-domain delta | Off-domain degradation |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant, effect in summary["specialist_effects"].items():
        off_delta = effect["off_domain_delta"]
        degradation = effect["off_domain_degradation"]
        lines.append(
            f"| {variant} | {effect['own_domain_lift']:+.3f} | "
            f"{f'{off_delta:+.3f}' if off_delta is not None else '-'} | "
            f"{f'{degradation:+.3f}' if degradation is not None else '-'} |"
        )
    aggregate = summary["aggregate_differentiation"]
    oracle = summary["adapter_oracle"]
    mean_own_lift = aggregate["mean_own_domain_lift"]
    mean_degradation = aggregate["mean_off_domain_degradation"]
    mean_own_lift_text = f"{mean_own_lift:+.3f}" if mean_own_lift is not None else "-"
    mean_degradation_text = f"{mean_degradation:+.3f}" if mean_degradation is not None else "-"
    lines.extend(
        [
            "",
            "## Aggregate Differentiation",
            "",
            f"- Positive own-domain lifts: {aggregate['positive_own_domain_lift_count']} / "
            f"{aggregate['measured_specialists']}",
            f"- Mean own-domain lift: {mean_own_lift_text}",
            f"- Mean off-domain degradation: {mean_degradation_text}",
            "",
            "## Observed Adapter Oracle",
            "",
            f"- Base score: {oracle['base_score']:.3f}",
            f"- Oracle adapter score: {oracle['oracle_score']:.3f}",
            f"- Routing opportunity: {oracle['routing_opportunity']:+.3f}",
            "",
            "> This report contains aggregate opportunity only; it does not emit router labels.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_lora_matrix_unlocked(
    benchmark_path: Path,
    output_dir: Path,
    config: InferenceConfig,
    prompts: PromptCatalog,
    hardware: HardwareReport,
    adapters: AdapterCatalog,
    backend: TextGenerationBackend | None,
    sandbox: CodeSandbox,
    options: LoraMatrixOptions,
) -> LoraMatrixResult:
    results_path = output_dir / "adapter_results.jsonl"
    summary_path = output_dir / "lora_specialization_summary.json"
    report_path = output_dir / "lora_specialization_report.md"
    _check_aggregate_collision(output_dir, options)
    selected_examples = _select_examples(load_benchmark(benchmark_path), options)
    selected_ids = [example.id for example in selected_examples]
    summaries: dict[str, dict[str, Any]] = {}
    all_results: list[AdapterEvaluationResult] = []
    prediction_paths: dict[str, Path] = {}
    for variant, adapter in _variants(adapters):
        variant_dir = output_dir / variant
        prediction_path = variant_dir / "predictions.jsonl"
        prediction_run = run_predictions(
            benchmark_path,
            prediction_path,
            config,
            prompts,
            hardware,
            backend,
            _variant_options(prediction_path, variant, adapter, options),
        )
        if prediction_run.manifest.example_ids != selected_ids:
            raise ValueError(f"weight variant {variant} selected a different benchmark slice")
        artifacts = evaluate_records_to_directory(
            selected_examples,
            prediction_run.predictions,
            variant_dir / "evaluation",
            sandbox,
        )
        evaluation_results = _load_evaluation_results(artifacts.results_path)
        if any(result.status is EvaluationStatus.SANDBOX_ERROR for result in evaluation_results):
            raise RuntimeError(
                f"weight variant {variant} encountered a sandbox infrastructure error"
            )
        summaries[variant] = artifacts.summary
        all_results.extend(_adapter_rows(variant, adapter, evaluation_results))
        prediction_paths[variant] = prediction_path

    summary = _build_summary(
        summaries,
        all_results,
        adapters,
        options.prompt_summary_path,
    )
    result_lines = [canonical_json(result.model_dump(mode="json")) for result in all_results]
    _write_atomic(results_path, ("\n".join(result_lines) + "\n").encode("utf-8"))
    _write_atomic(
        summary_path,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _write_atomic(report_path, render_lora_specialization_report(summary).encode("utf-8"))
    return LoraMatrixResult(
        results_path=results_path,
        summary_path=summary_path,
        report_path=report_path,
        summary=summary,
        prediction_paths=prediction_paths,
    )


def run_lora_matrix(
    benchmark_path: Path,
    output_dir: Path,
    config: InferenceConfig,
    prompts: PromptCatalog,
    hardware: HardwareReport,
    adapters: AdapterCatalog,
    backend: TextGenerationBackend | None,
    sandbox: CodeSandbox,
    options: LoraMatrixOptions | None = None,
) -> LoraMatrixResult:
    resolved_options = options or LoraMatrixOptions()
    lock_target = output_dir / "adapter_results.jsonl"
    with acquire_run_lock(lock_target):
        return _run_lora_matrix_unlocked(
            benchmark_path,
            output_dir,
            config,
            prompts,
            hardware,
            adapters,
            backend,
            sandbox,
            resolved_options,
        )
