"""Command-line interface for benchmark preparation and evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from small_models_society.data import BenchmarkConfig, load_config, prepare_benchmark
from small_models_society.data.prepare import load_benchmark
from small_models_society.evaluation import evaluate_to_directory, write_predictions
from small_models_society.fixtures import oracle_predictions
from small_models_society.inference.config import load_inference_config
from small_models_society.inference.hardware import detect_hardware
from small_models_society.inference.huggingface import HuggingFaceBackend
from small_models_society.inference.prompts import (
    PromptProfileName,
    load_prompt_catalog,
)
from small_models_society.inference.runner import (
    PredictionRunOptions,
    inspect_prediction_run,
    run_predictions,
)
from small_models_society.resources import (
    DEFAULT_BENCHMARK_CONFIG,
    DEFAULT_INFERENCE_CONFIG,
    DEFAULT_PROMPT_PROFILES,
)
from small_models_society.sandbox import (
    DEFAULT_IMAGE,
    DockerSandbox,
    build_sandbox_image,
    docker_available,
    sandbox_image_available,
)
from small_models_society.schemas import Domain

CommandHandler = Callable[[argparse.Namespace], int]


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _prepare_data(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    if arguments.sample_per_domain is not None:
        values = config.model_dump(mode="python")
        values["sample_per_domain"] = arguments.sample_per_domain
        config = BenchmarkConfig.model_validate(values)
    prepared = prepare_benchmark(config, arguments.output_dir)
    _print_json(
        {
            "benchmark": str(prepared.benchmark_path),
            "manifest": str(prepared.manifest_path),
            "row_count": prepared.row_count,
            "sha256": prepared.sha256,
        }
    )
    return 0


def _write_oracle(arguments: argparse.Namespace) -> int:
    examples = load_benchmark(arguments.benchmark)
    predictions = oracle_predictions(examples, arguments.model_id)
    write_predictions(arguments.output, predictions)
    _print_json({"output": str(arguments.output), "prediction_count": len(predictions)})
    return 0


def _evaluate(arguments: argparse.Namespace) -> int:
    sandbox = DockerSandbox(
        image=arguments.sandbox_image,
        timeout_seconds=arguments.timeout_seconds,
    )
    artifacts = evaluate_to_directory(
        arguments.benchmark,
        arguments.predictions,
        arguments.output_dir,
        sandbox,
    )
    _print_json(
        {
            "macro_primary_score": artifacts.summary["macro_primary_score"],
            "micro_primary_score": artifacts.summary["micro_primary_score"],
            "report": str(artifacts.report_path),
            "results": str(artifacts.results_path),
            "summary": str(artifacts.summary_path),
        }
    )
    return 0


def _doctor(arguments: argparse.Namespace) -> int:
    checks: dict[str, dict[str, Any]] = {}
    python_ok = sys.version_info >= (3, 11)
    checks["python"] = {
        "ok": python_ok,
        "detail": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }

    try:
        config = load_config(arguments.config)
        checks["benchmark_config"] = {
            "ok": True,
            "detail": f"{len(config.sources)} pinned sources",
        }
    except (OSError, ValueError) as error:
        checks["benchmark_config"] = {"ok": False, "detail": str(error)}

    daemon_ok = docker_available()
    checks["docker_daemon"] = {
        "ok": daemon_ok,
        "detail": "available" if daemon_ok else "unavailable",
    }
    build_error: str | None = None
    if daemon_ok and arguments.build_sandbox:
        try:
            build_sandbox_image(arguments.sandbox_image)
        except (OSError, subprocess.SubprocessError) as error:
            build_error = str(error)
    image_ok = (
        daemon_ok and build_error is None and sandbox_image_available(arguments.sandbox_image)
    )
    checks["sandbox_image"] = {
        "ok": image_ok,
        "detail": build_error or ("available" if image_ok else "not built"),
    }

    ok = all(check["ok"] for check in checks.values())
    _print_json({"checks": checks, "ok": ok})
    return 0 if ok else 1


def _inference_doctor(arguments: argparse.Namespace) -> int:
    config = load_inference_config(arguments.config)
    report = detect_hardware(config)
    _print_json(report.model_dump(mode="json"))
    return 0 if report.ready else 1


def _inference_predict(arguments: argparse.Namespace) -> int:
    config = load_inference_config(arguments.config)
    if arguments.local_files_only:
        model = config.model.model_copy(update={"local_files_only": True})
        config = config.model_copy(update={"model": model})
    catalog = load_prompt_catalog(arguments.prompts)
    profile = PromptProfileName(arguments.profile)
    domains = (
        [Domain(domain) for domain in arguments.domains] if arguments.domains else list(Domain)
    )
    options = PredictionRunOptions(
        profile=profile,
        domains=domains,
        limit=arguments.limit,
        resume=arguments.resume,
        overwrite=arguments.overwrite,
        fail_fast=arguments.fail_fast,
    )
    hardware = detect_hardware(config)
    if not hardware.ready:
        details = "; ".join(hardware.errors) or "unknown inference readiness error"
        raise RuntimeError(f"inference prerequisites are not ready: {details}")
    prediction_plan = inspect_prediction_run(
        arguments.benchmark,
        arguments.output,
        config,
        catalog,
        hardware,
        options,
    )
    plan = {
        "device": hardware.selected_device,
        "dtype": hardware.selected_dtype,
        "model_id": config.model.model_id,
        "profile": profile.value,
        "pending": prediction_plan.pending_count,
    }
    print(f"inference plan: {json.dumps(plan, sort_keys=True)}", file=sys.stderr)
    backend = HuggingFaceBackend(config, hardware) if prediction_plan.pending_count else None
    result = run_predictions(
        arguments.benchmark,
        arguments.output,
        config,
        catalog,
        hardware,
        backend,
        options,
    )
    status_counts = Counter(prediction.status.value for prediction in result.predictions)
    total_latency_ms = sum(prediction.latency_ms for prediction in result.predictions)
    prediction_count = len(result.predictions)
    _print_json(
        {
            "completion_tokens": sum(
                prediction.completion_tokens for prediction in result.predictions
            ),
            "device": hardware.selected_device,
            "dtype": hardware.selected_dtype,
            "manifest": str(result.manifest_path),
            "mean_latency_ms": (total_latency_ms / prediction_count if prediction_count else 0.0),
            "output": str(result.output_path),
            "prediction_count": prediction_count,
            "profile": profile.value,
            "prompt_tokens": sum(prediction.prompt_tokens for prediction in result.predictions),
            "run_fingerprint": result.manifest.run_fingerprint,
            "status_counts": dict(sorted(status_counts.items())),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sms",
        description="Small Models Society benchmark harness",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    data_parser = commands.add_parser("data", help="prepare benchmark data")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    prepare_parser = data_commands.add_parser("prepare", help="download and normalize data")
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_BENCHMARK_CONFIG)
    prepare_parser.add_argument("--output-dir", type=Path)
    prepare_parser.add_argument("--sample-per-domain", type=int)
    prepare_parser.set_defaults(handler=_prepare_data)

    fixtures_parser = commands.add_parser("fixtures", help="create fixture predictions")
    fixture_commands = fixtures_parser.add_subparsers(dest="fixtures_command", required=True)
    oracle_parser = fixture_commands.add_parser("oracle", help="write oracle predictions")
    oracle_parser.add_argument("--benchmark", type=Path, required=True)
    oracle_parser.add_argument("--output", type=Path, required=True)
    oracle_parser.add_argument("--model-id", default="fixture-oracle")
    oracle_parser.set_defaults(handler=_write_oracle)

    evaluate_parser = commands.add_parser("evaluate", help="score prediction JSONL")
    evaluate_parser.add_argument("--benchmark", type=Path, required=True)
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, default=Path("reports/latest"))
    evaluate_parser.add_argument("--sandbox-image", default=DEFAULT_IMAGE)
    evaluate_parser.add_argument("--timeout-seconds", type=float, default=2.0)
    evaluate_parser.set_defaults(handler=_evaluate)

    inference_parser = commands.add_parser("inference", help="run local model inference")
    inference_commands = inference_parser.add_subparsers(dest="inference_command", required=True)
    inference_doctor_parser = inference_commands.add_parser(
        "doctor", help="check local model prerequisites"
    )
    inference_doctor_parser.add_argument("--config", type=Path, default=DEFAULT_INFERENCE_CONFIG)
    inference_doctor_parser.set_defaults(handler=_inference_doctor)
    inference_predict_parser = inference_commands.add_parser(
        "predict", help="generate local model predictions"
    )
    inference_predict_parser.add_argument("--benchmark", type=Path, required=True)
    inference_predict_parser.add_argument("--output", type=Path, required=True)
    inference_predict_parser.add_argument("--config", type=Path, default=DEFAULT_INFERENCE_CONFIG)
    inference_predict_parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPT_PROFILES)
    inference_predict_parser.add_argument(
        "--profile",
        choices=[profile.value for profile in PromptProfileName],
        default=PromptProfileName.GENERAL.value,
    )
    inference_predict_parser.add_argument(
        "--domain",
        dest="domains",
        action="append",
        choices=[domain.value for domain in Domain],
    )
    inference_predict_parser.add_argument("--limit", type=int)
    collision_group = inference_predict_parser.add_mutually_exclusive_group()
    collision_group.add_argument("--resume", action="store_true")
    collision_group.add_argument("--overwrite", action="store_true")
    inference_predict_parser.add_argument("--local-files-only", action="store_true")
    inference_predict_parser.add_argument("--fail-fast", action="store_true")
    inference_predict_parser.set_defaults(handler=_inference_predict)

    doctor_parser = commands.add_parser("doctor", help="check local prerequisites")
    doctor_parser.add_argument("--config", type=Path, default=DEFAULT_BENCHMARK_CONFIG)
    doctor_parser.add_argument("--sandbox-image", default=DEFAULT_IMAGE)
    doctor_parser.add_argument("--build-sandbox", action="store_true")
    doctor_parser.set_defaults(handler=_doctor)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    handler = cast(CommandHandler, arguments.handler)
    try:
        return handler(arguments)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
