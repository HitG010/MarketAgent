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
from small_models_society.experiments.prompt_matrix import (
    PromptMatrixOptions,
    inspect_prompt_matrix,
    run_prompt_matrix,
)
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
    DEFAULT_TRAINING_CONFIG,
)
from small_models_society.sandbox import (
    DEFAULT_IMAGE,
    DockerSandbox,
    build_sandbox_image,
    docker_available,
    sandbox_image_available,
)
from small_models_society.schemas import Domain
from small_models_society.training.config import TrainingConfig, load_training_config
from small_models_society.training.formatting import (
    build_sft_eligibility_filter,
    prepare_sft_data,
)
from small_models_society.training.hardware import detect_training_hardware
from small_models_society.training.prepare import prepare_training_data
from small_models_society.training.runner import (
    AdapterRunOptions,
    inspect_adapter_training,
    run_adapter_training,
)
from small_models_society.training.trainer import LoraTrainerBackend, load_training_modules

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


def _prompt_matrix(arguments: argparse.Namespace) -> int:
    config = load_inference_config(arguments.config)
    if arguments.local_files_only:
        model = config.model.model_copy(update={"local_files_only": True})
        config = config.model_copy(update={"model": model})
    catalog = load_prompt_catalog(arguments.prompts)
    domains = (
        [Domain(domain) for domain in arguments.domains] if arguments.domains else list(Domain)
    )
    options = PromptMatrixOptions(
        domains=domains,
        limit=arguments.limit,
        resume=arguments.resume,
        overwrite=arguments.overwrite,
        fail_fast=arguments.fail_fast,
    )
    selected_examples = [
        example for example in load_benchmark(arguments.benchmark) if example.domain in domains
    ]
    if arguments.limit is not None:
        selected_examples = selected_examples[: arguments.limit]
    if (
        any(example.domain is Domain.CODE for example in selected_examples)
        and not docker_available()
    ):
        raise RuntimeError("Docker is required when the prompt matrix includes code examples")
    if any(
        example.domain is Domain.CODE for example in selected_examples
    ) and not sandbox_image_available(arguments.sandbox_image):
        raise RuntimeError(
            "The requested sandbox image is unavailable; run `sms doctor --build-sandbox`."
        )
    hardware = detect_hardware(config)
    if not hardware.ready:
        details = "; ".join(hardware.errors) or "unknown inference readiness error"
        raise RuntimeError(f"inference prerequisites are not ready: {details}")
    matrix_plan = inspect_prompt_matrix(
        arguments.benchmark,
        arguments.output_dir,
        config,
        catalog,
        hardware,
        options,
    )
    print(
        "prompt matrix plan: "
        + json.dumps(
            {
                "device": hardware.selected_device,
                "dtype": hardware.selected_dtype,
                "examples": len(selected_examples),
                "model_id": config.model.model_id,
                "profiles": len(PromptProfileName),
                "pending": matrix_plan.pending_generation_count,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    backend = HuggingFaceBackend(config, hardware) if matrix_plan.pending_generation_count else None
    sandbox = DockerSandbox(
        image=arguments.sandbox_image,
        timeout_seconds=arguments.timeout_seconds,
    )
    result = run_prompt_matrix(
        arguments.benchmark,
        arguments.output_dir,
        config,
        catalog,
        hardware,
        backend,
        sandbox,
        options,
    )
    oracle = result.summary["prompt_profile_oracle"]
    _print_json(
        {
            "oracle_score": oracle["oracle_score"],
            "report": str(result.report_path),
            "results": str(result.results_path),
            "routing_opportunity": oracle["routing_opportunity"],
            "summary": str(result.summary_path),
        }
    )
    return 0


def _configured_training(arguments: argparse.Namespace) -> TrainingConfig:
    config = load_training_config(arguments.config)
    model_updates: dict[str, object] = {}
    if getattr(arguments, "local_files_only", False):
        model_updates["local_files_only"] = True
    data_updates: dict[str, object] = {}
    output_dir = getattr(arguments, "output_dir", None)
    if output_dir is not None:
        data_updates["output_dir"] = str(output_dir)
    benchmark = getattr(arguments, "benchmark", None)
    if benchmark is not None:
        data_updates["benchmark_path"] = str(benchmark)
    benchmark_manifest = getattr(arguments, "benchmark_manifest", None)
    if benchmark_manifest is not None:
        data_updates["benchmark_manifest_path"] = str(benchmark_manifest)
    output_updates: dict[str, object] = {}
    adapter_root = getattr(arguments, "adapter_root", None)
    if adapter_root is not None:
        output_updates["adapter_root"] = str(adapter_root)
    return config.model_copy(
        update={
            "model": config.model.model_copy(update=model_updates),
            "data": config.data.model_copy(update=data_updates),
            "output": config.output.model_copy(update=output_updates),
        }
    )


def _training_doctor(arguments: argparse.Namespace) -> int:
    config = _configured_training(arguments)
    report = detect_training_hardware(config, allow_cpu=arguments.allow_cpu)
    _print_json(report.model_dump(mode="json"))
    return 0 if report.ready else 1


def _training_prepare(arguments: argparse.Namespace) -> int:
    config = _configured_training(arguments)
    output_dir = Path(config.data.output_dir)
    expected = [
        output_dir / "train.jsonl",
        output_dir / "validation.jsonl",
        output_dir / "manifest.json",
        output_dir / "sft" / "train.jsonl",
        output_dir / "sft" / "validation.jsonl",
        output_dir / "sft" / "manifest.json",
    ]
    if any(path.exists() for path in expected) and not arguments.overwrite:
        raise FileExistsError(
            "training data artifacts already exist; use overwrite explicitly"
        )
    catalog = load_prompt_catalog(arguments.prompts)
    modules = load_training_modules()
    tokenizer = modules.auto_tokenizer.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
    )
    source = prepare_training_data(
        config,
        output_dir,
        eligibility_filter=build_sft_eligibility_filter(
            catalog,
            tokenizer,
            config.data.max_length,
        ),
    )
    sft = prepare_sft_data(
        config,
        catalog,
        tokenizer,
        source.train_path,
        source.validation_path,
        source.manifest_path,
        output_dir / "sft",
    )
    _print_json(
        {
            "source_manifest": str(source.manifest_path),
            "sft_manifest": str(sft.manifest_path),
            "train": str(sft.train_path),
            "validation": str(sft.validation_path),
            "train_row_count": sft.train_row_count,
            "validation_row_count": sft.validation_row_count,
            "train_sha256": sft.train_sha256,
            "validation_sha256": sft.validation_sha256,
        }
    )
    return 0


def _sft_paths(
    arguments: argparse.Namespace,
    config: TrainingConfig,
) -> tuple[Path, Path, Path]:
    default_root = Path(config.data.output_dir) / "sft"
    return (
        arguments.sft_train or default_root / "train.jsonl",
        arguments.sft_validation or default_root / "validation.jsonl",
        arguments.sft_manifest or default_root / "manifest.json",
    )


def _training_train(arguments: argparse.Namespace) -> int:
    config = _configured_training(arguments)
    specialist = Domain(arguments.specialist)
    options = AdapterRunOptions(
        specialist=specialist,
        resume=arguments.resume,
        overwrite=arguments.overwrite,
    )
    train_path, validation_path, manifest_path = _sft_paths(arguments, config)
    hardware = detect_training_hardware(config, allow_cpu=arguments.allow_cpu)
    if not hardware.ready:
        details = "; ".join(hardware.errors) or "unknown training readiness error"
        raise RuntimeError(f"training prerequisites are not ready: {details}")
    plan = inspect_adapter_training(
        config,
        hardware,
        train_path,
        validation_path,
        manifest_path,
        options,
    )
    print(
        "training plan: "
        + json.dumps(
            {
                "device": hardware.selected_device,
                "dtype": hardware.selected_dtype,
                "pending": plan.pending,
                "specialist": specialist.value,
                "train_rows": plan.train_row_count,
                "validation_rows": plan.validation_row_count,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    backend = LoraTrainerBackend(config, hardware) if plan.pending else None
    result = run_adapter_training(
        config,
        hardware,
        train_path,
        validation_path,
        manifest_path,
        backend,
        options,
    )
    _print_json(
        {
            "adapter_dir": str(result.adapter_dir),
            "adapter_sha256": result.manifest.adapter_sha256,
            "device": hardware.selected_device,
            "dtype": hardware.selected_dtype,
            "duration_seconds": result.manifest.duration_seconds,
            "eval_metrics": result.manifest.eval_metrics,
            "manifest": str(result.manifest_path),
            "run_fingerprint": result.manifest.run_fingerprint,
            "specialist": specialist.value,
            "status": result.manifest.status,
            "train_metrics": result.manifest.train_metrics,
            "trainable_parameters": result.manifest.trainable_parameters,
        }
    )
    return 0


def _training_train_all(arguments: argparse.Namespace) -> int:
    completed: list[str] = []
    for domain in Domain:
        command = [
            sys.executable,
            "-m",
            "small_models_society.cli",
            "training",
            "train",
            "--config",
            str(arguments.config),
            "--specialist",
            domain.value,
        ]
        path_arguments = (
            ("--sft-train", arguments.sft_train),
            ("--sft-validation", arguments.sft_validation),
            ("--sft-manifest", arguments.sft_manifest),
            ("--adapter-root", arguments.adapter_root),
        )
        for flag, value in path_arguments:
            if value is not None:
                command.extend((flag, str(value)))
        for flag, enabled in (
            ("--resume", arguments.resume),
            ("--overwrite", arguments.overwrite),
            ("--local-files-only", arguments.local_files_only),
            ("--allow-cpu", arguments.allow_cpu),
        ):
            if enabled:
                command.append(flag)
        process = subprocess.run(command, check=False)
        if process.returncode != 0:
            return int(process.returncode)
        completed.append(domain.value)
    _print_json({"completed_specialists": completed, "status": "completed"})
    return 0


def _add_training_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--sft-train", type=Path)
    parser.add_argument("--sft-validation", type=Path)
    parser.add_argument("--sft-manifest", type=Path)
    parser.add_argument("--adapter-root", type=Path)
    collision_group = parser.add_mutually_exclusive_group()
    collision_group.add_argument("--resume", action="store_true")
    collision_group.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")


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

    experiment_parser = commands.add_parser("experiment", help="run research experiments")
    experiment_commands = experiment_parser.add_subparsers(dest="experiment_command", required=True)
    matrix_parser = experiment_commands.add_parser(
        "prompt-matrix", help="compare all prompt profiles across domains"
    )
    matrix_parser.add_argument("--benchmark", type=Path, required=True)
    matrix_parser.add_argument("--output-dir", type=Path, required=True)
    matrix_parser.add_argument("--config", type=Path, default=DEFAULT_INFERENCE_CONFIG)
    matrix_parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPT_PROFILES)
    matrix_parser.add_argument(
        "--domain",
        dest="domains",
        action="append",
        choices=[domain.value for domain in Domain],
    )
    matrix_parser.add_argument("--limit", type=int)
    matrix_collision_group = matrix_parser.add_mutually_exclusive_group()
    matrix_collision_group.add_argument("--resume", action="store_true")
    matrix_collision_group.add_argument("--overwrite", action="store_true")
    matrix_parser.add_argument("--local-files-only", action="store_true")
    matrix_parser.add_argument("--fail-fast", action="store_true")
    matrix_parser.add_argument("--sandbox-image", default=DEFAULT_IMAGE)
    matrix_parser.add_argument("--timeout-seconds", type=float, default=2.0)
    matrix_parser.set_defaults(handler=_prompt_matrix)

    training_parser = commands.add_parser("training", help="prepare and train LoRA specialists")
    training_commands = training_parser.add_subparsers(
        dest="training_command", required=True
    )
    training_doctor_parser = training_commands.add_parser(
        "doctor", help="check LoRA training prerequisites"
    )
    training_doctor_parser.add_argument(
        "--config", type=Path, default=DEFAULT_TRAINING_CONFIG
    )
    training_doctor_parser.add_argument("--adapter-root", type=Path)
    training_doctor_parser.add_argument("--local-files-only", action="store_true")
    training_doctor_parser.add_argument("--allow-cpu", action="store_true")
    training_doctor_parser.set_defaults(handler=_training_doctor)

    training_prepare_parser = training_commands.add_parser(
        "prepare", help="prepare leakage-free specialist SFT data"
    )
    training_prepare_parser.add_argument(
        "--config", type=Path, default=DEFAULT_TRAINING_CONFIG
    )
    training_prepare_parser.add_argument(
        "--prompts", type=Path, default=DEFAULT_PROMPT_PROFILES
    )
    training_prepare_parser.add_argument("--output-dir", type=Path)
    training_prepare_parser.add_argument("--benchmark", type=Path)
    training_prepare_parser.add_argument("--benchmark-manifest", type=Path)
    training_prepare_parser.add_argument("--local-files-only", action="store_true")
    training_prepare_parser.add_argument("--overwrite", action="store_true")
    training_prepare_parser.set_defaults(handler=_training_prepare)

    training_train_parser = training_commands.add_parser(
        "train", help="train one LoRA specialist"
    )
    _add_training_run_arguments(training_train_parser)
    training_train_parser.add_argument(
        "--specialist", choices=[domain.value for domain in Domain], required=True
    )
    training_train_parser.set_defaults(handler=_training_train)

    training_all_parser = training_commands.add_parser(
        "train-all", help="train all specialists in sequential processes"
    )
    _add_training_run_arguments(training_all_parser)
    training_all_parser.set_defaults(handler=_training_train_all)

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
