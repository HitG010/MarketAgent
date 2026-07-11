"""Command-line interface for benchmark preparation and evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from small_models_society.data import BenchmarkConfig, load_config, prepare_benchmark
from small_models_society.data.prepare import load_benchmark
from small_models_society.evaluation import evaluate_to_directory, write_predictions
from small_models_society.fixtures import oracle_predictions
from small_models_society.sandbox import (
    DEFAULT_IMAGE,
    DockerSandbox,
    build_sandbox_image,
    docker_available,
    sandbox_image_available,
)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sms",
        description="Small Models Society benchmark harness",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    data_parser = commands.add_parser("data", help="prepare benchmark data")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    prepare_parser = data_commands.add_parser("prepare", help="download and normalize data")
    prepare_parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
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

    doctor_parser = commands.add_parser("doctor", help="check local prerequisites")
    doctor_parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
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
