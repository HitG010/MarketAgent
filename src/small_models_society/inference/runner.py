"""Collision-safe, resumable orchestration for local prediction runs."""

from __future__ import annotations

import contextlib
import hashlib
import os
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from small_models_society.data.prepare import (
    canonical_json,
    load_benchmark,
    sha256_bytes,
)
from small_models_society.evaluation import load_predictions, write_predictions
from small_models_society.inference.config import InferenceConfig
from small_models_society.inference.contracts import (
    AdapterReference,
    TextGenerationBackend,
    to_inference_example,
)
from small_models_society.inference.hardware import HardwareReport
from small_models_society.inference.huggingface import InferenceOutOfMemoryError
from small_models_society.inference.prompts import (
    PromptCatalog,
    PromptProfileName,
    clean_response,
    render_generation_request,
)
from small_models_society.schemas import (
    BenchmarkExample,
    Domain,
    PredictionRecord,
    PredictionStatus,
    StrictModel,
)


class ResumeMismatchError(ValueError):
    """Raised when existing run artifacts were produced by a different run."""


class EmptyGenerationError(ValueError):
    """Raised when a backend returns no usable response text."""


class PredictionRunOptions(StrictModel):
    profile: PromptProfileName = PromptProfileName.GENERAL
    adapter: AdapterReference | None = None
    domains: list[Domain] = Field(default_factory=lambda: list(Domain), min_length=1)
    limit: int | None = Field(default=None, gt=0)
    resume: bool = False
    overwrite: bool = False
    fail_fast: bool = False

    @model_validator(mode="after")
    def validate_run_options(self) -> Self:
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        if len(set(self.domains)) != len(self.domains):
            raise ValueError("domains must not contain duplicates")
        return self


class RunManifest(StrictModel):
    schema_version: Literal[3] = 3
    benchmark_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inference_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_catalog_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    profile: PromptProfileName
    adapter_name: str | None = None
    adapter_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    adapter_run_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    selected_device: Literal["cpu", "cuda", "mps"]
    selected_dtype: Literal["float32", "float16", "bfloat16"]
    python_version: str
    cuda_device_name: str | None = None
    cuda_runtime_version: str | None = None
    cuda_vram_gb: float | None = Field(default=None, ge=0)
    implementation_version: Literal[3] = 3
    package_versions: dict[str, str | None]
    selected_domains: list[Domain]
    limit: int | None = Field(default=None, gt=0)
    fail_fast: bool
    example_ids: list[str]
    run_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    def fingerprint_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"run_fingerprint"})

    def calculated_fingerprint(self) -> str:
        payload = canonical_json(self.fingerprint_payload()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @model_validator(mode="after")
    def fingerprint_must_match(self) -> Self:
        if self.run_fingerprint != self.calculated_fingerprint():
            raise ValueError("run_fingerprint does not match manifest contents")
        return self


@dataclass(frozen=True)
class PredictionRunResult:
    output_path: Path
    manifest_path: Path
    manifest: RunManifest
    predictions: list[PredictionRecord]


@dataclass(frozen=True)
class PredictionRunPlan:
    output_path: Path
    manifest_path: Path
    manifest: RunManifest
    example_count: int
    completed_count: int
    pending_count: int


def manifest_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(".manifest.json")


def lock_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(f"{output_path.suffix}.lock")


@contextlib.contextmanager
def acquire_run_lock(output_path: Path) -> Iterator[Path]:
    """Hold a nonblocking process lock for one prediction output path."""

    lock_path = lock_path_for(output_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
    stream.seek(0)
    try:
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise FileExistsError(
                    f"another process is using prediction output: {output_path}"
                ) from error
            try:
                yield lock_path
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise FileExistsError(
                    f"another process is using prediction output: {output_path}"
                ) from error
            try:
                yield lock_path
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_manifest(path: Path, manifest: RunManifest) -> None:
    content = (canonical_json(manifest.model_dump(mode="json")) + "\n").encode("utf-8")
    _write_atomic(path, content)


def _load_manifest(path: Path) -> RunManifest:
    try:
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ResumeMismatchError(f"invalid run manifest: {path}") from error


def _select_examples(
    examples: list[BenchmarkExample],
    options: PredictionRunOptions,
) -> list[BenchmarkExample]:
    domains = set(options.domains)
    selected = [example for example in examples if example.domain in domains]
    if options.limit is not None:
        selected = selected[: options.limit]
    if not selected:
        raise ValueError("no benchmark examples match the requested filters")
    return selected


def _create_manifest(
    benchmark_path: Path,
    examples: list[BenchmarkExample],
    config: InferenceConfig,
    catalog: PromptCatalog,
    hardware: HardwareReport,
    options: PredictionRunOptions,
) -> RunManifest:
    values: dict[str, object] = {
        "schema_version": 3,
        "benchmark_sha256": sha256_bytes(benchmark_path.read_bytes()),
        "inference_config_fingerprint": config.fingerprint(),
        "prompt_catalog_fingerprint": catalog.fingerprint(),
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "profile": options.profile,
        "adapter_name": options.adapter.name if options.adapter else None,
        "adapter_sha256": options.adapter.sha256 if options.adapter else None,
        "adapter_run_fingerprint": (options.adapter.run_fingerprint if options.adapter else None),
        "selected_device": hardware.selected_device,
        "selected_dtype": hardware.selected_dtype,
        "python_version": hardware.python_version,
        "cuda_device_name": hardware.cuda_device_name,
        "cuda_runtime_version": hardware.cuda_runtime_version,
        "cuda_vram_gb": hardware.cuda_vram_gb,
        "implementation_version": 3,
        "package_versions": hardware.package_versions,
        "selected_domains": options.domains,
        "limit": options.limit,
        "fail_fast": options.fail_fast,
        "example_ids": [example.id for example in examples],
    }
    fingerprint = hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()
    return RunManifest.model_validate({**values, "run_fingerprint": fingerprint})


def _ordered_predictions(
    examples: list[BenchmarkExample],
    predictions_by_id: dict[str, PredictionRecord],
) -> list[PredictionRecord]:
    return [
        predictions_by_id[example.id] for example in examples if example.id in predictions_by_id
    ]


def _checkpoint(
    output_path: Path,
    examples: list[BenchmarkExample],
    predictions_by_id: dict[str, PredictionRecord],
) -> None:
    write_predictions(output_path, _ordered_predictions(examples, predictions_by_id))


def _error_metadata(
    error: Exception,
    manifest: RunManifest,
) -> dict[str, object]:
    message = " ".join(str(error).split())[:500]
    return {
        "profile": manifest.profile.value,
        "run_fingerprint": manifest.run_fingerprint,
        "model_revision": manifest.model_revision,
        "adapter": manifest.adapter_name,
        "adapter_sha256": manifest.adapter_sha256,
        "adapter_run_fingerprint": manifest.adapter_run_fingerprint,
        "error_type": type(error).__name__,
        "error_message": message or "generation failed without an error message",
    }


def _error_prediction(
    example: BenchmarkExample,
    config: InferenceConfig,
    manifest: RunManifest,
    error: Exception,
    latency_ms: float,
) -> PredictionRecord:
    return PredictionRecord(
        example_id=example.id,
        domain=example.domain,
        model_id=config.model.model_id,
        status=PredictionStatus.ERROR,
        latency_ms=latency_ms,
        metadata=_error_metadata(error, manifest),
    )


def _validate_resumed_predictions(
    examples: list[BenchmarkExample],
    manifest: RunManifest,
    predictions: list[PredictionRecord],
) -> dict[str, PredictionRecord]:
    examples_by_id = {example.id: example for example in examples}
    predictions_by_id: dict[str, PredictionRecord] = {}
    for prediction in predictions:
        example = examples_by_id.get(prediction.example_id)
        if example is None:
            raise ResumeMismatchError(
                f"existing prediction ID is outside this run: {prediction.example_id}"
            )
        if prediction.domain is not example.domain:
            raise ResumeMismatchError(
                f"existing prediction domain does not match {prediction.example_id}"
            )
        if prediction.model_id != manifest.model_id:
            raise ResumeMismatchError(
                f"existing prediction model does not match {prediction.example_id}"
            )
        if prediction.metadata.get("profile") != manifest.profile.value:
            raise ResumeMismatchError(
                f"existing prediction profile does not match {prediction.example_id}"
            )
        if prediction.metadata.get("adapter") != manifest.adapter_name:
            raise ResumeMismatchError(
                f"existing prediction adapter does not match {prediction.example_id}"
            )
        if prediction.metadata.get("adapter_sha256") != manifest.adapter_sha256:
            raise ResumeMismatchError(
                f"existing prediction adapter hash does not match {prediction.example_id}"
            )
        if prediction.metadata.get("adapter_run_fingerprint") != manifest.adapter_run_fingerprint:
            raise ResumeMismatchError(
                f"existing prediction adapter run does not match {prediction.example_id}"
            )
        if prediction.metadata.get("run_fingerprint") != manifest.run_fingerprint:
            raise ResumeMismatchError(
                f"existing prediction fingerprint does not match {prediction.example_id}"
            )
        predictions_by_id[prediction.example_id] = prediction
    return predictions_by_id


def _build_run_inputs(
    benchmark_path: Path,
    config: InferenceConfig,
    catalog: PromptCatalog,
    hardware: HardwareReport,
    options: PredictionRunOptions,
) -> tuple[list[BenchmarkExample], RunManifest]:
    if not hardware.ready:
        raise ValueError("hardware report is not ready for inference")
    catalog.get(options.profile)
    examples = _select_examples(load_benchmark(benchmark_path), options)
    return examples, _create_manifest(
        benchmark_path,
        examples,
        config,
        catalog,
        hardware,
        options,
    )


def _inspect_existing_artifacts(
    output_path: Path,
    manifest_path: Path,
    examples: list[BenchmarkExample],
    manifest: RunManifest,
    options: PredictionRunOptions,
) -> dict[str, PredictionRecord]:
    output_exists = output_path.exists()
    manifest_exists = manifest_path.exists()
    if options.resume:
        if not output_exists or not manifest_exists:
            raise ResumeMismatchError("resume requires both prediction and manifest files")
        existing_manifest = _load_manifest(manifest_path)
        if existing_manifest.run_fingerprint != manifest.run_fingerprint:
            raise ResumeMismatchError("existing run fingerprint does not match requested run")
        return _validate_resumed_predictions(
            examples,
            manifest,
            load_predictions(output_path),
        )
    if (output_exists or manifest_exists) and not options.overwrite:
        raise FileExistsError(
            "prediction artifacts already exist; use resume or overwrite explicitly"
        )
    return {}


def inspect_prediction_run(
    benchmark_path: Path,
    output_path: Path,
    config: InferenceConfig,
    catalog: PromptCatalog,
    hardware: HardwareReport,
    options: PredictionRunOptions | None = None,
) -> PredictionRunPlan:
    """Validate a run without mutating artifacts or loading a model backend."""

    resolved_options = options or PredictionRunOptions()
    examples, manifest = _build_run_inputs(
        benchmark_path,
        config,
        catalog,
        hardware,
        resolved_options,
    )
    manifest_path = manifest_path_for(output_path)
    with acquire_run_lock(output_path):
        existing = _inspect_existing_artifacts(
            output_path,
            manifest_path,
            examples,
            manifest,
            resolved_options,
        )
    completed_count = len(existing)
    return PredictionRunPlan(
        output_path=output_path,
        manifest_path=manifest_path,
        manifest=manifest,
        example_count=len(examples),
        completed_count=completed_count,
        pending_count=len(examples) - completed_count,
    )


def run_predictions(
    benchmark_path: Path,
    output_path: Path,
    config: InferenceConfig,
    catalog: PromptCatalog,
    hardware: HardwareReport,
    backend: TextGenerationBackend | None,
    options: PredictionRunOptions | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> PredictionRunResult:
    """Generate ordered predictions with atomic checkpoints and strict resume checks."""

    resolved_options = options or PredictionRunOptions()
    examples, manifest = _build_run_inputs(
        benchmark_path,
        config,
        catalog,
        hardware,
        resolved_options,
    )
    manifest_path = manifest_path_for(output_path)
    with acquire_run_lock(output_path):
        predictions_by_id = _inspect_existing_artifacts(
            output_path,
            manifest_path,
            examples,
            manifest,
            resolved_options,
        )
        if not resolved_options.resume:
            _checkpoint(output_path, examples, predictions_by_id)
            _write_manifest(manifest_path, manifest)

        if len(predictions_by_id) < len(examples) and backend is None:
            raise ValueError("a generation backend is required while predictions are pending")

        completed_since_checkpoint = 0
        for example in examples:
            if example.id in predictions_by_id:
                continue
            inference_example = to_inference_example(example)
            request = render_generation_request(
                inference_example,
                catalog,
                resolved_options.profile,
                config.generation.max_new_tokens[example.domain],
                manifest.adapter_name,
            )
            attempt_started = clock()
            try:
                assert backend is not None
                generation = backend.generate(request)
                response = clean_response(example.domain, generation.text)
                if not response:
                    raise EmptyGenerationError("backend returned an empty response")
                metadata = {
                    **generation.metadata,
                    "profile": resolved_options.profile.value,
                    "run_fingerprint": manifest.run_fingerprint,
                    "model_revision": manifest.model_revision,
                    "adapter": manifest.adapter_name,
                    "adapter_sha256": manifest.adapter_sha256,
                    "adapter_run_fingerprint": manifest.adapter_run_fingerprint,
                }
                prediction = PredictionRecord(
                    example_id=example.id,
                    domain=example.domain,
                    model_id=config.model.model_id,
                    response=response,
                    latency_ms=generation.latency_ms,
                    prompt_tokens=generation.prompt_tokens,
                    completion_tokens=generation.completion_tokens,
                    cost_usd=0,
                    metadata=metadata,
                )
            except KeyboardInterrupt:
                _checkpoint(output_path, examples, predictions_by_id)
                raise
            except InferenceOutOfMemoryError:
                _checkpoint(output_path, examples, predictions_by_id)
                raise
            except Exception as error:
                if resolved_options.fail_fast:
                    _checkpoint(output_path, examples, predictions_by_id)
                    raise
                prediction = _error_prediction(
                    example,
                    config,
                    manifest,
                    error,
                    (clock() - attempt_started) * 1000,
                )
            predictions_by_id[example.id] = prediction
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= config.generation.checkpoint_interval:
                _checkpoint(output_path, examples, predictions_by_id)
                completed_since_checkpoint = 0

        _checkpoint(output_path, examples, predictions_by_id)
        predictions = _ordered_predictions(examples, predictions_by_id)
        return PredictionRunResult(
            output_path=output_path,
            manifest_path=manifest_path,
            manifest=manifest,
            predictions=predictions,
        )
