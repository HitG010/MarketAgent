"""Collision-safe, resumable orchestration for one specialist adapter run."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self, cast

from pydantic import Field, model_validator

from small_models_society.data.prepare import canonical_json, sha256_bytes
from small_models_society.schemas import Domain, StrictModel
from small_models_society.training.config import TrainingConfig
from small_models_society.training.contracts import SFTTrainingRecord, TrainingSplit
from small_models_society.training.formatting import load_sft_training_records
from small_models_society.training.hardware import TrainingHardwareReport
from small_models_society.training.trainer import TrainerBackendResult


class TrainingResumeMismatchError(ValueError):
    """Raised when existing adapter artifacts belong to another run."""


class AdapterRunOptions(StrictModel):
    specialist: Domain
    resume: bool = False
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_collision_policy(self) -> Self:
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        return self


class AdapterRunManifest(StrictModel):
    schema_version: Literal[1] = 1
    training_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    sft_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sft_train_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sft_validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_catalog_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    specialist: Domain
    selected_device: Literal["cpu", "cuda", "mps"]
    selected_dtype: Literal["float32", "float16", "bfloat16"]
    python_version: str
    platform_system: str
    platform_machine: str
    package_versions: dict[str, str | None]
    implementation_commit: str | None = None
    implementation_version: Literal[1] = 1
    train_source_ids: list[str]
    validation_source_ids: list[str]
    run_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["running", "completed"] = "running"
    adapter_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    train_metrics: dict[str, object] | None = None
    eval_metrics: dict[str, object] | None = None
    trainable_parameters: int | None = Field(default=None, gt=0)
    total_parameters: int | None = Field(default=None, gt=0)
    duration_seconds: float | None = Field(default=None, ge=0)
    resumed_from_checkpoint: str | None = None

    def fingerprint_payload(self) -> dict[str, object]:
        return self.model_dump(
            mode="json",
            exclude={
                "run_fingerprint",
                "status",
                "adapter_sha256",
                "train_metrics",
                "eval_metrics",
                "trainable_parameters",
                "total_parameters",
                "duration_seconds",
                "resumed_from_checkpoint",
            },
        )

    def calculated_fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(self.fingerprint_payload()).encode("utf-8")
        ).hexdigest()

    @model_validator(mode="after")
    def validate_fingerprint_and_completion(self) -> Self:
        if self.run_fingerprint != self.calculated_fingerprint():
            raise ValueError("run_fingerprint does not match manifest contents")
        completion_fields = (
            self.adapter_sha256,
            self.train_metrics,
            self.eval_metrics,
            self.trainable_parameters,
            self.total_parameters,
            self.duration_seconds,
        )
        if self.status == "completed" and any(value is None for value in completion_fields):
            raise ValueError("completed adapter manifest is missing training results")
        if self.status == "running" and any(value is not None for value in completion_fields):
            raise ValueError("running adapter manifest cannot contain completed results")
        if (
            self.trainable_parameters is not None
            and self.total_parameters is not None
            and self.trainable_parameters >= self.total_parameters
        ):
            raise ValueError("trainable parameters must be a strict subset of total parameters")
        return self


@dataclass(frozen=True)
class SFTDatasetBundle:
    train_records: list[SFTTrainingRecord]
    validation_records: list[SFTTrainingRecord]
    manifest_sha256: str
    train_sha256: str
    validation_sha256: str
    prompt_catalog_fingerprint: str


@dataclass(frozen=True)
class AdapterTrainingPlan:
    specialist: Domain
    final_dir: Path
    work_dir: Path
    manifest: AdapterRunManifest
    train_row_count: int
    validation_row_count: int
    pending: bool
    resume_from_checkpoint: Path | None


@dataclass(frozen=True)
class AdapterTrainingResult:
    specialist: Domain
    adapter_dir: Path
    manifest_path: Path
    manifest: AdapterRunManifest


class SpecialistTrainerBackend(Protocol):
    def train(
        self,
        specialist: Domain,
        train_records: list[SFTTrainingRecord],
        validation_records: list[SFTTrainingRecord],
        work_dir: Path,
        resume_from_checkpoint: Path | None = None,
    ) -> TrainerBackendResult: ...


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return cast(Mapping[str, Any], value)


def _verified_sft_file(
    path: Path,
    metadata: object,
    split: TrainingSplit,
) -> tuple[list[SFTTrainingRecord], str]:
    file_metadata = _mapping(metadata, f"SFT {split.value} file metadata")
    actual_sha256 = sha256_bytes(path.read_bytes())
    if file_metadata.get("sha256") != actual_sha256:
        raise ValueError(f"SFT {split.value} hash does not match SFT manifest")
    records = load_sft_training_records(path)
    if file_metadata.get("row_count") != len(records):
        raise ValueError(f"SFT {split.value} row count does not match SFT manifest")
    if any(record.split is not split for record in records):
        raise ValueError(f"SFT {split.value} file contains the wrong split")
    return records, actual_sha256


def load_sft_dataset_bundle(
    config: TrainingConfig,
    train_path: Path,
    validation_path: Path,
    manifest_path: Path,
) -> SFTDatasetBundle:
    """Load and cross-check all model-facing SFT artifacts before model loading."""

    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = _mapping(json.loads(manifest_bytes), "SFT manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid SFT manifest: {manifest_path}") from error
    if not config.accepts_fingerprint(manifest.get("training_config_sha256")):
        raise ValueError("SFT manifest uses a different training configuration")
    if manifest.get("prompt_profile") != "general":
        raise ValueError("SFT data must use the fixed general prompt profile")
    if manifest.get("completion_only_loss") is not True:
        raise ValueError("SFT data must use completion-only loss")
    if manifest.get("max_length") != config.data.max_length:
        raise ValueError("SFT data max_length does not match training configuration")
    tokenizer = _mapping(manifest.get("tokenizer"), "SFT tokenizer")
    if tokenizer.get("model_id") != config.model.model_id:
        raise ValueError("SFT tokenizer model does not match training model")
    if tokenizer.get("revision") != config.model.revision:
        raise ValueError("SFT tokenizer revision does not match training model")
    prompt_fingerprint = manifest.get("prompt_catalog_sha256")
    if not isinstance(prompt_fingerprint, str) or len(prompt_fingerprint) != 64:
        raise ValueError("SFT manifest has an invalid prompt catalog fingerprint")

    files = _mapping(manifest.get("files"), "SFT files")
    train_records, train_sha256 = _verified_sft_file(
        train_path,
        files.get("train"),
        TrainingSplit.TRAIN,
    )
    validation_records, validation_sha256 = _verified_sft_file(
        validation_path,
        files.get("validation"),
        TrainingSplit.VALIDATION,
    )
    expected_train = config.data.train_size_per_domain * len(Domain)
    expected_validation = config.data.validation_size_per_domain * len(Domain)
    if len(train_records) != expected_train or len(validation_records) != expected_validation:
        raise ValueError("SFT artifact counts do not match the balanced configured domain counts")
    for domain in Domain:
        if sum(record.domain is domain for record in train_records) != (
            config.data.train_size_per_domain
        ):
            raise ValueError(f"SFT train data is not balanced for {domain.value}")
        if sum(record.domain is domain for record in validation_records) != (
            config.data.validation_size_per_domain
        ):
            raise ValueError(f"SFT validation data is not balanced for {domain.value}")
    train_ids = {record.source_id for record in train_records}
    if not train_ids.isdisjoint(record.source_id for record in validation_records):
        raise ValueError("SFT train and validation source IDs overlap")
    train_content = {record.source_content_sha256 for record in train_records}
    if not train_content.isdisjoint(record.source_content_sha256 for record in validation_records):
        raise ValueError("SFT train and validation source content overlaps")
    return SFTDatasetBundle(
        train_records=train_records,
        validation_records=validation_records,
        manifest_sha256=sha256_bytes(manifest_bytes),
        train_sha256=train_sha256,
        validation_sha256=validation_sha256,
        prompt_catalog_fingerprint=prompt_fingerprint,
    )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


def _create_manifest(
    config: TrainingConfig,
    hardware: TrainingHardwareReport,
    bundle: SFTDatasetBundle,
    specialist: Domain,
    train_records: list[SFTTrainingRecord],
    validation_records: list[SFTTrainingRecord],
) -> AdapterRunManifest:
    values: dict[str, object] = {
        "schema_version": 1,
        "training_config_fingerprint": config.fingerprint(),
        "sft_manifest_sha256": bundle.manifest_sha256,
        "sft_train_sha256": bundle.train_sha256,
        "sft_validation_sha256": bundle.validation_sha256,
        "prompt_catalog_fingerprint": bundle.prompt_catalog_fingerprint,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "specialist": specialist,
        "selected_device": hardware.selected_device,
        "selected_dtype": hardware.selected_dtype,
        "python_version": hardware.python_version,
        "platform_system": hardware.platform_system,
        "platform_machine": hardware.platform_machine,
        "package_versions": hardware.package_versions,
        "implementation_commit": _git_commit(),
        "implementation_version": 1,
        "train_source_ids": [record.source_id for record in train_records],
        "validation_source_ids": [record.source_id for record in validation_records],
        "status": "running",
    }
    fingerprint_values = {key: value for key, value in values.items() if key != "status"}
    fingerprint = hashlib.sha256(canonical_json(fingerprint_values).encode("utf-8")).hexdigest()
    return AdapterRunManifest.model_validate({**values, "run_fingerprint": fingerprint})


def _paths(config: TrainingConfig, specialist: Domain) -> tuple[Path, Path]:
    root = Path(config.output.adapter_root)
    return root / specialist.value, root / f".{specialist.value}.work"


def _lock_path(config: TrainingConfig, specialist: Domain) -> Path:
    return Path(config.output.adapter_root) / f".{specialist.value}.lock"


@contextlib.contextmanager
def acquire_adapter_lock(config: TrainingConfig, specialist: Domain) -> Iterator[Path]:
    """Hold a nonblocking process lock for one specialist output."""

    lock_path = _lock_path(config, specialist)
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
                    f"another process is training the {specialist.value} adapter"
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
                    f"another process is training the {specialist.value} adapter"
                ) from error
            try:
                yield lock_path
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def _write_manifest(path: Path, manifest: AdapterRunManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary_path.write_text(
            canonical_json(manifest.model_dump(mode="json")) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_manifest(path: Path) -> AdapterRunManifest:
    try:
        return AdapterRunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TrainingResumeMismatchError(f"invalid adapter run manifest: {path}") from error


def _latest_checkpoint(work_dir: Path) -> Path | None:
    checkpoint_root = work_dir / "checkpoints"
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.glob("checkpoint-*"):
        try:
            number = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if path.is_dir():
            candidates.append((number, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _validate_completed_adapter(final_dir: Path, manifest: AdapterRunManifest) -> None:
    weights_path = final_dir / "adapter_model.safetensors"
    if not weights_path.is_file() or weights_path.stat().st_size <= 0:
        raise TrainingResumeMismatchError("completed adapter weights are missing")
    if manifest.adapter_sha256 != sha256_bytes(weights_path.read_bytes()):
        raise TrainingResumeMismatchError("completed adapter hash does not match manifest")


def _validate_backend_adapter(result: TrainerBackendResult) -> None:
    weights_path = result.adapter_dir / "adapter_model.safetensors"
    config_path = result.adapter_dir / "adapter_config.json"
    if not weights_path.is_file() or weights_path.stat().st_size <= 0:
        raise RuntimeError("trainer backend did not produce adapter weights")
    if not config_path.is_file():
        raise RuntimeError("trainer backend did not produce an adapter configuration")
    if sha256_bytes(weights_path.read_bytes()) != result.adapter_sha256:
        raise RuntimeError("trainer backend adapter hash does not match adapter bytes")


def _inspect_artifacts(
    config: TrainingConfig,
    options: AdapterRunOptions,
    manifest: AdapterRunManifest,
) -> tuple[bool, Path | None]:
    final_dir, work_dir = _paths(config, options.specialist)
    final_exists = final_dir.exists()
    work_exists = work_dir.exists()
    if options.resume:
        if final_exists and work_exists:
            raise TrainingResumeMismatchError(
                "resume found both completed and in-progress adapter artifacts"
            )
        if final_exists:
            existing = _load_manifest(final_dir / "manifest.json")
            if existing.run_fingerprint != manifest.run_fingerprint:
                raise TrainingResumeMismatchError(
                    "completed adapter fingerprint does not match requested run"
                )
            if existing.status != "completed":
                raise TrainingResumeMismatchError("final adapter manifest is not completed")
            _validate_completed_adapter(final_dir, existing)
            return False, None
        if work_exists:
            existing = _load_manifest(work_dir / "run-manifest.json")
            if existing.run_fingerprint != manifest.run_fingerprint:
                raise TrainingResumeMismatchError(
                    "in-progress adapter fingerprint does not match requested run"
                )
            if existing.status != "running":
                raise TrainingResumeMismatchError("work adapter manifest is not running")
            return True, _latest_checkpoint(work_dir)
        raise TrainingResumeMismatchError(
            "resume requires either in-progress or completed adapter artifacts"
        )
    if (final_exists or work_exists) and not options.overwrite:
        raise FileExistsError("adapter artifacts already exist; use resume or overwrite explicitly")
    return True, None


def _build_inputs(
    config: TrainingConfig,
    hardware: TrainingHardwareReport,
    train_path: Path,
    validation_path: Path,
    sft_manifest_path: Path,
    options: AdapterRunOptions,
) -> tuple[SFTDatasetBundle, list[SFTTrainingRecord], list[SFTTrainingRecord], AdapterRunManifest]:
    if not hardware.ready:
        raise ValueError("hardware report is not ready for training")
    bundle = load_sft_dataset_bundle(
        config,
        train_path,
        validation_path,
        sft_manifest_path,
    )
    train_records = [
        record for record in bundle.train_records if record.domain is options.specialist
    ]
    validation_records = [
        record for record in bundle.validation_records if record.domain is options.specialist
    ]
    manifest = _create_manifest(
        config,
        hardware,
        bundle,
        options.specialist,
        train_records,
        validation_records,
    )
    return bundle, train_records, validation_records, manifest


def inspect_adapter_training(
    config: TrainingConfig,
    hardware: TrainingHardwareReport,
    train_path: Path,
    validation_path: Path,
    sft_manifest_path: Path,
    options: AdapterRunOptions,
) -> AdapterTrainingPlan:
    """Validate one adapter run without loading the base model."""

    _bundle, train_records, validation_records, manifest = _build_inputs(
        config,
        hardware,
        train_path,
        validation_path,
        sft_manifest_path,
        options,
    )
    final_dir, work_dir = _paths(config, options.specialist)
    with acquire_adapter_lock(config, options.specialist):
        pending, checkpoint = _inspect_artifacts(config, options, manifest)
    return AdapterTrainingPlan(
        specialist=options.specialist,
        final_dir=final_dir,
        work_dir=work_dir,
        manifest=manifest,
        train_row_count=len(train_records),
        validation_row_count=len(validation_records),
        pending=pending,
        resume_from_checkpoint=checkpoint,
    )


def _completed_manifest(
    running: AdapterRunManifest,
    result: TrainerBackendResult,
) -> AdapterRunManifest:
    values = running.model_dump(mode="json")
    values.update(
        {
            "status": "completed",
            "adapter_sha256": result.adapter_sha256,
            "train_metrics": result.train_metrics,
            "eval_metrics": result.eval_metrics,
            "trainable_parameters": result.trainable_parameters,
            "total_parameters": result.total_parameters,
            "duration_seconds": result.duration_seconds,
            "resumed_from_checkpoint": result.resumed_from_checkpoint,
        }
    )
    return AdapterRunManifest.model_validate(values)


def _publish_adapter(source: Path, destination: Path, overwrite: bool) -> None:
    backup = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.backup"
    moved_existing = False
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"adapter destination already exists: {destination}")
        destination.replace(backup)
        moved_existing = True
    try:
        source.replace(destination)
    except BaseException:
        if moved_existing and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def run_adapter_training(
    config: TrainingConfig,
    hardware: TrainingHardwareReport,
    train_path: Path,
    validation_path: Path,
    sft_manifest_path: Path,
    backend: SpecialistTrainerBackend | None,
    options: AdapterRunOptions,
) -> AdapterTrainingResult:
    """Train or resume one adapter and atomically publish only completed artifacts."""

    _bundle, train_records, validation_records, manifest = _build_inputs(
        config,
        hardware,
        train_path,
        validation_path,
        sft_manifest_path,
        options,
    )
    final_dir, work_dir = _paths(config, options.specialist)
    with acquire_adapter_lock(config, options.specialist):
        pending, checkpoint = _inspect_artifacts(config, options, manifest)
        if not pending:
            completed = _load_manifest(final_dir / "manifest.json")
            return AdapterTrainingResult(
                specialist=options.specialist,
                adapter_dir=final_dir,
                manifest_path=final_dir / "manifest.json",
                manifest=completed,
            )
        if backend is None:
            raise ValueError("a trainer backend is required while adapter training is pending")
        if options.overwrite and work_dir.exists():
            shutil.rmtree(work_dir)
        if not work_dir.exists():
            work_dir.mkdir(parents=True)
            _write_manifest(work_dir / "run-manifest.json", manifest)

        result = backend.train(
            options.specialist,
            train_records,
            validation_records,
            work_dir,
            checkpoint,
        )
        if result.specialist is not options.specialist:
            raise RuntimeError("trainer backend returned an adapter for another specialist")
        if result.adapter_dir.parent.resolve() != work_dir.resolve():
            raise RuntimeError("trainer backend wrote the adapter outside its work directory")
        _validate_backend_adapter(result)
        completed = _completed_manifest(manifest, result)
        _write_manifest(result.adapter_dir / "manifest.json", completed)
        _publish_adapter(result.adapter_dir, final_dir, options.overwrite)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        _validate_completed_adapter(final_dir, completed)
        return AdapterTrainingResult(
            specialist=options.specialist,
            adapter_dir=final_dir,
            manifest_path=final_dir / "manifest.json",
            manifest=completed,
        )
