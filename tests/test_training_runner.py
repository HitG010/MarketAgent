from __future__ import annotations

import json
from pathlib import Path

import pytest

from small_models_society.data.prepare import canonical_json, sha256_bytes
from small_models_society.inference.contracts import ChatMessage
from small_models_society.schemas import Domain
from small_models_society.training.config import TrainingConfig, load_training_config
from small_models_society.training.contracts import SFTTrainingRecord, TrainingSplit
from small_models_society.training.hardware import TrainingHardwareReport
from small_models_society.training.runner import (
    AdapterRunOptions,
    TrainingResumeMismatchError,
    acquire_adapter_lock,
    inspect_adapter_training,
    load_sft_dataset_bundle,
    run_adapter_training,
)
from small_models_society.training.trainer import TrainerBackendResult

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "training.yaml"


def _config(tmp_path: Path) -> TrainingConfig:
    value = load_training_config(CONFIG_PATH).model_dump(mode="json")
    data = value["data"]
    output = value["output"]
    assert isinstance(data, dict)
    assert isinstance(output, dict)
    data["pilot_size_per_domain"] = 2
    data["train_size_per_domain"] = 1
    data["validation_size_per_domain"] = 1
    output["adapter_root"] = str(tmp_path / "adapters")
    return TrainingConfig.model_validate(value)


def _hardware() -> TrainingHardwareReport:
    return TrainingHardwareReport(
        ready=True,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        python_version="3.11.9",
        platform_system="Windows",
        platform_machine="AMD64",
        package_versions={"torch": "2.13.0", "peft": "0.19.1", "trl": "1.8.0"},
        selected_device="cpu",
        selected_dtype="float32",
        estimated_fit="cpu_debug_only",
        cpu_debug_only=True,
        cuda_available=False,
        mps_built=False,
        mps_available=False,
        mps_fallback_enabled=False,
        system_ram_gb=32,
        model_cache_path="C:/cache/model",
        model_cached=True,
        artifact_root="C:/artifacts",
        artifact_root_writable=True,
    )


def _record(domain: Domain, split: TrainingSplit) -> SFTTrainingRecord:
    suffix = f"{domain.value}-{split.value}"
    return SFTTrainingRecord(
        source_id=f"source::{suffix}",
        domain=domain,
        split=split,
        source_content_sha256=sha256_bytes(suffix.encode()),
        prompt=[
            ChatMessage(role="system", content="General prompt"),
            ChatMessage(role="user", content=f"Question for {domain.value}"),
        ],
        completion=[ChatMessage(role="assistant", content=f"Answer for {domain.value}")],
        prompt_tokens=10,
        completion_tokens=4,
    )


def _write_records(path: Path, records: list[SFTTrainingRecord]) -> str:
    content = (
        "\n".join(canonical_json(record.model_dump(mode="json")) for record in records) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha256_bytes(content)


def _sft_artifacts(
    tmp_path: Path,
    config: TrainingConfig,
) -> tuple[Path, Path, Path]:
    train_records = [_record(domain, TrainingSplit.TRAIN) for domain in Domain]
    validation_records = [_record(domain, TrainingSplit.VALIDATION) for domain in Domain]
    train_path = tmp_path / "sft" / "train.jsonl"
    validation_path = tmp_path / "sft" / "validation.jsonl"
    train_sha256 = _write_records(train_path, train_records)
    validation_sha256 = _write_records(validation_path, validation_records)
    manifest = {
        "schema_version": 1,
        "training_config_sha256": config.fingerprint(),
        "prompt_catalog_sha256": "a" * 64,
        "prompt_profile": "general",
        "completion_only_loss": True,
        "max_length": config.data.max_length,
        "tokenizer": {
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "class": "FakeTokenizer",
        },
        "row_count": 8,
        "files": {
            "train": {
                "path": train_path.name,
                "row_count": 4,
                "sha256": train_sha256,
            },
            "validation": {
                "path": validation_path.name,
                "row_count": 4,
                "sha256": validation_sha256,
            },
        },
    }
    manifest_path = tmp_path / "sft" / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return train_path, validation_path, manifest_path


def test_offline_training_accepts_legacy_online_sft_manifest(tmp_path: Path) -> None:
    online = _config(tmp_path)
    train_path, validation_path, manifest_path = _sft_artifacts(tmp_path, online)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_config_sha256"] = online._fingerprint(include_local_files_only=True)
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    offline = online.model_copy(
        update={"model": online.model.model_copy(update={"local_files_only": True})}
    )

    bundle = load_sft_dataset_bundle(
        offline,
        train_path,
        validation_path,
        manifest_path,
    )

    assert len(bundle.train_records) == 4
    assert len(bundle.validation_records) == 4


class FakeBackend:
    def __init__(self, *, fail: bool = False, hash_mismatch: bool = False) -> None:
        self.fail = fail
        self.hash_mismatch = hash_mismatch
        self.calls: list[tuple[Domain, Path | None]] = []

    def train(
        self,
        specialist: Domain,
        train_records: list[SFTTrainingRecord],
        validation_records: list[SFTTrainingRecord],
        work_dir: Path,
        resume_from_checkpoint: Path | None = None,
    ) -> TrainerBackendResult:
        del train_records, validation_records
        self.calls.append((specialist, resume_from_checkpoint))
        checkpoint = work_dir / "checkpoints" / "checkpoint-7"
        checkpoint.mkdir(parents=True, exist_ok=True)
        if self.fail:
            raise RuntimeError("injected interruption")
        adapter_dir = work_dir / "adapter"
        adapter_dir.mkdir()
        weights = b"adapter-weights"
        (adapter_dir / "adapter_model.safetensors").write_bytes(weights)
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        return TrainerBackendResult(
            specialist=specialist,
            adapter_dir=adapter_dir,
            adapter_sha256=("f" * 64 if self.hash_mismatch else sha256_bytes(weights)),
            train_metrics={"train_loss": 1.0},
            eval_metrics={"eval_loss": 0.5},
            trainable_parameters=16,
            total_parameters=1_016,
            duration_seconds=2.5,
            resumed_from_checkpoint=(
                str(resume_from_checkpoint) if resume_from_checkpoint else None
            ),
            package_versions={"peft": "test"},
        )


def _arguments(
    tmp_path: Path,
) -> tuple[TrainingConfig, Path, Path, Path, AdapterRunOptions]:
    config = _config(tmp_path)
    train_path, validation_path, manifest_path = _sft_artifacts(tmp_path, config)
    options = AdapterRunOptions(specialist=Domain.MATH)
    return config, train_path, validation_path, manifest_path, options


def test_loads_verified_balanced_sft_bundle(tmp_path: Path) -> None:
    config, train_path, validation_path, manifest_path, _options = _arguments(tmp_path)

    bundle = load_sft_dataset_bundle(config, train_path, validation_path, manifest_path)

    assert len(bundle.train_records) == 4
    assert len(bundle.validation_records) == 4
    assert bundle.prompt_catalog_fingerprint == "a" * 64


def test_trains_and_atomically_publishes_completed_adapter(tmp_path: Path) -> None:
    config, train_path, validation_path, manifest_path, options = _arguments(tmp_path)
    backend = FakeBackend()

    result = run_adapter_training(
        config,
        _hardware(),
        train_path,
        validation_path,
        manifest_path,
        backend,
        options,
    )

    assert result.adapter_dir == Path(config.output.adapter_root) / "math"
    assert result.manifest.status == "completed"
    assert result.manifest.adapter_sha256 == sha256_bytes(b"adapter-weights")
    assert result.manifest.trainable_parameters == 16
    assert result.manifest.train_source_ids == ["source::math-train"]
    assert result.manifest_path.is_file()
    assert not (Path(config.output.adapter_root) / ".math.work").exists()
    assert backend.calls == [(Domain.MATH, None)]


def test_collision_is_rejected_during_inspection(tmp_path: Path) -> None:
    config, train_path, validation_path, manifest_path, options = _arguments(tmp_path)
    final_dir = Path(config.output.adapter_root) / "math"
    final_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="resume or overwrite"):
        inspect_adapter_training(
            config,
            _hardware(),
            train_path,
            validation_path,
            manifest_path,
            options,
        )


def test_interrupted_run_resumes_latest_checkpoint(tmp_path: Path) -> None:
    config, train_path, validation_path, manifest_path, options = _arguments(tmp_path)

    with pytest.raises(RuntimeError, match="injected interruption"):
        run_adapter_training(
            config,
            _hardware(),
            train_path,
            validation_path,
            manifest_path,
            FakeBackend(fail=True),
            options,
        )

    work_dir = Path(config.output.adapter_root) / ".math.work"
    assert (work_dir / "run-manifest.json").is_file()
    assert (work_dir / "checkpoints" / "checkpoint-7").is_dir()

    backend = FakeBackend()
    resumed = run_adapter_training(
        config,
        _hardware(),
        train_path,
        validation_path,
        manifest_path,
        backend,
        AdapterRunOptions(specialist=Domain.MATH, resume=True),
    )

    assert resumed.manifest.status == "completed"
    assert backend.calls[0][1] == work_dir / "checkpoints" / "checkpoint-7"
    assert resumed.manifest.resumed_from_checkpoint == str(backend.calls[0][1])


def test_completed_resume_does_not_require_backend(tmp_path: Path) -> None:
    config, train_path, validation_path, manifest_path, options = _arguments(tmp_path)
    first = run_adapter_training(
        config,
        _hardware(),
        train_path,
        validation_path,
        manifest_path,
        FakeBackend(),
        options,
    )
    resume_options = AdapterRunOptions(specialist=Domain.MATH, resume=True)

    plan = inspect_adapter_training(
        config,
        _hardware(),
        train_path,
        validation_path,
        manifest_path,
        resume_options,
    )
    resumed = run_adapter_training(
        config,
        _hardware(),
        train_path,
        validation_path,
        manifest_path,
        None,
        resume_options,
    )

    assert plan.pending is False
    assert resumed.manifest == first.manifest


def test_resume_rejects_changed_sft_manifest(tmp_path: Path) -> None:
    config, train_path, validation_path, manifest_path, options = _arguments(tmp_path)
    with pytest.raises(RuntimeError):
        run_adapter_training(
            config,
            _hardware(),
            train_path,
            validation_path,
            manifest_path,
            FakeBackend(fail=True),
            options,
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompt_catalog_sha256"] = "b" * 64
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    with pytest.raises(TrainingResumeMismatchError, match="fingerprint"):
        inspect_adapter_training(
            config,
            _hardware(),
            train_path,
            validation_path,
            manifest_path,
            AdapterRunOptions(specialist=Domain.MATH, resume=True),
        )


def test_concurrent_training_lock_is_rejected(tmp_path: Path) -> None:
    config, train_path, validation_path, manifest_path, options = _arguments(tmp_path)

    with (
        acquire_adapter_lock(config, Domain.MATH),
        pytest.raises(FileExistsError, match="another process"),
    ):
        run_adapter_training(
            config,
            _hardware(),
            train_path,
            validation_path,
            manifest_path,
            FakeBackend(),
            options,
        )


def test_failed_overwrite_preserves_completed_adapter(tmp_path: Path) -> None:
    config, train_path, validation_path, manifest_path, options = _arguments(tmp_path)
    completed = run_adapter_training(
        config,
        _hardware(),
        train_path,
        validation_path,
        manifest_path,
        FakeBackend(),
        options,
    )
    original_weights = (completed.adapter_dir / "adapter_model.safetensors").read_bytes()

    with pytest.raises(RuntimeError, match="injected interruption"):
        run_adapter_training(
            config,
            _hardware(),
            train_path,
            validation_path,
            manifest_path,
            FakeBackend(fail=True),
            AdapterRunOptions(specialist=Domain.MATH, overwrite=True),
        )

    assert (completed.adapter_dir / "adapter_model.safetensors").read_bytes() == original_weights


def test_hash_mismatch_does_not_replace_completed_adapter(tmp_path: Path) -> None:
    config, train_path, validation_path, manifest_path, options = _arguments(tmp_path)
    completed = run_adapter_training(
        config,
        _hardware(),
        train_path,
        validation_path,
        manifest_path,
        FakeBackend(),
        options,
    )
    original_weights = (completed.adapter_dir / "adapter_model.safetensors").read_bytes()

    with pytest.raises(RuntimeError, match="hash does not match"):
        run_adapter_training(
            config,
            _hardware(),
            train_path,
            validation_path,
            manifest_path,
            FakeBackend(hash_mismatch=True),
            AdapterRunOptions(specialist=Domain.MATH, overwrite=True),
        )

    assert (completed.adapter_dir / "adapter_model.safetensors").read_bytes() == original_weights


def test_completed_resume_rejects_tampered_adapter(tmp_path: Path) -> None:
    config, train_path, validation_path, manifest_path, options = _arguments(tmp_path)
    completed = run_adapter_training(
        config,
        _hardware(),
        train_path,
        validation_path,
        manifest_path,
        FakeBackend(),
        options,
    )
    (completed.adapter_dir / "adapter_model.safetensors").write_bytes(b"tampered")

    with pytest.raises(TrainingResumeMismatchError, match="hash does not match"):
        inspect_adapter_training(
            config,
            _hardware(),
            train_path,
            validation_path,
            manifest_path,
            AdapterRunOptions(specialist=Domain.MATH, resume=True),
        )
