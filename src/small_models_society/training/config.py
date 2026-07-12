"""Strict configuration for reproducible LoRA specialist training."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from small_models_society.data.config import DatasetSource
from small_models_society.data.prepare import canonical_json
from small_models_society.schemas import Domain, StrictModel


class TrainingDevicePreference(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class TrainingDTypePreference(StrEnum):
    AUTO = "auto"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class TrainingModelConfig(StrictModel):
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    trust_remote_code: Literal[False] = False
    use_safetensors: Literal[True] = True
    device: TrainingDevicePreference = TrainingDevicePreference.AUTO
    dtype: TrainingDTypePreference = TrainingDTypePreference.AUTO
    local_files_only: bool = False


class TrainingDataConfig(StrictModel):
    seed: int = 42
    pilot_size_per_domain: int = Field(gt=1)
    train_size_per_domain: int = Field(gt=0)
    validation_size_per_domain: int = Field(gt=0)
    max_length: int = Field(gt=0, le=4096)
    output_dir: str = Field(min_length=1)
    sources: dict[Domain, DatasetSource]

    @model_validator(mode="after")
    def require_balanced_training_sources(self) -> Self:
        missing = set(Domain) - set(self.sources)
        extra = set(self.sources) - set(Domain)
        if missing or extra:
            raise ValueError(
                "sources must contain exactly every domain; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        non_training = {
            domain.value: source.split
            for domain, source in self.sources.items()
            if source.split != "train"
        }
        if non_training:
            raise ValueError(f"every specialist source must use its training split: {non_training}")
        if (
            self.train_size_per_domain + self.validation_size_per_domain
            != self.pilot_size_per_domain
        ):
            raise ValueError(
                "train_size_per_domain plus validation_size_per_domain must equal "
                "pilot_size_per_domain"
            )
        return self


class AttentionProjection(StrEnum):
    QUERY = "q_proj"
    KEY = "k_proj"
    VALUE = "v_proj"
    OUTPUT = "o_proj"


class LoraTrainingConfig(StrictModel):
    rank: int = Field(gt=0, le=256)
    alpha: int = Field(gt=0, le=512)
    dropout: float = Field(ge=0.0, lt=1.0)
    bias: Literal["none"] = "none"
    task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"
    target_modules: tuple[AttentionProjection, ...] = Field(min_length=1)
    init_lora_weights: Literal[True] = True

    @model_validator(mode="after")
    def reject_duplicate_targets(self) -> Self:
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("target_modules must not contain duplicates")
        return self


class SupervisedFineTuningConfig(StrictModel):
    per_device_train_batch_size: Literal[1] = 1
    per_device_eval_batch_size: Literal[1] = 1
    gradient_accumulation_steps: int = Field(gt=0)
    num_train_epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0, le=0.01)
    warmup_ratio: float = Field(ge=0.0, lt=1.0)
    gradient_checkpointing: Literal[True] = True
    completion_only_loss: Literal[True] = True
    packing: Literal[False] = False
    optim: Literal["adamw_torch"] = "adamw_torch"
    lr_scheduler_type: Literal["linear"] = "linear"
    eval_strategy: Literal["epoch"] = "epoch"
    save_strategy: Literal["epoch"] = "epoch"
    save_total_limit: int = Field(gt=0)
    logging_steps: int = Field(gt=0)
    report_to: Literal["none"] = "none"


class TrainingOutputConfig(StrictModel):
    adapter_root: str = Field(min_length=1)
    save_safetensors: Literal[True] = True
    atomic_publish: Literal[True] = True


class TrainingConfig(StrictModel):
    schema_version: Literal[1] = 1
    model: TrainingModelConfig
    data: TrainingDataConfig
    lora: LoraTrainingConfig
    sft: SupervisedFineTuningConfig
    output: TrainingOutputConfig

    def fingerprint(self) -> str:
        """Return a stable hash of every behavior-affecting training field."""

        values = self.model_dump(mode="json")
        for field in (
            "output_dir",
            "benchmark_path",
            "benchmark_manifest_path",
        ):
            values["data"].pop(field)
        values["output"].pop("adapter_root")
        payload = canonical_json(values).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_training_config(path: Path) -> TrainingConfig:
    """Load training YAML without executing custom YAML tags."""

    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return TrainingConfig.model_validate(value)