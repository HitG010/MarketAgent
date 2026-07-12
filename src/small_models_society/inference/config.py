"""Strict configuration for deterministic local model inference."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from small_models_society.data.prepare import canonical_json
from small_models_society.schemas import Domain, StrictModel


class DevicePreference(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class DTypePreference(StrEnum):
    AUTO = "auto"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class ModelConfig(StrictModel):
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    trust_remote_code: Literal[False] = False
    use_safetensors: Literal[True] = True
    device: DevicePreference = DevicePreference.AUTO
    dtype: DTypePreference = DTypePreference.AUTO
    local_files_only: bool = False


class GenerationConfig(StrictModel):
    seed: int = 42
    batch_size: Literal[1] = 1
    max_input_tokens: int = Field(gt=0, le=32_768)
    max_new_tokens: dict[Domain, int]
    do_sample: Literal[False] = False
    checkpoint_interval: int = Field(gt=0)

    @model_validator(mode="after")
    def require_every_domain_budget(self) -> Self:
        missing = set(Domain) - set(self.max_new_tokens)
        extra = set(self.max_new_tokens) - set(Domain)
        if missing or extra:
            raise ValueError(
                "max_new_tokens must contain exactly every domain; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        invalid = {
            domain.value: tokens
            for domain, tokens in self.max_new_tokens.items()
            if not 0 < tokens <= 4096
        }
        if invalid:
            raise ValueError(f"max_new_tokens values must be between 1 and 4096: {invalid}")
        return self


class InferenceConfig(StrictModel):
    schema_version: Literal[1] = 1
    model: ModelConfig
    generation: GenerationConfig

    def fingerprint(self) -> str:
        """Return a stable hash of every behavior-affecting configuration field."""

        payload = canonical_json(self.model_dump(mode="json")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_inference_config(path: Path) -> InferenceConfig:
    """Load inference YAML without executing custom YAML tags."""

    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return InferenceConfig.model_validate(value)
