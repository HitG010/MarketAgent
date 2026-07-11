"""Benchmark configuration contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, model_validator

from small_models_society.schemas import Domain, StrictModel


class DatasetSource(StrictModel):
    dataset: str = Field(min_length=1)
    config: str = Field(min_length=1)
    split: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class BenchmarkConfig(StrictModel):
    schema_version: int = Field(default=1, ge=1)
    seed: int = 42
    sample_per_domain: int = Field(gt=0)
    output_dir: str = Field(min_length=1)
    sources: dict[Domain, DatasetSource]

    @model_validator(mode="after")
    def require_every_domain(self) -> Self:
        missing = set(Domain) - set(self.sources)
        extra = set(self.sources) - set(Domain)
        if missing or extra:
            raise ValueError(
                "sources must contain exactly every domain; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return self


def load_config(path: Path) -> BenchmarkConfig:
    """Load and validate a benchmark YAML file without executing custom YAML tags."""

    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return BenchmarkConfig.model_validate(value)
