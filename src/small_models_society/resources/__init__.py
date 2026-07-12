"""Filesystem paths for default configuration packaged with the distribution."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def packaged_config_path(filename: str) -> Path:
    resource = files(__name__).joinpath(filename)
    path = Path(str(resource))
    if not path.is_file():
        raise FileNotFoundError(f"packaged configuration is unavailable: {filename}")
    return path


DEFAULT_BENCHMARK_CONFIG = packaged_config_path("benchmark.yaml")
DEFAULT_INFERENCE_CONFIG = packaged_config_path("inference.yaml")
DEFAULT_PROMPT_PROFILES = packaged_config_path("prompt_profiles.yaml")
DEFAULT_TRAINING_CONFIG = packaged_config_path("training.yaml")

__all__ = [
    "DEFAULT_BENCHMARK_CONFIG",
    "DEFAULT_INFERENCE_CONFIG",
    "DEFAULT_PROMPT_PROFILES",
    "DEFAULT_TRAINING_CONFIG",
    "packaged_config_path",
]
