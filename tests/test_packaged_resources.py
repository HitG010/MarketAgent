from __future__ import annotations

from pathlib import Path

import pytest

from small_models_society.cli import main
from small_models_society.inference.config import load_inference_config
from small_models_society.inference.prompts import load_prompt_catalog
from small_models_society.resources import (
    DEFAULT_BENCHMARK_CONFIG,
    DEFAULT_INFERENCE_CONFIG,
    DEFAULT_PROMPT_PROFILES,
    DEFAULT_ROUTING_CONFIG,
    DEFAULT_TRAINING_CONFIG,
)
from small_models_society.routing.config import EXPECTED_ACTION_IDS, load_routing_config
from small_models_society.training.config import load_training_config

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_packaged_configs_match_research_configs() -> None:
    pairs = [
        (DEFAULT_BENCHMARK_CONFIG, REPOSITORY_ROOT / "configs" / "benchmark.yaml"),
        (DEFAULT_INFERENCE_CONFIG, REPOSITORY_ROOT / "configs" / "inference.yaml"),
        (DEFAULT_PROMPT_PROFILES, REPOSITORY_ROOT / "configs" / "prompt_profiles.yaml"),
        (DEFAULT_ROUTING_CONFIG, REPOSITORY_ROOT / "configs" / "routing.yaml"),
        (DEFAULT_TRAINING_CONFIG, REPOSITORY_ROOT / "configs" / "training.yaml"),
    ]

    for packaged, research in pairs:
        assert packaged.read_bytes() == research.read_bytes()


def test_inference_defaults_load_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_inference_config(DEFAULT_INFERENCE_CONFIG)
    catalog = load_prompt_catalog(DEFAULT_PROMPT_PROFILES)

    assert config.model.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert len(catalog.profiles) == 5


def test_training_defaults_load_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_training_config(DEFAULT_TRAINING_CONFIG)

    assert config.data.train_size_per_domain == 96
    assert config.data.validation_size_per_domain == 24


def test_routing_defaults_load_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_routing_config(DEFAULT_ROUTING_CONFIG)

    assert config.data.development_size_per_domain == 50
    assert config.data.test_size_per_domain == 50
    assert set(config.actions) == EXPECTED_ACTION_IDS


def test_inference_doctor_uses_packaged_default_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["inference", "doctor"])

    assert exit_code in {0, 1}
