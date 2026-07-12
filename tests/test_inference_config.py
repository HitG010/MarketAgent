from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from small_models_society.inference.config import (
    DevicePreference,
    DTypePreference,
    InferenceConfig,
    load_inference_config,
)
from small_models_society.schemas import Domain

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "inference.yaml"


def _valid_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "a" * 40,
            "trust_remote_code": False,
            "use_safetensors": True,
            "device": "auto",
            "dtype": "auto",
            "local_files_only": False,
        },
        "generation": {
            "seed": 42,
            "batch_size": 1,
            "max_input_tokens": 4096,
            "max_new_tokens": {domain.value: 128 for domain in Domain},
            "do_sample": False,
            "checkpoint_interval": 5,
        },
    }


def test_loads_pinned_default_configuration() -> None:
    config = load_inference_config(CONFIG_PATH)

    assert config.model.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.model.revision == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert config.model.device is DevicePreference.AUTO
    assert config.model.dtype is DTypePreference.AUTO
    assert config.generation.max_new_tokens[Domain.CODE] == 512


def test_configuration_fingerprint_is_stable() -> None:
    first = InferenceConfig.model_validate(_valid_config())
    second = InferenceConfig.model_validate(first.model_dump(mode="json"))

    assert first.fingerprint() == second.fingerprint()
    assert len(first.fingerprint()) == 64


def test_configuration_fingerprint_changes_with_generation_budget() -> None:
    first = InferenceConfig.model_validate(_valid_config())
    changed_value = _valid_config()
    generation = changed_value["generation"]
    assert isinstance(generation, dict)
    generation["max_input_tokens"] = 2048
    second = InferenceConfig.model_validate(changed_value)

    assert first.fingerprint() != second.fingerprint()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model", "revision"), "main"),
        (("model", "device"), "mps"),
        (("model", "dtype"), "float8"),
        (("model", "trust_remote_code"), True),
        (("model", "use_safetensors"), False),
        (("generation", "batch_size"), 2),
        (("generation", "do_sample"), True),
        (("generation", "max_input_tokens"), 0),
        (("generation", "checkpoint_interval"), 0),
    ],
)
def test_rejects_unsafe_or_nondeterministic_settings(path: tuple[str, str], value: object) -> None:
    config = _valid_config()
    section = config[path[0]]
    assert isinstance(section, dict)
    section[path[1]] = value

    with pytest.raises(ValidationError):
        InferenceConfig.model_validate(config)


def test_requires_a_generation_budget_for_every_domain() -> None:
    config = _valid_config()
    generation = config["generation"]
    assert isinstance(generation, dict)
    budgets = generation["max_new_tokens"]
    assert isinstance(budgets, dict)
    del budgets[Domain.KNOWLEDGE.value]

    with pytest.raises(ValidationError, match="exactly every domain"):
        InferenceConfig.model_validate(config)


def test_rejects_out_of_range_generation_budget() -> None:
    config = _valid_config()
    generation = config["generation"]
    assert isinstance(generation, dict)
    budgets = generation["max_new_tokens"]
    assert isinstance(budgets, dict)
    budgets[Domain.CODE.value] = 0

    with pytest.raises(ValidationError, match="between 1 and 4096"):
        InferenceConfig.model_validate(config)
