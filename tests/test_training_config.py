from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from small_models_society.schemas import Domain
from small_models_society.training.config import (
    AttentionProjection,
    TrainingConfig,
    TrainingDevicePreference,
    load_training_config,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "training.yaml"


def _valid_config() -> dict[str, object]:
    return load_training_config(CONFIG_PATH).model_dump(mode="json")


def _section(config: dict[str, object], name: str) -> dict[str, object]:
    value = config[name]
    assert isinstance(value, dict)
    return value


def test_loads_controlled_lora_pilot() -> None:
    config = load_training_config(CONFIG_PATH)

    assert config.model.revision == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert config.model.device is TrainingDevicePreference.AUTO
    assert config.data.pilot_size_per_domain == 120
    assert config.data.train_size_per_domain == 96
    assert config.data.validation_size_per_domain == 24
    assert config.data.max_length == 512
    assert set(config.data.sources) == set(Domain)
    assert {source.split for source in config.data.sources.values()} == {"train"}
    assert config.lora.target_modules == tuple(AttentionProjection)
    assert config.sft.completion_only_loss is True
    assert config.sft.packing is False


def test_training_fingerprint_is_stable_and_complete() -> None:
    first = load_training_config(CONFIG_PATH)
    second = TrainingConfig.model_validate(first.model_dump(mode="json"))

    assert first.fingerprint() == second.fingerprint()
    assert len(first.fingerprint()) == 64

    changed = first.model_dump(mode="json")
    _section(changed, "sft")["learning_rate"] = 0.0002
    assert first.fingerprint() != TrainingConfig.model_validate(changed).fingerprint()


def test_training_fingerprint_excludes_artifact_locations() -> None:
    first = load_training_config(CONFIG_PATH)
    changed = first.model_dump(mode="json")
    data = _section(changed, "data")
    output = _section(changed, "output")
    data["output_dir"] = "D:/different/training-data"
    data["benchmark_path"] = "D:/different/benchmark.jsonl"
    data["benchmark_manifest_path"] = "D:/different/manifest.json"
    output["adapter_root"] = "D:/different/adapters"

    assert first.fingerprint() == TrainingConfig.model_validate(changed).fingerprint()


def test_training_fingerprint_excludes_cache_access_policy() -> None:
    online = load_training_config(CONFIG_PATH)
    offline = online.model_copy(
        update={"model": online.model.model_copy(update={"local_files_only": True})}
    )

    assert online.fingerprint() == offline.fingerprint()
    assert online.accepted_fingerprints() == offline.accepted_fingerprints()
    assert online.accepts_fingerprint(online._fingerprint(include_local_files_only=True))
    assert offline.accepts_fingerprint(offline._fingerprint(include_local_files_only=True))

    changed = online.model_copy(
        update={"sft": online.sft.model_copy(update={"learning_rate": 0.0002})}
    )
    assert not changed.accepts_fingerprint(online.fingerprint())


@pytest.mark.parametrize(
    ("section_name", "field", "value"),
    [
        ("model", "revision", "main"),
        ("model", "trust_remote_code", True),
        ("model", "use_safetensors", False),
        ("data", "max_length", 0),
        ("lora", "rank", 0),
        ("lora", "target_modules", ["query_key_value"]),
        ("sft", "per_device_train_batch_size", 2),
        ("sft", "gradient_checkpointing", False),
        ("sft", "completion_only_loss", False),
        ("sft", "packing", True),
        ("output", "save_safetensors", False),
        ("output", "atomic_publish", False),
    ],
)
def test_rejects_unsafe_or_uncontrolled_settings(
    section_name: str,
    field: str,
    value: object,
) -> None:
    config = _valid_config()
    _section(config, section_name)[field] = value

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(config)


def test_requires_exactly_every_domain() -> None:
    config = _valid_config()
    sources = _section(_section(config, "data"), "sources")
    del sources[Domain.KNOWLEDGE.value]

    with pytest.raises(ValidationError, match="exactly every domain"):
        TrainingConfig.model_validate(config)


def test_requires_source_training_splits() -> None:
    config = _valid_config()
    sources = _section(_section(config, "data"), "sources")
    math_source = sources[Domain.MATH.value]
    assert isinstance(math_source, dict)
    math_source["split"] = "test"

    with pytest.raises(ValidationError, match="training split"):
        TrainingConfig.model_validate(config)


def test_requires_train_and_validation_counts_to_fill_pilot() -> None:
    config = _valid_config()
    _section(config, "data")["validation_size_per_domain"] = 23

    with pytest.raises(ValidationError, match="must equal pilot_size_per_domain"):
        TrainingConfig.model_validate(config)


def test_rejects_duplicate_lora_targets() -> None:
    config = _valid_config()
    _section(config, "lora")["target_modules"] = ["q_proj", "q_proj"]

    with pytest.raises(ValidationError, match="must not contain duplicates"):
        TrainingConfig.model_validate(config)
