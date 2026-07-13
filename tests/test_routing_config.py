from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from small_models_society.routing.config import (
    EXPECTED_ACTION_IDS,
    LocalModelActionConfig,
    ObjectiveMetric,
    RoutingConfig,
    load_routing_config,
)
from small_models_society.schemas import Domain

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "routing.yaml"
LOCK_PATH = ROOT / "requirements-routing.lock"


def _valid_config() -> dict[str, object]:
    return load_routing_config(CONFIG_PATH).model_dump(mode="json")


def _section(config: dict[str, object], name: str) -> dict[str, object]:
    value = config[name]
    assert isinstance(value, dict)
    return value


def test_loads_candidate_workflow_research_defaults() -> None:
    config = load_routing_config(CONFIG_PATH)

    assert config.seed == 42
    assert config.model.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.model.revision == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert config.data.development_size_per_domain == 50
    assert config.data.test_size_per_domain == 50
    assert config.data.local_files_only is False
    assert set(config.data.sources) == set(Domain)
    assert set(config.actions) == EXPECTED_ACTION_IDS
    assert config.retrieval.top_k_values == (1, 3, 5, 10)
    assert config.retrieval.generation_top_k == 5
    assert config.calculator.max_expression_length == 256
    assert config.analysis.quality_floors == (0.0, 0.5, 0.8, 1.0)
    assert config.analysis.objective_metrics == tuple(ObjectiveMetric)
    assert config.replay.allow_network_calls is False
    assert config.telemetry.energy_required_for_primary_analysis is False
    assert config.policy_defaults.required_quality == 0.8

    for domain in Domain:
        action = config.actions[f"local.qwen-lora-{domain.value}.v1"]
        assert isinstance(action, LocalModelActionConfig)
        assert action.adapter is domain
        assert action.approved is False
    assert config.actions["local.qwen-base.v1"].max_new_tokens == 512
    assert config.actions["rag.bm25-qwen-base.v1"].max_new_tokens == 128
    assert config.actions["remote.strong-replay.reference.v1"].max_new_tokens == 512


def test_default_source_revisions_and_splits_match_held_out_sources() -> None:
    config = load_routing_config(CONFIG_PATH)
    expected = {
        Domain.MATH: ("test", "740312add88f781978c0658806c59bc2815b9866"),
        Domain.CODE: ("test", "4bb6404fdc6cacfda99d4ac4205087b89d32030c"),
        Domain.LOGIC: ("test", "210d026faf9955653af8916fad021475a3f00453"),
        Domain.KNOWLEDGE: ("validation", "1908d6afbbead072334abe2965f91bd2709910ab"),
    }

    assert {
        domain: (source.split, source.revision) for domain, source in config.data.sources.items()
    } == expected


def test_routing_fingerprint_is_stable_and_behavior_complete() -> None:
    first = load_routing_config(CONFIG_PATH)
    second = RoutingConfig.model_validate(first.model_dump(mode="json"))

    assert first.fingerprint() == second.fingerprint()
    assert len(first.fingerprint()) == 64

    changed = first.model_dump(mode="json")
    _section(changed, "retrieval")["generation_top_k"] = 3
    assert first.fingerprint() != RoutingConfig.model_validate(changed).fingerprint()


def test_routing_fingerprint_excludes_relocatable_paths() -> None:
    first = load_routing_config(CONFIG_PATH)
    changed = first.model_dump(mode="json")
    _section(changed, "model")["adapter_root"] = "D:/adapters"
    data = _section(changed, "data")
    for field in (
        "output_dir",
        "benchmark_path",
        "benchmark_manifest_path",
        "training_train_path",
        "training_validation_path",
        "training_manifest_path",
    ):
        data[field] = f"D:/{field}"
    _section(changed, "replay")["directory"] = "D:/replay"
    _section(changed, "replay")["pricing_path"] = "D:/pricing.json"
    _section(changed, "output")["report_root"] = "D:/reports"

    assert first.fingerprint() == RoutingConfig.model_validate(changed).fingerprint()


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("model", "revision", "main", None),
        ("retrieval", "top_k_values", [1, 5, 3], "unique and sorted"),
        ("retrieval", "generation_top_k", 2, "included"),
        ("analysis", "quality_floors", [0.0, 0.8, 0.5, 1.0], "unique and sorted"),
        ("analysis", "quality_floors", [0.2, 0.8], "include 0 and 1"),
        ("replay", "allow_network_calls", True, None),
        ("telemetry", "energy_required_for_primary_analysis", True, None),
    ],
)
def test_rejects_unsafe_or_ambiguous_research_settings(
    section: str,
    field: str,
    value: object,
    message: str | None,
) -> None:
    config = _valid_config()
    _section(config, section)[field] = value

    with pytest.raises(ValidationError, match=message):
        RoutingConfig.model_validate(config)


def test_requires_exact_candidate_action_set() -> None:
    config = _valid_config()
    actions = _section(config, "actions")
    del actions["tool.calculator.v1"]

    with pytest.raises(ValidationError, match="exactly the Phase 4 candidate set"):
        RoutingConfig.model_validate(config)


def test_requires_matching_lora_adapter_identity() -> None:
    config = _valid_config()
    actions = _section(config, "actions")
    math_action = actions["local.qwen-lora-math.v1"]
    assert isinstance(math_action, dict)
    math_action["adapter"] = "code"

    with pytest.raises(ValidationError, match="matching adapter"):
        RoutingConfig.model_validate(config)


def test_requires_exactly_every_domain_source() -> None:
    config = _valid_config()
    sources = _section(_section(config, "data"), "sources")
    del sources[Domain.KNOWLEDGE.value]

    with pytest.raises(ValidationError, match="exactly every domain"):
        RoutingConfig.model_validate(config)


def test_routing_lock_is_python_311_and_pins_bm25() -> None:
    content = LOCK_PATH.read_text(encoding="utf-8")

    assert "pip-compile with Python 3.11" in content
    assert "rank-bm25==0.2.2" in content
