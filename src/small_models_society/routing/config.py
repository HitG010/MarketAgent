"""Strict configuration for candidate workflow research."""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, model_validator

from small_models_society.data.config import DatasetSource
from small_models_society.data.prepare import canonical_json
from small_models_society.schemas import Domain, StrictModel

EXPECTED_ACTION_IDS = frozenset(
    {
        "tool.calculator.v1",
        "local.qwen-base.v1",
        "local.qwen-lora-math.v1",
        "local.qwen-lora-code.v1",
        "local.qwen-lora-logic.v1",
        "local.qwen-lora-knowledge.v1",
        "rag.bm25-qwen-base.v1",
        "remote.strong-replay.reference.v1",
    }
)
_ACTION_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class ActionKind(StrEnum):
    TOOL = "tool"
    LOCAL_MODEL = "local_model"
    RETRIEVAL = "retrieval"
    STRONG_REPLAY = "strong_replay"


class ToolActionConfig(StrictModel):
    kind: Literal[ActionKind.TOOL] = ActionKind.TOOL
    enabled: Literal[True] = True
    tool_id: Literal["calculator.v1"] = "calculator.v1"


class LocalModelActionConfig(StrictModel):
    kind: Literal[ActionKind.LOCAL_MODEL] = ActionKind.LOCAL_MODEL
    enabled: Literal[True] = True
    adapter: Domain | None = None
    approved: bool
    max_new_tokens: int = Field(gt=0, le=4096)


class RetrievalActionConfig(StrictModel):
    kind: Literal[ActionKind.RETRIEVAL] = ActionKind.RETRIEVAL
    enabled: Literal[True] = True
    retriever_id: Literal["bm25.rank-bm25-0.2.2"] = "bm25.rank-bm25-0.2.2"
    corpus_id: Literal["hotpotqa.routing.v1"] = "hotpotqa.routing.v1"
    generator_action_id: Literal["local.qwen-base.v1"] = "local.qwen-base.v1"
    max_new_tokens: int = Field(gt=0, le=4096)


class StrongReplayActionConfig(StrictModel):
    kind: Literal[ActionKind.STRONG_REPLAY] = ActionKind.STRONG_REPLAY
    enabled: Literal[True] = True
    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    pricing_schedule_id: str = Field(min_length=1)
    max_new_tokens: int = Field(gt=0, le=4096)


ActionConfig = Annotated[
    ToolActionConfig | LocalModelActionConfig | RetrievalActionConfig | StrongReplayActionConfig,
    Field(discriminator="kind"),
]


class RoutingModelConfig(StrictModel):
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    prompt_profile: Literal["general"] = "general"
    adapter_root: str = Field(min_length=1)


class RoutingDataConfig(StrictModel):
    development_size_per_domain: int = Field(gt=0)
    test_size_per_domain: int = Field(gt=0)
    local_files_only: bool = False
    output_dir: str = Field(min_length=1)
    benchmark_path: str = Field(min_length=1)
    benchmark_manifest_path: str = Field(min_length=1)
    training_train_path: str = Field(min_length=1)
    training_validation_path: str = Field(min_length=1)
    training_manifest_path: str = Field(min_length=1)
    sources: dict[Domain, DatasetSource]

    @model_validator(mode="after")
    def require_evaluation_sources_for_every_domain(self) -> Self:
        missing = set(Domain) - set(self.sources)
        extra = set(self.sources) - set(Domain)
        if missing or extra:
            raise ValueError(
                "sources must contain exactly every domain; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        expected_splits = {
            Domain.MATH: "test",
            Domain.CODE: "test",
            Domain.LOGIC: "test",
            Domain.KNOWLEDGE: "validation",
        }
        invalid = {
            domain.value: source.split
            for domain, source in self.sources.items()
            if source.split != expected_splits[domain]
        }
        if invalid:
            raise ValueError(f"routing sources use unexpected evaluation splits: {invalid}")
        return self


class RetrievalConfig(StrictModel):
    library: Literal["rank_bm25"] = "rank_bm25"
    library_version: Literal["0.2.2"] = "0.2.2"
    algorithm: Literal["BM25Okapi"] = "BM25Okapi"
    tokenizer: Literal["unicode-lower-regex-v1"] = "unicode-lower-regex-v1"
    corpus_id: Literal["hotpotqa.routing.v1"] = "hotpotqa.routing.v1"
    top_k_values: tuple[int, ...] = Field(min_length=1)
    generation_top_k: int = Field(gt=0)
    stable_tie_break: Literal["document_id"] = "document_id"

    @model_validator(mode="after")
    def require_stable_top_k_sweep(self) -> Self:
        if any(value <= 0 for value in self.top_k_values):
            raise ValueError("top_k_values must be positive")
        if tuple(sorted(set(self.top_k_values))) != self.top_k_values:
            raise ValueError("top_k_values must be unique and sorted")
        if self.generation_top_k not in self.top_k_values:
            raise ValueError("generation_top_k must be included in top_k_values")
        return self


class CalculatorOperator(StrEnum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"


class CalculatorConfig(StrictModel):
    tool_id: Literal["calculator.v1"] = "calculator.v1"
    operators: tuple[CalculatorOperator, ...] = Field(min_length=1)
    max_expression_length: int = Field(gt=0, le=4096)
    max_ast_depth: int = Field(gt=0, le=64)
    max_operations: int = Field(gt=0, le=256)
    max_abs_value: float = Field(gt=0)
    max_exponent: int = Field(gt=0, le=100)

    @model_validator(mode="after")
    def reject_duplicate_operators(self) -> Self:
        if len(set(self.operators)) != len(self.operators):
            raise ValueError("calculator operators must not contain duplicates")
        return self


class DataClassification(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class PolicyDefaultsConfig(StrictModel):
    data_classification: DataClassification
    network_allowed: bool
    allowed_corpus_ids: tuple[str, ...]
    allowed_tool_ids: tuple[str, ...]
    required_quality: float = Field(ge=0, le=1)
    allow_unknown_output_safety: Literal[False] = False

    @model_validator(mode="after")
    def reject_duplicate_allowlist_entries(self) -> Self:
        if len(set(self.allowed_corpus_ids)) != len(self.allowed_corpus_ids):
            raise ValueError("allowed_corpus_ids must not contain duplicates")
        if len(set(self.allowed_tool_ids)) != len(self.allowed_tool_ids):
            raise ValueError("allowed_tool_ids must not contain duplicates")
        return self


class ObjectiveMetric(StrEnum):
    PROVIDER_FEE = "provider_fee_usd"
    WALL_LATENCY = "wall_latency_ms"
    ENERGY = "energy_joules"


class AnalysisConfig(StrictModel):
    quality_floors: tuple[float, ...] = Field(min_length=2)
    objective_metrics: tuple[ObjectiveMetric, ...] = Field(min_length=1)
    bootstrap_resamples: int = Field(gt=0)
    confidence_level: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def require_quality_sweep_and_objectives(self) -> Self:
        if any(not 0 <= value <= 1 for value in self.quality_floors):
            raise ValueError("quality_floors must be between 0 and 1")
        if tuple(sorted(set(self.quality_floors))) != self.quality_floors:
            raise ValueError("quality_floors must be unique and sorted")
        if self.quality_floors[0] != 0 or self.quality_floors[-1] != 1:
            raise ValueError("quality_floors must include 0 and 1")
        if tuple(ObjectiveMetric) != self.objective_metrics:
            raise ValueError("objective_metrics must include provider fee, latency, and energy")
        return self


class ReplayConfig(StrictModel):
    directory: str = Field(min_length=1)
    pricing_path: str = Field(min_length=1)
    require_verified_safety_status: Literal[True] = True
    allow_network_calls: Literal[False] = False


class TelemetryConfig(StrictModel):
    currency: Literal["USD"] = "USD"
    energy_required_for_primary_analysis: Literal[False] = False
    allow_estimated_energy_in_primary_analysis: Literal[False] = False


class RoutingOutputConfig(StrictModel):
    report_root: str = Field(min_length=1)


class RoutingConfig(StrictModel):
    schema_version: Literal[1] = 1
    seed: int = 42
    model: RoutingModelConfig
    data: RoutingDataConfig
    actions: dict[str, ActionConfig]
    retrieval: RetrievalConfig
    calculator: CalculatorConfig
    policy_defaults: PolicyDefaultsConfig
    analysis: AnalysisConfig
    replay: ReplayConfig
    telemetry: TelemetryConfig
    output: RoutingOutputConfig

    @model_validator(mode="after")
    def require_exact_candidate_action_set(self) -> Self:
        actual = set(self.actions)
        missing = EXPECTED_ACTION_IDS - actual
        extra = actual - EXPECTED_ACTION_IDS
        if missing or extra:
            raise ValueError(
                "actions must contain exactly the Phase 4 candidate set; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        invalid_ids = sorted(
            action_id for action_id in actual if not _ACTION_ID.fullmatch(action_id)
        )
        if invalid_ids:
            raise ValueError(f"invalid action IDs: {invalid_ids}")

        base = self.actions["local.qwen-base.v1"]
        if not isinstance(base, LocalModelActionConfig) or base.adapter is not None:
            raise ValueError("local.qwen-base.v1 must be the adapter-free local action")
        if not base.approved:
            raise ValueError("local.qwen-base.v1 must be approved")
        for domain in Domain:
            action_id = f"local.qwen-lora-{domain.value}.v1"
            action = self.actions[action_id]
            if not isinstance(action, LocalModelActionConfig) or action.adapter is not domain:
                raise ValueError(f"{action_id} must reference the matching adapter")

        rag = self.actions["rag.bm25-qwen-base.v1"]
        if not isinstance(rag, RetrievalActionConfig):
            raise ValueError("rag.bm25-qwen-base.v1 must be a retrieval action")
        if rag.corpus_id != self.retrieval.corpus_id:
            raise ValueError("RAG action corpus must match retrieval configuration")
        if rag.generator_action_id != "local.qwen-base.v1":
            raise ValueError("RAG must use the local base generator in Phase 4")
        return self

    def fingerprint(self) -> str:
        """Hash behavior-affecting fields while excluding relocatable paths."""

        values = self.model_dump(mode="json")
        values["model"].pop("adapter_root")
        for field in (
            "output_dir",
            "benchmark_path",
            "benchmark_manifest_path",
            "training_train_path",
            "training_validation_path",
            "training_manifest_path",
        ):
            values["data"].pop(field)
        values["replay"].pop("directory")
        values["replay"].pop("pricing_path")
        values["output"].pop("report_root")
        return hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()


def load_routing_config(path: Path) -> RoutingConfig:
    """Load routing YAML without executing custom YAML tags."""

    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return RoutingConfig.model_validate(value)
