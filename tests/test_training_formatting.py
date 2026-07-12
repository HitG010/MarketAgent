from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from small_models_society.data.config import DatasetSource
from small_models_society.data.loaders import SourceRow
from small_models_society.data.prepare import load_benchmark
from small_models_society.inference.prompts import PromptProfileName, load_prompt_catalog
from small_models_society.schemas import Domain
from small_models_society.training.config import TrainingConfig, load_training_config
from small_models_society.training.contracts import SourceTrainingRecord, TrainingSplit
from small_models_society.training.formatting import (
    IGNORE_INDEX,
    SFTLengthError,
    build_sft_eligibility_filter,
    format_source_record,
    load_sft_training_records,
    prepare_sft_data,
    render_completion,
    tokenize_prompt_completion,
)
from small_models_society.training.prepare import (
    BenchmarkLeakageIndex,
    normalized_content_sha256,
    prepare_training_data,
)

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "training.yaml"
PROMPT_PATH = ROOT / "configs" / "prompt_profiles.yaml"
FIXTURE_BENCHMARK = Path(__file__).parent / "fixtures" / "benchmark.jsonl"


class FakeTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        text = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in conversation
        )
        if kwargs.get("add_generation_prompt"):
            text += "<assistant>"
        return text

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        assert kwargs == {"add_special_tokens": False}
        return {"input_ids": [ord(character) for character in text]}


def _source_record(index: int, split: TrainingSplit = TrainingSplit.TRAIN) -> SourceTrainingRecord:
    example = load_benchmark(FIXTURE_BENCHMARK)[index]
    return SourceTrainingRecord(
        source_id=f"fixture::{example.id}",
        domain=example.domain,
        split=split,
        content_sha256=normalized_content_sha256(example),
        example=example,
    )


def _small_config() -> TrainingConfig:
    value = load_training_config(CONFIG_PATH).model_dump(mode="json")
    data = value["data"]
    assert isinstance(data, dict)
    data["pilot_size_per_domain"] = 2
    data["train_size_per_domain"] = 1
    data["validation_size_per_domain"] = 1
    data["max_length"] = 4096
    return TrainingConfig.model_validate(value)


def _domain_for_source(source: DatasetSource) -> Domain:
    return {
        "openai/gsm8k": Domain.MATH,
        "google-research-datasets/mbpp": Domain.CODE,
        "allenai/ai2_arc": Domain.LOGIC,
        "hotpotqa/hotpot_qa": Domain.KNOWLEDGE,
    }[source.dataset]


def _rows(domain: Domain) -> list[dict[str, Any]]:
    if domain is Domain.MATH:
        return [
            {"question": f"What is {index} + 1?", "answer": f"work\n#### {index + 1}"}
            for index in range(3)
        ]
    if domain is Domain.CODE:
        return [
            {
                "source_file": "fixture.jsonl",
                "task_id": index,
                "prompt": f"Return {index}.",
                "code": f"def answer():\n    return {index}",
                "test_imports": [],
                "test_list": [f"assert answer() == {index}"],
            }
            for index in range(3)
        ]
    if domain is Domain.LOGIC:
        return [
            {
                "id": str(index),
                "question": f"Select A for item {index}.",
                "choices": {"label": ["A", "B"], "text": ["yes", "no"]},
                "answerKey": "A",
            }
            for index in range(3)
        ]
    return [
        {
            "id": str(index),
            "question": f"Who is person {index}?",
            "answer": f"Name {index}",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": {"title": ["People"], "sent_id": [0]},
            "context": {
                "title": ["People"],
                "sentences": [[f"Person {index} is Name {index}."]],
            },
        }
        for index in range(3)
    ]


def _fixture_rows(source: DatasetSource) -> Iterable[SourceRow]:
    return _rows(_domain_for_source(source))


def test_formats_domain_completions_without_reference_leakage() -> None:
    catalog = load_prompt_catalog(PROMPT_PATH)
    tokenizer = FakeTokenizer()

    records = [
        format_source_record(_source_record(index), catalog, tokenizer, 4096) for index in range(4)
    ]

    assert records[0].completion[0].content.endswith("Final answer: 10")
    assert records[1].completion[0].content.startswith("def add")
    assert records[2].completion[0].content == "A"
    assert records[3].completion[0].content == "Paris"
    general_prompt = catalog.get(PromptProfileName.GENERAL).system_prompt
    assert {record.prompt[0].content for record in records} == {general_prompt}

    code_source = _source_record(1)
    serialized_code = records[1].model_dump_json()
    assert all(test not in serialized_code for test in code_source.example.reference.tests)
    assert "test_setup" not in serialized_code
    assert "canonical_solution" not in records[1].prompt[1].content


def test_masks_every_prompt_token_and_preserves_completion_tokens() -> None:
    catalog = load_prompt_catalog(PROMPT_PATH)
    tokenizer = FakeTokenizer()
    record = format_source_record(_source_record(0), catalog, tokenizer, 4096)

    tokenized = tokenize_prompt_completion(
        record.prompt,
        record.completion,
        tokenizer,
        4096,
    )

    assert tokenized.labels[: tokenized.prompt_tokens] == (IGNORE_INDEX,) * tokenized.prompt_tokens
    assert (
        tokenized.labels[tokenized.prompt_tokens :]
        == tokenized.input_ids[tokenized.prompt_tokens :]
    )
    assert tokenized.completion_tokens == record.completion_tokens
    assert any(label != IGNORE_INDEX for label in tokenized.labels)


def test_rejects_overlength_rows_without_truncating_completion() -> None:
    catalog = load_prompt_catalog(PROMPT_PATH)
    tokenizer = FakeTokenizer()
    source = _source_record(3)

    with pytest.raises(SFTLengthError, match="fully templated conversation"):
        format_source_record(source, catalog, tokenizer, 20)

    eligibility = build_sft_eligibility_filter(catalog, tokenizer, 20)
    assert eligibility(source.example) is False


def test_prepares_reproducible_model_facing_artifacts(tmp_path: Path) -> None:
    config = _small_config()
    catalog = load_prompt_catalog(PROMPT_PATH)
    tokenizer = FakeTokenizer()
    leakage = BenchmarkLeakageIndex(frozenset(), frozenset(), "0" * 64)
    source = prepare_training_data(
        config,
        tmp_path / "source",
        _fixture_rows,
        build_sft_eligibility_filter(catalog, tokenizer, config.data.max_length),
        leakage,
    )

    first = prepare_sft_data(
        config,
        catalog,
        tokenizer,
        source.train_path,
        source.validation_path,
        source.manifest_path,
        tmp_path / "first",
    )
    second = prepare_sft_data(
        config,
        catalog,
        tokenizer,
        source.train_path,
        source.validation_path,
        source.manifest_path,
        tmp_path / "second",
    )

    assert first.train_row_count == 4
    assert first.validation_row_count == 4
    assert first.train_sha256 == second.train_sha256
    assert first.validation_sha256 == second.validation_sha256
    train = load_sft_training_records(first.train_path)
    validation = load_sft_training_records(first.validation_path)
    assert {record.domain for record in train} == set(Domain)
    assert {record.source_id for record in train}.isdisjoint(
        record.source_id for record in validation
    )

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["prompt_profile"] == "general"
    assert manifest["completion_only_loss"] is True
    assert manifest["tokenizer"]["revision"] == config.model.revision
    assert manifest["files"]["train"]["sha256"] == first.train_sha256


def test_source_manifest_hash_is_enforced(tmp_path: Path) -> None:
    config = _small_config()
    catalog = load_prompt_catalog(PROMPT_PATH)
    tokenizer = FakeTokenizer()
    source = prepare_training_data(
        config,
        tmp_path / "source",
        _fixture_rows,
        leakage_index=BenchmarkLeakageIndex(frozenset(), frozenset(), "0" * 64),
    )
    source.train_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        prepare_sft_data(
            config,
            catalog,
            tokenizer,
            source.train_path,
            source.validation_path,
            source.manifest_path,
            tmp_path / "sft",
        )


def test_missing_code_solution_is_rejected() -> None:
    source = _source_record(1)
    example = source.example.model_copy(
        update={
            "reference": source.example.reference.model_copy(update={"canonical_solution": None})
        }
    )

    with pytest.raises(ValueError, match="missing a canonical solution"):
        render_completion(example)
