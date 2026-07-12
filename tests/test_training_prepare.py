from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from small_models_society.data.config import BenchmarkConfig, DatasetSource
from small_models_society.data.loaders import SourceRow, normalize_row
from small_models_society.data.prepare import prepare_benchmark
from small_models_society.schemas import Domain
from small_models_society.training.config import TrainingConfig, load_training_config
from small_models_society.training.contracts import TrainingSplit
from small_models_society.training.prepare import (
    BenchmarkLeakageIndex,
    load_benchmark_leakage_index,
    load_source_training_records,
    normalized_content_sha256,
    prepare_training_data,
    qualified_source_id,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "training.yaml"


def _empty_leakage() -> BenchmarkLeakageIndex:
    return BenchmarkLeakageIndex(frozenset(), frozenset(), "0" * 64)


def _training_config(
    *,
    pilot_size: int = 120,
    train_size: int = 96,
    validation_size: int = 24,
) -> TrainingConfig:
    value = load_training_config(CONFIG_PATH).model_dump(mode="json")
    data = value["data"]
    assert isinstance(data, dict)
    data["pilot_size_per_domain"] = pilot_size
    data["train_size_per_domain"] = train_size
    data["validation_size_per_domain"] = validation_size
    return TrainingConfig.model_validate(value)


def _domain_for_source(source: DatasetSource) -> Domain:
    return {
        "openai/gsm8k": Domain.MATH,
        "google-research-datasets/mbpp": Domain.CODE,
        "allenai/ai2_arc": Domain.LOGIC,
        "hotpotqa/hotpot_qa": Domain.KNOWLEDGE,
    }[source.dataset]


def _rows(domain: Domain, count: int = 125) -> list[dict[str, Any]]:
    if domain is Domain.MATH:
        return [
            {"question": f"What is {index} + 1?", "answer": f"work\n#### {index + 1}"}
            for index in range(count)
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
            for index in range(count)
        ]
    if domain is Domain.LOGIC:
        return [
            {
                "id": str(index),
                "question": f"Select A for item {index}.",
                "choices": {"label": ["A", "B"], "text": ["yes", "no"]},
                "answerKey": "A",
            }
            for index in range(count)
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
        for index in range(count)
    ]


def _fixture_rows(source: DatasetSource) -> Iterable[SourceRow]:
    return _rows(_domain_for_source(source))


def test_prepares_reproducible_balanced_384_96_split(tmp_path: Path) -> None:
    config = _training_config()
    first = prepare_training_data(
        config,
        tmp_path / "first",
        _fixture_rows,
        leakage_index=_empty_leakage(),
    )
    second = prepare_training_data(
        config,
        tmp_path / "second",
        _fixture_rows,
        leakage_index=_empty_leakage(),
    )

    assert first.train_row_count == 384
    assert first.validation_row_count == 96
    assert first.train_sha256 == second.train_sha256
    assert first.validation_sha256 == second.validation_sha256
    assert first.train_path.read_bytes() == second.train_path.read_bytes()
    assert first.validation_path.read_bytes() == second.validation_path.read_bytes()

    train = load_source_training_records(first.train_path)
    validation = load_source_training_records(first.validation_path)
    assert {record.split for record in train} == {TrainingSplit.TRAIN}
    assert {record.split for record in validation} == {TrainingSplit.VALIDATION}
    assert {record.source_id for record in train}.isdisjoint(
        record.source_id for record in validation
    )
    assert {record.content_sha256 for record in train}.isdisjoint(
        record.content_sha256 for record in validation
    )
    assert {domain: sum(record.domain is domain for record in train) for domain in Domain} == {
        domain: 96 for domain in Domain
    }
    assert {domain: sum(record.domain is domain for record in validation) for domain in Domain} == {
        domain: 24 for domain in Domain
    }

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["row_count"] == 480
    assert manifest["files"]["train"]["sha256"] == first.train_sha256
    assert manifest["files"]["validation"]["sha256"] == first.validation_sha256
    assert all(source["available_rows"] == 125 for source in manifest["sources"].values())


def test_filters_benchmark_content_before_selection(tmp_path: Path) -> None:
    config = _training_config(pilot_size=3, train_size=2, validation_size=1)
    math_source = config.data.sources[Domain.MATH]
    blocked_example = normalize_row(Domain.MATH, _rows(Domain.MATH, 6)[0], 0)
    blocked_hash = normalized_content_sha256(blocked_example)
    leakage = BenchmarkLeakageIndex(
        source_ids=frozenset({qualified_source_id(math_source, "not-selected")}),
        content_sha256s=frozenset({blocked_hash}),
        benchmark_sha256="b" * 64,
    )

    prepared = prepare_training_data(
        config,
        tmp_path,
        lambda source: _rows(_domain_for_source(source), 6),
        leakage_index=leakage,
    )
    records = load_source_training_records(prepared.train_path) + load_source_training_records(
        prepared.validation_path
    )

    assert blocked_hash not in {record.content_sha256 for record in records}
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["sources"]["math"]["excluded_rows"]["benchmark_content"] == 1


def test_applies_eligibility_before_selection(tmp_path: Path) -> None:
    config = _training_config(pilot_size=3, train_size=2, validation_size=1)
    baseline = prepare_training_data(
        config,
        tmp_path / "baseline",
        lambda source: _rows(_domain_for_source(source), 8),
        leakage_index=_empty_leakage(),
    )
    baseline_records = load_source_training_records(
        baseline.train_path
    ) + load_source_training_records(baseline.validation_path)
    blocked_id = next(
        record.example.id for record in baseline_records if record.domain is Domain.MATH
    )
    prepared = prepare_training_data(
        config,
        tmp_path / "filtered",
        lambda source: _rows(_domain_for_source(source), 8),
        eligibility_filter=lambda example: example.id != blocked_id,
        leakage_index=_empty_leakage(),
    )

    filtered_records = load_source_training_records(
        prepared.train_path
    ) + load_source_training_records(prepared.validation_path)
    assert blocked_id not in {
        record.example.id for record in filtered_records if record.domain is Domain.MATH
    }
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["sources"]["math"]["excluded_rows"]["ineligible"] == 1


def test_loads_verified_benchmark_leakage_index(tmp_path: Path) -> None:
    training = _training_config(pilot_size=3, train_size=2, validation_size=1)
    benchmark_sources = {
        domain: source.model_copy(update={"split": "test"})
        for domain, source in training.data.sources.items()
    }
    benchmark_config = BenchmarkConfig(
        seed=42,
        sample_per_domain=2,
        output_dir="unused",
        sources=benchmark_sources,
    )
    prepared = prepare_benchmark(benchmark_config, tmp_path, _fixture_rows)

    index = load_benchmark_leakage_index(prepared.benchmark_path, prepared.manifest_path)

    assert index.benchmark_sha256 == prepared.sha256
    assert len(index.source_ids) == 8
    assert len(index.content_sha256s) == 8
    train_math_id = qualified_source_id(training.data.sources[Domain.MATH], "gsm8k-00000")
    assert train_math_id not in index.source_ids


def test_rejects_tampered_benchmark_bytes(tmp_path: Path) -> None:
    training = _training_config(pilot_size=3, train_size=2, validation_size=1)
    benchmark_config = BenchmarkConfig(
        seed=42,
        sample_per_domain=2,
        output_dir="unused",
        sources={
            domain: source.model_copy(update={"split": "test"})
            for domain, source in training.data.sources.items()
        },
    )
    prepared = prepare_benchmark(benchmark_config, tmp_path, _fixture_rows)
    prepared.benchmark_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        load_benchmark_leakage_index(prepared.benchmark_path, prepared.manifest_path)


def test_rejects_insufficient_eligible_rows(tmp_path: Path) -> None:
    config = _training_config(pilot_size=3, train_size=2, validation_size=1)

    with pytest.raises(ValueError, match=r"2 eligible rows.*3 requested"):
        prepare_training_data(
            config,
            tmp_path,
            lambda source: _rows(_domain_for_source(source), 2),
            leakage_index=_empty_leakage(),
        )


def test_rejects_modified_source_record_content_hash(tmp_path: Path) -> None:
    config = _training_config(pilot_size=3, train_size=2, validation_size=1)
    prepared = prepare_training_data(
        config,
        tmp_path,
        lambda source: _rows(_domain_for_source(source), 4),
        leakage_index=_empty_leakage(),
    )
    rows = prepared.train_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["content_sha256"] = "f" * 64
    prepared.train_path.write_text(
        "\n".join([json.dumps(first), *rows[1:]]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content hash mismatch"):
        load_source_training_records(prepared.train_path)


def test_default_source_loader_honors_local_files_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config(pilot_size=3, train_size=2, validation_size=1)
    model = config.model.model_copy(update={"local_files_only": True})
    config = config.model_copy(update={"model": model})
    calls: list[tuple[str, bool]] = []

    def load_rows(
        source: DatasetSource,
        *,
        local_files_only: bool,
    ) -> Iterable[SourceRow]:
        calls.append((source.dataset, local_files_only))
        return _rows(_domain_for_source(source), 4)

    monkeypatch.setattr(
        "small_models_society.training.prepare.load_source_rows",
        load_rows,
    )

    prepare_training_data(config, tmp_path, leakage_index=_empty_leakage())

    assert calls == [(config.data.sources[domain].dataset, True) for domain in Domain]
