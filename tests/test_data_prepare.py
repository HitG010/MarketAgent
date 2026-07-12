from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from small_models_society.data.config import BenchmarkConfig, DatasetSource
from small_models_society.data.loaders import SourceRow, load_source_rows, normalize_code
from small_models_society.data.prepare import load_benchmark, prepare_benchmark
from small_models_society.schemas import Domain


def _source(name: str) -> DatasetSource:
    return DatasetSource(dataset=name, config="fixture", split="test", revision="a" * 40)


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        seed=42,
        sample_per_domain=5,
        output_dir="unused",
        sources={
            Domain.MATH: _source("math"),
            Domain.CODE: _source("code"),
            Domain.LOGIC: _source("logic"),
            Domain.KNOWLEDGE: _source("knowledge"),
        },
    )


def _fixture_rows(source: DatasetSource) -> Iterable[SourceRow]:
    rows: dict[str, list[dict[str, Any]]] = {
        "math": [
            {"question": f"What is {index} + 1?", "answer": f"work\n#### {index + 1}"}
            for index in range(7)
        ],
        "code": [
            {
                "source_file": "fixture.jsonl",
                "task_id": index,
                "prompt": f"Return {index}.",
                "code": f"def answer():\n    return {index}",
                "test_imports": [],
                "test_list": [f"assert answer() == {index}"],
            }
            for index in range(7)
        ],
        "logic": [
            {
                "id": str(index),
                "question": f"Select A for item {index}.",
                "choices": {"label": ["A", "B"], "text": ["yes", "no"]},
                "answerKey": "A",
            }
            for index in range(7)
        ],
        "knowledge": [
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
            for index in range(7)
        ],
    }
    return rows[source.dataset]


def test_preparing_twice_produces_identical_balanced_hashes(tmp_path: Path) -> None:
    first = prepare_benchmark(_config(), tmp_path / "first", _fixture_rows)
    second = prepare_benchmark(_config(), tmp_path / "second", _fixture_rows)

    assert first.sha256 == second.sha256
    assert first.benchmark_path.read_bytes() == second.benchmark_path.read_bytes()
    assert first.row_count == 20

    examples = load_benchmark(first.benchmark_path)
    assert {domain: sum(item.domain == domain for item in examples) for domain in Domain} == {
        domain: 5 for domain in Domain
    }

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["benchmark_sha256"] == first.sha256
    assert manifest["row_count"] == 20


def test_normalized_input_does_not_include_gold_references(tmp_path: Path) -> None:
    prepared = prepare_benchmark(_config(), tmp_path, _fixture_rows)

    for example in load_benchmark(prepared.benchmark_path):
        model_input = example.input.model_dump(mode="json")
        assert "reference" not in model_input
        assert "answer" not in model_input
        assert "answer_label" not in model_input
        assert "canonical_solution" not in model_input


def test_mbpp_entry_point_parsing_suppresses_reference_syntax_warnings() -> None:
    row: SourceRow = {
        "source_file": "fixture.jsonl",
        "task_id": 1,
        "prompt": "Return a regular expression.",
        "code": "def pattern():\n    return '\\w'",
        "test_imports": [],
        "test_list": ["assert pattern() == '\\\\w'"],
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        warnings.simplefilter("error", DeprecationWarning)
        example = normalize_code(row, 0)

    assert example.input.entry_point == "pattern"


def test_source_loader_forwards_pinned_revision_and_offline_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_load_dataset(*args: object, **kwargs: Any) -> list[SourceRow]:
        calls.append({"args": args, **kwargs})
        return []

    monkeypatch.setattr("small_models_society.data.loaders.load_dataset", fake_load_dataset)
    source = DatasetSource(
        dataset="owner/dataset",
        config="configuration",
        split="train",
        revision="a" * 40,
    )

    assert list(load_source_rows(source, local_files_only=True)) == []

    assert calls[0]["args"] == ("owner/dataset", "configuration")
    assert calls[0]["split"] == "train"
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["download_config"].local_files_only is True
