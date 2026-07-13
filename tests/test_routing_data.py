from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from small_models_society.data.config import BenchmarkConfig, DatasetSource
from small_models_society.data.loaders import SourceRow, normalize_row
from small_models_society.data.prepare import canonical_json, prepare_benchmark, sha256_bytes
from small_models_society.inference.prompts import load_prompt_catalog
from small_models_society.routing.artifacts import load_workflow_requests
from small_models_society.routing.config import RoutingConfig, load_routing_config
from small_models_society.routing.contracts import OutputContract
from small_models_society.routing.data import (
    RoutingSplit,
    load_calculator_evaluator_records,
    load_routing_evaluator_records,
    load_training_exclusion_index,
    load_verified_routing_split,
    prepare_routing_data,
)
from small_models_society.schemas import Domain, KnowledgeExample
from small_models_society.training.contracts import SourceTrainingRecord, TrainingSplit
from small_models_society.training.prepare import (
    load_benchmark_leakage_index,
    normalized_content_sha256,
    qualified_source_id,
)

ROOT = Path(__file__).parents[1]
ROUTING_CONFIG = ROOT / "configs" / "routing.yaml"
PROMPT_CONFIG = ROOT / "configs" / "prompt_profiles.yaml"


def _domain_for_source(source: DatasetSource) -> Domain:
    return {
        "openai/gsm8k": Domain.MATH,
        "google-research-datasets/mbpp": Domain.CODE,
        "allenai/ai2_arc": Domain.LOGIC,
        "hotpotqa/hotpot_qa": Domain.KNOWLEDGE,
    }[source.dataset]


def _rows(domain: Domain, count: int = 12) -> list[dict[str, Any]]:
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


def _write_source_records(path: Path, records: list[SourceTrainingRecord]) -> str:
    content = (
        "\n".join(canonical_json(record.model_dump(mode="json")) for record in records) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha256_bytes(content)


def _training_exclusions(
    tmp_path: Path,
    config: RoutingConfig,
) -> tuple[Path, Path, Path]:
    train_records: list[SourceTrainingRecord] = []
    validation_records: list[SourceTrainingRecord] = []
    for domain in Domain:
        source = config.data.sources[domain]
        train_example = normalize_row(domain, _rows(domain)[1], 1)
        validation_example = normalize_row(domain, _rows(domain)[2], 2)
        train_records.append(
            SourceTrainingRecord(
                source_id=qualified_source_id(
                    source.model_copy(update={"split": "train"}),
                    train_example.id,
                ),
                domain=domain,
                split=TrainingSplit.TRAIN,
                content_sha256=normalized_content_sha256(train_example),
                example=train_example,
            )
        )
        validation_records.append(
            SourceTrainingRecord(
                source_id=qualified_source_id(
                    source.model_copy(update={"split": "train"}),
                    validation_example.id,
                ),
                domain=domain,
                split=TrainingSplit.VALIDATION,
                content_sha256=normalized_content_sha256(validation_example),
                example=validation_example,
            )
        )
    train_path = tmp_path / "training" / "train.jsonl"
    validation_path = tmp_path / "training" / "validation.jsonl"
    train_sha256 = _write_source_records(train_path, train_records)
    validation_sha256 = _write_source_records(validation_path, validation_records)
    manifest = {
        "schema_version": 1,
        "row_count": len(train_records) + len(validation_records),
        "files": {
            "train": {
                "path": train_path.name,
                "row_count": len(train_records),
                "sha256": train_sha256,
            },
            "validation": {
                "path": validation_path.name,
                "row_count": len(validation_records),
                "sha256": validation_sha256,
            },
        },
    }
    manifest_path = tmp_path / "training" / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return train_path, validation_path, manifest_path


def _routing_config(tmp_path: Path) -> RoutingConfig:
    config = load_routing_config(ROUTING_CONFIG)
    benchmark_config = BenchmarkConfig(
        seed=42,
        sample_per_domain=1,
        output_dir="unused",
        sources=config.data.sources,
    )
    benchmark = prepare_benchmark(
        benchmark_config,
        tmp_path / "benchmark",
        _fixture_rows,
    )
    train_path, validation_path, training_manifest = _training_exclusions(
        tmp_path,
        config,
    )
    data = config.data.model_copy(
        update={
            "development_size_per_domain": 2,
            "test_size_per_domain": 2,
            "output_dir": str(tmp_path / "routing"),
            "benchmark_path": str(benchmark.benchmark_path),
            "benchmark_manifest_path": str(benchmark.manifest_path),
            "training_train_path": str(train_path),
            "training_validation_path": str(validation_path),
            "training_manifest_path": str(training_manifest),
        }
    )
    return config.model_copy(update={"data": data})


def test_prepares_reproducible_disjoint_routing_splits(tmp_path: Path) -> None:
    config = _routing_config(tmp_path)
    catalog = load_prompt_catalog(PROMPT_CONFIG)
    first = prepare_routing_data(
        config,
        catalog,
        tmp_path / "first",
        _fixture_rows,
    )
    second = prepare_routing_data(
        config,
        catalog,
        tmp_path / "second",
        _fixture_rows,
    )

    assert first.development_row_count == 8
    assert first.test_row_count == 8
    assert first.development_requests_sha256 == second.development_requests_sha256
    assert first.test_requests_sha256 == second.test_requests_sha256
    assert first.development_requests_path.read_bytes() == (
        second.development_requests_path.read_bytes()
    )
    assert first.test_requests_path.read_bytes() == second.test_requests_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()

    development = load_routing_evaluator_records(first.development_evaluator_path)
    test = load_routing_evaluator_records(first.test_evaluator_path)
    assert {record.split for record in development} == {RoutingSplit.DEVELOPMENT}
    assert {record.split for record in test} == {RoutingSplit.TEST}
    assert {record.request_id for record in development}.isdisjoint(
        record.request_id for record in test
    )
    assert {record.source_id for record in development}.isdisjoint(
        record.source_id for record in test
    )
    assert {record.source_content_sha256 for record in development}.isdisjoint(
        record.source_content_sha256 for record in test
    )
    assert {
        domain: sum(record.example.domain is domain for record in development) for domain in Domain
    } == {domain: 2 for domain in Domain}
    assert {
        domain: sum(record.example.domain is domain for record in test) for domain in Domain
    } == {domain: 2 for domain in Domain}


def test_calculator_companion_is_deterministic_and_separate(tmp_path: Path) -> None:
    config = _routing_config(tmp_path)
    prepared = prepare_routing_data(
        config,
        load_prompt_catalog(PROMPT_CONFIG),
        row_loader=_fixture_rows,
    )
    development_requests = load_workflow_requests(prepared.calculator_development_requests_path)
    development_evaluators = load_calculator_evaluator_records(
        prepared.calculator_development_evaluator_path
    )
    test_requests = load_workflow_requests(prepared.calculator_test_requests_path)
    test_evaluators = load_calculator_evaluator_records(prepared.calculator_test_evaluator_path)

    assert len(development_requests) == len(development_evaluators) == 6
    assert len(test_requests) == len(test_evaluators) == 6
    assert {request.request_id for request in development_requests} == {
        record.request_id for record in development_evaluators
    }
    assert {request.request_id for request in test_requests} == {
        record.request_id for record in test_evaluators
    }
    assert {request.request_id for request in development_requests}.isdisjoint(
        request.request_id for request in test_requests
    )
    development_by_id = {request.request_id: request for request in development_requests}
    for evaluator in development_evaluators:
        request = development_by_id[evaluator.request_id]
        assert request.messages[-1].content == evaluator.expression
        assert request.output_contract is OutputContract.NUMERIC
        assert '"expected_output"' not in request.model_dump_json()

    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    calculator = manifest["companion_suites"]["calculator"]
    assert calculator["included_in_four_domain_aggregate"] is False
    assert calculator["rows_by_split"] == {"development": 6, "test": 6}
    assert manifest["row_count"] == 16


def test_requests_are_opaque_and_evaluator_records_retain_gold(tmp_path: Path) -> None:
    config = _routing_config(tmp_path)
    prepared = prepare_routing_data(
        config,
        load_prompt_catalog(PROMPT_CONFIG),
        row_loader=_fixture_rows,
    )
    requests = load_workflow_requests(prepared.development_requests_path)
    evaluators = load_routing_evaluator_records(prepared.development_evaluator_path)
    requests_by_id = {request.request_id: request for request in requests}

    assert set(requests_by_id) == {record.request_id for record in evaluators}
    for request in requests:
        serialized = request.model_dump_json()
        assert '"domain"' not in serialized
        assert '"reference"' not in serialized
        assert '"supporting_facts"' not in serialized
        assert request.policy.required_quality == 0.8
        assert request.request_id.startswith("routing-")

    output_contract_by_domain = {
        Domain.MATH: OutputContract.NUMERIC,
        Domain.CODE: OutputContract.PYTHON_SOURCE,
        Domain.LOGIC: OutputContract.CHOICE_LABEL,
        Domain.KNOWLEDGE: OutputContract.SHORT_ANSWER,
    }
    for evaluator in evaluators:
        assert (
            requests_by_id[evaluator.request_id].output_contract
            is (output_contract_by_domain[evaluator.example.domain])
        )
        assert evaluator.example.reference is not None
        if isinstance(evaluator.example, KnowledgeExample):
            request = requests_by_id[evaluator.request_id]
            request_text = request.model_dump_json()
            assert request.attributes == {"retrieval_query": evaluator.example.input.question}
            for passage in evaluator.example.input.context:
                assert passage not in request_text
        else:
            assert requests_by_id[evaluator.request_id].attributes == {}


def test_selected_rows_exclude_verified_phase1_and_phase3_content(tmp_path: Path) -> None:
    config = _routing_config(tmp_path)
    prepared = prepare_routing_data(
        config,
        load_prompt_catalog(PROMPT_CONFIG),
        row_loader=_fixture_rows,
    )
    selected = [
        *load_routing_evaluator_records(prepared.development_evaluator_path),
        *load_routing_evaluator_records(prepared.test_evaluator_path),
    ]
    training = load_training_exclusion_index(
        Path(config.data.training_train_path),
        Path(config.data.training_validation_path),
        Path(config.data.training_manifest_path),
    )
    benchmark = load_benchmark_leakage_index(
        Path(config.data.benchmark_path),
        Path(config.data.benchmark_manifest_path),
    )

    assert {record.source_id for record in selected}.isdisjoint(benchmark.source_ids)
    assert {record.source_content_sha256 for record in selected}.isdisjoint(
        benchmark.content_sha256s
    )
    assert {record.source_id for record in selected}.isdisjoint(training.source_ids)
    assert {record.source_content_sha256 for record in selected}.isdisjoint(
        training.content_sha256s
    )
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["row_count"] == 16
    assert manifest["rows_by_split"] == {"development": 8, "test": 8}
    assert manifest["exclusions"]["phase3_manifest_sha256"] == (training.manifest_sha256)
    assert all(source["selected_development_rows"] == 2 for source in manifest["sources"].values())
    assert all(source["selected_test_rows"] == 2 for source in manifest["sources"].values())


def test_rejects_existing_artifacts_without_overwrite(tmp_path: Path) -> None:
    config = _routing_config(tmp_path)
    output_dir = tmp_path / "routing-output"
    prepared = prepare_routing_data(
        config,
        load_prompt_catalog(PROMPT_CONFIG),
        output_dir,
        _fixture_rows,
    )

    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_routing_data(
            config,
            load_prompt_catalog(PROMPT_CONFIG),
            output_dir,
            _fixture_rows,
        )

    overwritten = prepare_routing_data(
        config,
        load_prompt_catalog(PROMPT_CONFIG),
        output_dir,
        _fixture_rows,
        overwrite=True,
    )
    assert overwritten.development_requests_sha256 == prepared.development_requests_sha256


def test_rejects_tampered_training_exclusion_bytes(tmp_path: Path) -> None:
    config = _routing_config(tmp_path)
    Path(config.data.training_train_path).write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        prepare_routing_data(
            config,
            load_prompt_catalog(PROMPT_CONFIG),
            row_loader=_fixture_rows,
        )


def test_verified_split_rejects_tampered_or_mispaired_artifacts(tmp_path: Path) -> None:
    config = _routing_config(tmp_path)
    prepared = prepare_routing_data(
        config,
        load_prompt_catalog(PROMPT_CONFIG),
        row_loader=_fixture_rows,
    )
    verified = load_verified_routing_split(
        prepared.development_requests_path,
        prepared.development_evaluator_path,
        prepared.manifest_path,
        RoutingSplit.DEVELOPMENT,
    )
    assert len(verified.requests) == len(verified.evaluator_records) == 8
    assert verified.requests_sha256 == prepared.development_requests_sha256

    prepared.development_requests_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="request hash does not match"):
        load_verified_routing_split(
            prepared.development_requests_path,
            prepared.development_evaluator_path,
            prepared.manifest_path,
            RoutingSplit.DEVELOPMENT,
        )


def test_rejects_insufficient_rows_after_exclusions(tmp_path: Path) -> None:
    config = _routing_config(tmp_path)

    with pytest.raises(ValueError, match=r"eligible routing rows.*4 requested"):
        prepare_routing_data(
            config,
            load_prompt_catalog(PROMPT_CONFIG),
            row_loader=lambda source: _rows(_domain_for_source(source), 3),
        )
