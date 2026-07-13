from __future__ import annotations

from pathlib import Path

import pytest

from small_models_society.data.prepare import canonical_json, sha256_bytes
from small_models_society.retrieval.bm25 import BM25Retriever
from small_models_society.retrieval.corpus import (
    load_retrieval_corpus,
    prepare_retrieval_corpus,
    verify_retrieval_corpus_source,
)
from small_models_society.routing.config import RoutingConfig, load_routing_config
from small_models_society.routing.data import RoutingEvaluatorRecord, RoutingSplit
from small_models_society.schemas import (
    KnowledgeExample,
    KnowledgeInput,
    KnowledgeReference,
    SupportingFact,
)
from small_models_society.training.prepare import normalized_content_sha256

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "routing.yaml"


def _config() -> RoutingConfig:
    return load_routing_config(CONFIG_PATH)


def _record(
    request_id: str,
    context: list[str],
    supporting_titles: list[str],
    split: RoutingSplit = RoutingSplit.DEVELOPMENT,
) -> RoutingEvaluatorRecord:
    example = KnowledgeExample(
        id=request_id,
        input=KnowledgeInput(question=f"Question for {request_id}?", context=context),
        reference=KnowledgeReference(
            answers=["hidden answer"],
            supporting_facts=[
                SupportingFact(title=title, sentence_index=0) for title in supporting_titles
            ],
        ),
    )
    return RoutingEvaluatorRecord(
        request_id=request_id,
        split=split,
        source_id=f"source::{request_id}",
        source_content_sha256=normalized_content_sha256(example),
        example=example,
    )


def _routing_inputs(
    tmp_path: Path,
    records: list[RoutingEvaluatorRecord],
    split: RoutingSplit = RoutingSplit.DEVELOPMENT,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    evaluator_path = tmp_path / f"{split.value}.evaluator.jsonl"
    content = (
        "\n".join(canonical_json(record.model_dump(mode="json")) for record in records) + "\n"
    ).encode("utf-8")
    evaluator_path.write_bytes(content)
    metadata = {
        "path": evaluator_path.name,
        "sha256": sha256_bytes(content),
        "row_count": len(records),
    }
    manifest = {
        "routing_config_fingerprint": _config().fingerprint(),
        "files": {f"{split.value}_evaluator": metadata},
    }
    manifest_path = tmp_path / "routing-manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return evaluator_path, manifest_path


def _records() -> list[RoutingEvaluatorRecord]:
    return [
        _record(
            "routing-one",
            [
                "Paris\nParis is the capital of France.",
                "France\nFrance is a country in Europe.",
                "Noise\nThis passage is irrelevant.",
            ],
            ["Paris", "France"],
        ),
        _record(
            "routing-two",
            [
                "Paris\nParis is the capital of France.",
                "Berlin\nBerlin is the capital of Germany.",
            ],
            ["Berlin", "Missing title"],
        ),
    ]


def test_builds_reproducible_deduplicated_corpus_with_hidden_labels(
    tmp_path: Path,
) -> None:
    evaluator_path, routing_manifest_path = _routing_inputs(tmp_path, _records())
    first = prepare_retrieval_corpus(
        _config(),
        evaluator_path,
        routing_manifest_path,
        RoutingSplit.DEVELOPMENT,
        tmp_path / "first",
    )
    second = prepare_retrieval_corpus(
        _config(),
        evaluator_path,
        routing_manifest_path,
        RoutingSplit.DEVELOPMENT,
        tmp_path / "second",
    )

    assert first.documents_path.read_bytes() == second.documents_path.read_bytes()
    assert first.relevance_path.read_bytes() == second.relevance_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    loaded = load_retrieval_corpus(first.manifest_path.parent)
    verify_retrieval_corpus_source(
        loaded,
        RoutingSplit.DEVELOPMENT,
        loaded.manifest.source_evaluator_sha256,
    )
    assert len(loaded.documents) == 4
    assert len(loaded.relevance) == 2
    assert loaded.manifest.indexed_supporting_titles == 3
    assert loaded.manifest.total_supporting_titles == 4
    assert loaded.manifest.corpus_coverage == 0.75

    corpus_text = first.documents_path.read_text(encoding="utf-8")
    assert "routing-one" not in corpus_text
    assert "hidden answer" not in corpus_text
    assert "supporting_facts" not in corpus_text
    relevance = {record.request_id: record for record in loaded.relevance}
    assert len(relevance["routing-one"].relevant_document_ids) == 2
    assert relevance["routing-two"].unresolved_titles == ("Missing title",)
    retriever = BM25Retriever.from_loaded_corpus(loaded, _config().retrieval)
    assert retriever.corpus_fingerprint == loaded.manifest.corpus_fingerprint

    with pytest.raises(ValueError, match="split does not match"):
        verify_retrieval_corpus_source(
            loaded,
            RoutingSplit.TEST,
            loaded.manifest.source_evaluator_sha256,
        )
    with pytest.raises(ValueError, match="source evaluator does not match"):
        verify_retrieval_corpus_source(
            loaded,
            RoutingSplit.DEVELOPMENT,
            "a" * 64,
        )


def test_development_and_test_corpora_have_distinct_identities(tmp_path: Path) -> None:
    development_records = _records()
    development_path, development_manifest = _routing_inputs(
        tmp_path / "development-source",
        development_records,
    )
    test_records = [
        _record(
            "routing-test",
            ["Tokyo\nTokyo is the capital of Japan."],
            ["Tokyo"],
            split=RoutingSplit.TEST,
        )
    ]
    test_path, test_manifest = _routing_inputs(
        tmp_path / "test-source",
        test_records,
        split=RoutingSplit.TEST,
    )
    development = prepare_retrieval_corpus(
        _config(),
        development_path,
        development_manifest,
        RoutingSplit.DEVELOPMENT,
        tmp_path / "development-corpus",
    )
    test = prepare_retrieval_corpus(
        _config(),
        test_path,
        test_manifest,
        RoutingSplit.TEST,
        tmp_path / "test-corpus",
    )

    assert development.manifest.split == "development"
    assert test.manifest.split == "test"
    assert development.manifest.corpus_fingerprint != test.manifest.corpus_fingerprint


def test_rejects_tampered_routing_evaluator(tmp_path: Path) -> None:
    evaluator_path, routing_manifest_path = _routing_inputs(tmp_path, _records())
    evaluator_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        prepare_retrieval_corpus(
            _config(),
            evaluator_path,
            routing_manifest_path,
            RoutingSplit.DEVELOPMENT,
            tmp_path / "corpus",
        )


def test_rejects_tampered_corpus_bytes(tmp_path: Path) -> None:
    evaluator_path, routing_manifest_path = _routing_inputs(tmp_path, _records())
    prepared = prepare_retrieval_corpus(
        _config(),
        evaluator_path,
        routing_manifest_path,
        RoutingSplit.DEVELOPMENT,
        tmp_path / "corpus",
    )
    prepared.documents_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        load_retrieval_corpus(prepared.manifest_path.parent)


def test_refuses_artifact_collision_without_explicit_overwrite(tmp_path: Path) -> None:
    evaluator_path, routing_manifest_path = _routing_inputs(tmp_path, _records())
    output_dir = tmp_path / "corpus"
    prepare_retrieval_corpus(
        _config(),
        evaluator_path,
        routing_manifest_path,
        RoutingSplit.DEVELOPMENT,
        output_dir,
    )

    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_retrieval_corpus(
            _config(),
            evaluator_path,
            routing_manifest_path,
            RoutingSplit.DEVELOPMENT,
            output_dir,
        )


def test_rejects_ambiguous_titles_with_different_content(tmp_path: Path) -> None:
    records = [
        _record(
            "routing-ambiguous",
            ["Same\nFirst body.", "Same\nSecond body."],
            ["Same"],
        )
    ]
    evaluator_path, routing_manifest_path = _routing_inputs(tmp_path, records)

    with pytest.raises(ValueError, match="ambiguous duplicate context title"):
        prepare_retrieval_corpus(
            _config(),
            evaluator_path,
            routing_manifest_path,
            RoutingSplit.DEVELOPMENT,
            tmp_path / "corpus",
        )
