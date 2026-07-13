"""Deterministic HotpotQA corpus construction with hidden relevance labels."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from small_models_society.data.prepare import canonical_json, sha256_bytes
from small_models_society.retrieval.contracts import (
    RetrievalCorpusManifest,
    RetrievalDocument,
    RetrievalRelevanceRecord,
    create_retrieval_document,
)
from small_models_society.routing.artifacts import RoutingArtifact
from small_models_society.routing.config import RoutingConfig
from small_models_society.routing.data import (
    RoutingEvaluatorRecord,
    RoutingSplit,
    load_routing_evaluator_records,
)
from small_models_society.schemas import KnowledgeExample

ContractT = TypeVar("ContractT", bound=BaseModel)


@dataclass(frozen=True)
class PreparedRetrievalCorpus:
    documents_path: Path
    relevance_path: Path
    manifest_path: Path
    manifest: RetrievalCorpusManifest


@dataclass(frozen=True)
class LoadedRetrievalCorpus:
    documents: tuple[RetrievalDocument, ...]
    relevance: tuple[RetrievalRelevanceRecord, ...]
    manifest: RetrievalCorpusManifest


def verify_retrieval_corpus_source(
    corpus: LoadedRetrievalCorpus,
    split: RoutingSplit,
    evaluator_sha256: str,
) -> None:
    if corpus.manifest.split != split.value:
        raise ValueError("retrieval corpus split does not match routing split")
    if corpus.manifest.source_evaluator_sha256 != evaluator_sha256:
        raise ValueError("retrieval corpus source evaluator does not match routing split")


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return cast(Mapping[str, Any], value)


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _normalize_title_key(value: str) -> str:
    return _normalize_text(value).casefold()


def _document_from_passage(passage: str) -> RetrievalDocument:
    title, separator, text = passage.partition("\n")
    normalized_title = _normalize_text(title)
    normalized_text = _normalize_text(text)
    if not separator or not normalized_title or not normalized_text:
        raise ValueError("HotpotQA context passage must contain a nonempty title and body")
    return create_retrieval_document(normalized_title, normalized_text)


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_records(path: Path, records: Sequence[BaseModel]) -> RoutingArtifact:
    if not records:
        raise ValueError("retrieval artifact requires at least one row")
    content = (
        "\n".join(canonical_json(record.model_dump(mode="json")) for record in records) + "\n"
    ).encode("utf-8")
    _write_atomic(path, content)
    return RoutingArtifact(path=path, sha256=sha256_bytes(content), row_count=len(records))


def _load_records(
    path: Path,
    contract: type[ContractT],
) -> list[ContractT]:
    records: list[ContractT] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(contract.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid retrieval row at {path}:{line_number}") from error
    if not records:
        raise ValueError(f"retrieval artifact contains no rows: {path}")
    return records


def load_retrieval_documents(path: Path) -> list[RetrievalDocument]:
    records = _load_records(path, RetrievalDocument)
    document_ids = [record.document_id for record in records]
    content_hashes = [record.content_sha256 for record in records]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError(f"retrieval corpus contains duplicate document IDs: {path}")
    if len(set(content_hashes)) != len(content_hashes):
        raise ValueError(f"retrieval corpus contains duplicate document content: {path}")
    return records


def load_retrieval_relevance(path: Path) -> list[RetrievalRelevanceRecord]:
    records = _load_records(path, RetrievalRelevanceRecord)
    request_ids = [record.request_id for record in records]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError(f"retrieval relevance contains duplicate request IDs: {path}")
    return records


def _corpus_fingerprint(
    corpus_id: str,
    split: str,
    tokenizer_id: str,
    documents: Sequence[RetrievalDocument],
) -> str:
    payload = {
        "corpus_id": corpus_id,
        "split": split,
        "tokenizer_id": tokenizer_id,
        "documents": [document.model_dump(mode="json") for document in documents],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _verify_routing_evaluator(
    config: RoutingConfig,
    evaluator_path: Path,
    routing_manifest_path: Path,
    split: RoutingSplit,
) -> tuple[list[RoutingEvaluatorRecord], str]:
    try:
        manifest = _mapping(
            json.loads(routing_manifest_path.read_bytes()),
            "routing data manifest",
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid routing data manifest: {routing_manifest_path}") from error
    if manifest.get("routing_config_fingerprint") != config.fingerprint():
        raise ValueError("routing data manifest uses a different configuration fingerprint")
    files = _mapping(manifest.get("files"), "routing data files")
    metadata = _mapping(files.get(f"{split.value}_evaluator"), "routing evaluator metadata")
    try:
        evaluator_bytes = evaluator_path.read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read routing evaluator: {evaluator_path}") from error
    evaluator_sha256 = sha256_bytes(evaluator_bytes)
    if metadata.get("path") != evaluator_path.name:
        raise ValueError("routing evaluator path does not match routing data manifest")
    if metadata.get("sha256") != evaluator_sha256:
        raise ValueError("routing evaluator hash does not match routing data manifest")
    records = load_routing_evaluator_records(evaluator_path)
    if metadata.get("row_count") != len(records):
        raise ValueError("routing evaluator row count does not match routing data manifest")
    if any(record.split is not split for record in records):
        raise ValueError("routing evaluator contains rows from a different split")
    return records, evaluator_sha256


def _build_documents_and_relevance(
    records: Sequence[RoutingEvaluatorRecord],
    split: RoutingSplit,
) -> tuple[list[RetrievalDocument], list[RetrievalRelevanceRecord]]:
    documents: dict[str, RetrievalDocument] = {}
    relevance: list[RetrievalRelevanceRecord] = []
    for record in records:
        if not isinstance(record.example, KnowledgeExample):
            continue
        local_documents: dict[str, RetrievalDocument] = {}
        for passage in record.example.input.context:
            document = _document_from_passage(passage)
            title_key = _normalize_title_key(document.title)
            previous = local_documents.get(title_key)
            if previous is not None and previous.document_id != document.document_id:
                raise ValueError("HotpotQA record contains an ambiguous duplicate context title")
            local_documents[title_key] = document
            documents[document.document_id] = document

        supporting_titles = sorted(
            {_normalize_text(fact.title) for fact in record.example.reference.supporting_facts},
            key=str.casefold,
        )
        relevant_document_ids: set[str] = set()
        unresolved_titles: list[str] = []
        for supporting_title in supporting_titles:
            relevant_document = local_documents.get(_normalize_title_key(supporting_title))
            if relevant_document is None:
                unresolved_titles.append(supporting_title)
            else:
                relevant_document_ids.add(relevant_document.document_id)
        relevance.append(
            RetrievalRelevanceRecord(
                request_id=record.request_id,
                split=split.value,
                relevant_document_ids=tuple(sorted(relevant_document_ids)),
                unresolved_titles=tuple(sorted(unresolved_titles)),
            )
        )

    if not relevance:
        raise ValueError("routing evaluator contains no knowledge records")
    return (
        sorted(documents.values(), key=lambda document: document.document_id),
        sorted(relevance, key=lambda record: record.request_id),
    )


def prepare_retrieval_corpus(
    config: RoutingConfig,
    evaluator_path: Path,
    routing_manifest_path: Path,
    split: RoutingSplit,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> PreparedRetrievalCorpus:
    """Build one verified, content-addressed HotpotQA corpus for a routing split."""

    documents_path = output_dir / "documents.jsonl"
    relevance_path = output_dir / "relevance.jsonl"
    manifest_path = output_dir / "manifest.json"
    artifacts_exist = any(path.exists() for path in (documents_path, relevance_path, manifest_path))
    if artifacts_exist and not overwrite:
        raise FileExistsError("retrieval corpus artifacts already exist; use overwrite explicitly")

    records, evaluator_sha256 = _verify_routing_evaluator(
        config,
        evaluator_path,
        routing_manifest_path,
        split,
    )
    documents, relevance = _build_documents_and_relevance(records, split)
    documents_artifact = _write_records(documents_path, documents)
    relevance_artifact = _write_records(relevance_path, relevance)
    indexed_supporting_titles = sum(len(record.relevant_document_ids) for record in relevance)
    total_supporting_titles = sum(record.supporting_title_count for record in relevance)
    corpus_fingerprint = _corpus_fingerprint(
        config.retrieval.corpus_id,
        split.value,
        config.retrieval.tokenizer,
        documents,
    )
    values: dict[str, object] = {
        "schema_version": 1,
        "corpus_id": config.retrieval.corpus_id,
        "split": split.value,
        "tokenizer_id": config.retrieval.tokenizer,
        "routing_config_fingerprint": config.fingerprint(),
        "source_evaluator_sha256": evaluator_sha256,
        "documents_file": documents_path.name,
        "documents_sha256": documents_artifact.sha256,
        "document_count": documents_artifact.row_count,
        "relevance_file": relevance_path.name,
        "relevance_sha256": relevance_artifact.sha256,
        "relevance_count": relevance_artifact.row_count,
        "indexed_supporting_titles": indexed_supporting_titles,
        "total_supporting_titles": total_supporting_titles,
        "corpus_coverage": indexed_supporting_titles / total_supporting_titles,
        "corpus_fingerprint": corpus_fingerprint,
    }
    manifest = RetrievalCorpusManifest.model_validate(
        {
            **values,
            "manifest_fingerprint": hashlib.sha256(
                canonical_json(values).encode("utf-8")
            ).hexdigest(),
        }
    )
    _write_atomic(
        manifest_path,
        (canonical_json(manifest.model_dump(mode="json")) + "\n").encode("utf-8"),
    )
    return PreparedRetrievalCorpus(
        documents_path=documents_path,
        relevance_path=relevance_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _verified_artifact_bytes(
    directory: Path,
    filename: str,
    expected_sha256: str,
) -> Path:
    if Path(filename).name != filename:
        raise ValueError("retrieval manifest artifact paths must be local filenames")
    path = directory / filename
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read retrieval artifact: {path}") from error
    if sha256_bytes(content) != expected_sha256:
        raise ValueError(f"retrieval artifact hash does not match manifest: {path}")
    return path


def load_retrieval_corpus(directory: Path) -> LoadedRetrievalCorpus:
    manifest_path = directory / "manifest.json"
    try:
        manifest = RetrievalCorpusManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid retrieval corpus manifest: {manifest_path}") from error
    documents_path = _verified_artifact_bytes(
        directory,
        manifest.documents_file,
        manifest.documents_sha256,
    )
    relevance_path = _verified_artifact_bytes(
        directory,
        manifest.relevance_file,
        manifest.relevance_sha256,
    )
    documents = load_retrieval_documents(documents_path)
    relevance = load_retrieval_relevance(relevance_path)
    if len(documents) != manifest.document_count:
        raise ValueError("retrieval document count does not match manifest")
    if len(relevance) != manifest.relevance_count:
        raise ValueError("retrieval relevance count does not match manifest")
    if any(record.split != manifest.split for record in relevance):
        raise ValueError("retrieval relevance split does not match manifest")
    document_ids = {document.document_id for document in documents}
    if any(not set(record.relevant_document_ids).issubset(document_ids) for record in relevance):
        raise ValueError("retrieval relevance references an unknown document")
    fingerprint = _corpus_fingerprint(
        manifest.corpus_id,
        manifest.split,
        manifest.tokenizer_id,
        documents,
    )
    if fingerprint != manifest.corpus_fingerprint:
        raise ValueError("retrieval corpus fingerprint does not match documents")
    indexed = sum(len(record.relevant_document_ids) for record in relevance)
    total = sum(record.supporting_title_count for record in relevance)
    if indexed != manifest.indexed_supporting_titles or total != manifest.total_supporting_titles:
        raise ValueError("retrieval relevance coverage counts do not match manifest")
    if not math.isclose(manifest.corpus_coverage, indexed / total, rel_tol=0, abs_tol=1e-12):
        raise ValueError("retrieval corpus coverage does not match relevance records")
    return LoadedRetrievalCorpus(
        documents=tuple(documents),
        relevance=tuple(relevance),
        manifest=manifest,
    )
