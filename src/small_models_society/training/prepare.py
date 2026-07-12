"""Deterministic and leakage-aware specialist data preparation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from small_models_society.data.config import DatasetSource
from small_models_society.data.loaders import SourceRow, load_source_rows, normalize_row
from small_models_society.data.prepare import (
    canonical_json,
    load_benchmark,
    sha256_bytes,
)
from small_models_society.schemas import BenchmarkExample, Domain
from small_models_society.training.config import TrainingConfig
from small_models_society.training.contracts import (
    SourceTrainingRecord,
    TrainingSplit,
    validate_source_training_record,
)

RowLoader = Callable[[DatasetSource], Iterable[SourceRow]]
EligibilityFilter = Callable[[BenchmarkExample], bool]


@dataclass(frozen=True)
class BenchmarkLeakageIndex:
    source_ids: frozenset[str]
    content_sha256s: frozenset[str]
    benchmark_sha256: str


@dataclass(frozen=True)
class PreparedTrainingData:
    train_path: Path
    validation_path: Path
    manifest_path: Path
    train_sha256: str
    validation_sha256: str
    train_row_count: int
    validation_row_count: int


def normalized_content_sha256(example: BenchmarkExample) -> str:
    """Hash the exact model-visible source input with its domain."""

    payload = {
        "domain": example.domain.value,
        "input": example.input.model_dump(mode="json"),
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def qualified_source_id(source: DatasetSource, example_id: str) -> str:
    """Qualify split-local IDs so identities remain unambiguous across source splits."""

    return "::".join((source.dataset, source.config, source.split, source.revision, example_id))


def _require_mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return cast(Mapping[str, Any], value)


def _manifest_source(value: object, domain: Domain) -> DatasetSource:
    source = _require_mapping(value, f"benchmark source {domain.value}")
    try:
        selected = {key: source[key] for key in ("dataset", "config", "split", "revision")}
    except KeyError as error:
        raise ValueError(f"benchmark source {domain.value} is incomplete") from error
    return DatasetSource.model_validate(selected)


def load_benchmark_leakage_index(
    benchmark_path: Path,
    manifest_path: Path,
) -> BenchmarkLeakageIndex:
    """Build qualified ID and content guards from a verified benchmark artifact."""

    try:
        manifest = _require_mapping(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            "benchmark manifest",
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid benchmark manifest: {manifest_path}") from error

    benchmark_bytes = benchmark_path.read_bytes()
    benchmark_sha256 = sha256_bytes(benchmark_bytes)
    if manifest.get("benchmark_sha256") != benchmark_sha256:
        raise ValueError("benchmark manifest hash does not match benchmark bytes")

    manifest_sources = _require_mapping(manifest.get("sources"), "benchmark sources")
    sources = {
        domain: _manifest_source(manifest_sources.get(domain.value), domain) for domain in Domain
    }
    examples = load_benchmark(benchmark_path)
    return BenchmarkLeakageIndex(
        source_ids=frozenset(
            qualified_source_id(sources[example.domain], example.id) for example in examples
        ),
        content_sha256s=frozenset(normalized_content_sha256(example) for example in examples),
        benchmark_sha256=benchmark_sha256,
    )


def _sample_key(source_id: str, domain: Domain, seed: int) -> str:
    return sha256_bytes(f"{seed}:{domain.value}:{source_id}".encode())


def _records_bytes(records: Iterable[SourceTrainingRecord]) -> bytes:
    lines = [canonical_json(record.model_dump(mode="json")) for record in records]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def prepare_training_data(
    config: TrainingConfig,
    output_dir: Path | None = None,
    row_loader: RowLoader | None = None,
    eligibility_filter: EligibilityFilter | None = None,
    leakage_index: BenchmarkLeakageIndex | None = None,
) -> PreparedTrainingData:
    """Prepare balanced train and validation records from pinned source training splits."""

    loader = row_loader or (
        lambda source: load_source_rows(
            source,
            local_files_only=config.model.local_files_only,
        )
    )
    is_eligible = eligibility_filter or (lambda _: True)
    leakage = leakage_index or load_benchmark_leakage_index(
        Path(config.data.benchmark_path),
        Path(config.data.benchmark_manifest_path),
    )
    records: list[SourceTrainingRecord] = []
    source_statistics: dict[str, dict[str, object]] = {}

    for domain in Domain:
        source = config.data.sources[domain]
        normalized = [normalize_row(domain, row, index) for index, row in enumerate(loader(source))]
        source_ids = [qualified_source_id(source, example.id) for example in normalized]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError(f"{domain.value} source contains duplicate normalized IDs")

        ranked = sorted(
            zip(source_ids, normalized, strict=True),
            key=lambda item: _sample_key(item[0], domain, config.data.seed),
        )
        eligible: list[tuple[str, BenchmarkExample, str]] = []
        seen_content: set[str] = set()
        exclusions: Counter[str] = Counter()
        requested = config.data.pilot_size_per_domain
        scanned_rows = 0
        for source_id, example in ranked:
            if eligibility_filter is not None and len(eligible) >= requested:
                break
            scanned_rows += 1
            content_sha256 = normalized_content_sha256(example)
            if content_sha256 in seen_content:
                exclusions["duplicate_content"] += 1
                continue
            seen_content.add(content_sha256)
            if source_id in leakage.source_ids:
                exclusions["benchmark_source_id"] += 1
                continue
            if content_sha256 in leakage.content_sha256s:
                exclusions["benchmark_content"] += 1
                continue
            if not is_eligible(example):
                exclusions["ineligible"] += 1
                continue
            eligible.append((source_id, example, content_sha256))

        if len(eligible) < requested:
            raise ValueError(
                f"{domain.value} has {len(eligible)} eligible rows after exclusions; "
                f"{requested} requested"
            )

        selected = eligible[:requested]
        for index, (source_id, example, content_sha256) in enumerate(selected):
            split = (
                TrainingSplit.TRAIN
                if index < config.data.train_size_per_domain
                else TrainingSplit.VALIDATION
            )
            records.append(
                SourceTrainingRecord(
                    source_id=source_id,
                    domain=domain,
                    split=split,
                    content_sha256=content_sha256,
                    example=example,
                )
            )

        source_statistics[domain.value] = {
            **source.model_dump(mode="json"),
            "available_rows": len(normalized),
            "scanned_rows": scanned_rows,
            "eligible_rows": len(eligible),
            "selected_rows": len(selected),
            "train_rows": config.data.train_size_per_domain,
            "validation_rows": config.data.validation_size_per_domain,
            "excluded_rows": dict(sorted(exclusions.items())),
        }

    records.sort(key=lambda record: (record.domain.value, record.source_id))
    train_records = [record for record in records if record.split is TrainingSplit.TRAIN]
    validation_records = [record for record in records if record.split is TrainingSplit.VALIDATION]
    train_bytes = _records_bytes(train_records)
    validation_bytes = _records_bytes(validation_records)
    train_sha256 = sha256_bytes(train_bytes)
    validation_sha256 = sha256_bytes(validation_bytes)

    destination = output_dir or Path(config.data.output_dir)
    train_path = destination / "train.jsonl"
    validation_path = destination / "validation.jsonl"
    manifest_path = destination / "manifest.json"
    _write_atomic(train_path, train_bytes)
    _write_atomic(validation_path, validation_bytes)

    counts = Counter(record.domain.value for record in records)
    manifest = {
        "schema_version": config.schema_version,
        "training_config_sha256": config.fingerprint(),
        "seed": config.data.seed,
        "pilot_size_per_domain": config.data.pilot_size_per_domain,
        "train_size_per_domain": config.data.train_size_per_domain,
        "validation_size_per_domain": config.data.validation_size_per_domain,
        "row_count": len(records),
        "rows_by_domain": dict(sorted(counts.items())),
        "benchmark": {
            "path": config.data.benchmark_path,
            "manifest_path": config.data.benchmark_manifest_path,
            "sha256": leakage.benchmark_sha256,
        },
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
        "sources": source_statistics,
    }
    _write_atomic(manifest_path, (canonical_json(manifest) + "\n").encode("utf-8"))
    return PreparedTrainingData(
        train_path=train_path,
        validation_path=validation_path,
        manifest_path=manifest_path,
        train_sha256=train_sha256,
        validation_sha256=validation_sha256,
        train_row_count=len(train_records),
        validation_row_count=len(validation_records),
    )


def load_source_training_records(path: Path) -> list[SourceTrainingRecord]:
    """Read and validate source training records and their content hashes."""

    records: list[SourceTrainingRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = validate_source_training_record(json.loads(line))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid source training row at {path}:{line_number}") from error
            if record.content_sha256 != normalized_content_sha256(record.example):
                raise ValueError(f"source training content hash mismatch at {path}:{line_number}")
            records.append(record)

    source_ids = [record.source_id for record in records]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError(f"source training data contains duplicate source IDs: {path}")
    return records
