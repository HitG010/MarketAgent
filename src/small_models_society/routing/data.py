"""Leakage-free preparation of routing development and test artifacts."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import Field, model_validator

from small_models_society.data.config import DatasetSource
from small_models_society.data.loaders import SourceRow, load_source_rows, normalize_row
from small_models_society.data.prepare import canonical_json, sha256_bytes
from small_models_society.inference.contracts import (
    ChatMessage,
    CodeInferenceExample,
    InferenceExample,
    KnowledgeInferenceExample,
    LogicInferenceExample,
    MathInferenceExample,
    to_inference_example,
)
from small_models_society.inference.prompts import (
    PromptCatalog,
    PromptProfileName,
    render_messages,
)
from small_models_society.routing.artifacts import (
    RoutingArtifact,
    load_workflow_requests,
    write_workflow_requests,
)
from small_models_society.routing.config import RoutingConfig
from small_models_society.routing.contracts import (
    OutputContract,
    RequestPolicyContext,
    WorkflowRequest,
    create_workflow_request,
)
from small_models_society.schemas import BenchmarkExample, Domain, StrictModel
from small_models_society.training.contracts import SourceTrainingRecord, TrainingSplit
from small_models_society.training.prepare import (
    load_benchmark_leakage_index,
    load_source_training_records,
    normalized_content_sha256,
    qualified_source_id,
)

RowLoader = Callable[[DatasetSource], Iterable[SourceRow]]


class RoutingSplit(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"


class RoutingEvaluatorRecord(StrictModel):
    schema_version: Literal[1] = 1
    request_id: str = Field(min_length=1)
    split: RoutingSplit
    source_id: str = Field(min_length=1)
    source_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    example: BenchmarkExample

    @model_validator(mode="after")
    def validate_evaluator_identity(self) -> Self:
        if self.example.id != self.request_id:
            raise ValueError("evaluator example ID must match request ID")
        if normalized_content_sha256(self.example) != self.source_content_sha256:
            raise ValueError("evaluator source content hash does not match example")
        return self


class CalculatorEvaluatorRecord(StrictModel):
    schema_version: Literal[1] = 1
    request_id: str = Field(min_length=1)
    split: RoutingSplit
    expression: str = Field(min_length=1)
    expected_output: str = Field(min_length=1)
    case_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_case_fingerprint(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"case_fingerprint"})
        expected = sha256_bytes(canonical_json(payload).encode("utf-8"))
        if self.case_fingerprint != expected:
            raise ValueError("calculator case fingerprint does not match contents")
        return self


@dataclass(frozen=True)
class TrainingExclusionIndex:
    source_ids: frozenset[str]
    content_sha256s: frozenset[str]
    manifest_sha256: str
    train_sha256: str
    validation_sha256: str


@dataclass(frozen=True)
class PreparedRoutingData:
    development_requests_path: Path
    development_evaluator_path: Path
    test_requests_path: Path
    test_evaluator_path: Path
    calculator_development_requests_path: Path
    calculator_development_evaluator_path: Path
    calculator_test_requests_path: Path
    calculator_test_evaluator_path: Path
    manifest_path: Path
    development_row_count: int
    test_row_count: int
    development_requests_sha256: str
    test_requests_sha256: str


@dataclass(frozen=True)
class VerifiedRoutingSplit:
    split: RoutingSplit
    requests: tuple[WorkflowRequest, ...]
    evaluator_records: tuple[RoutingEvaluatorRecord, ...]
    requests_sha256: str
    evaluator_sha256: str
    routing_config_fingerprint: str
    prompt_catalog_fingerprint: str


_CALCULATOR_CASES: dict[RoutingSplit, tuple[tuple[str, str], ...]] = {
    RoutingSplit.DEVELOPMENT: (
        ("7 + 5", "12"),
        ("18 / 3", "6"),
        ("(4 + 6) * 3", "30"),
        ("2 ** 8", "256"),
        ("-7 + 2", "-5"),
        ("3.5 * 2", "7"),
    ),
    RoutingSplit.TEST: (
        ("42 - 19", "23"),
        ("8 * (3 + 2)", "40"),
        ("81 / 9 + 4", "13"),
        ("2 ** 5 - 3", "29"),
        ("-(6 - 10)", "4"),
        ("0.125 * 8", "1"),
    ),
}


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return cast(Mapping[str, Any], value)


def _verified_training_file(
    path: Path,
    metadata: object,
    description: str,
    expected_split: TrainingSplit,
) -> tuple[list[SourceTrainingRecord], str]:
    file_metadata = _mapping(metadata, f"{description} metadata")
    if file_metadata.get("path") != path.name:
        raise ValueError(f"{description} path does not match training manifest")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read {description}: {path}") from error
    actual_sha256 = sha256_bytes(content)
    if file_metadata.get("sha256") != actual_sha256:
        raise ValueError(f"{description} hash does not match training manifest")
    records = load_source_training_records(path)
    if file_metadata.get("row_count") != len(records):
        raise ValueError(f"{description} row count does not match training manifest")
    if any(record.split is not expected_split for record in records):
        raise ValueError(f"{description} contains rows from the wrong split")
    return records, actual_sha256


def load_training_exclusion_index(
    train_path: Path,
    validation_path: Path,
    manifest_path: Path,
) -> TrainingExclusionIndex:
    """Load selected Phase 3 rows after verifying their manifest hashes."""

    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = _mapping(json.loads(manifest_bytes), "training manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid training manifest: {manifest_path}") from error
    files = _mapping(manifest.get("files"), "training files")
    train_records, train_sha256 = _verified_training_file(
        train_path,
        files.get("train"),
        "training train file",
        TrainingSplit.TRAIN,
    )
    validation_records, validation_sha256 = _verified_training_file(
        validation_path,
        files.get("validation"),
        "training validation file",
        TrainingSplit.VALIDATION,
    )
    records = [*train_records, *validation_records]
    if manifest.get("row_count") != len(records):
        raise ValueError("training row count does not match training manifest")
    source_ids = [record.source_id for record in records]
    content_sha256s = [record.content_sha256 for record in records]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Phase 3 training exclusions contain duplicate source IDs")
    if len(set(content_sha256s)) != len(content_sha256s):
        raise ValueError("Phase 3 training exclusions contain duplicate content hashes")
    return TrainingExclusionIndex(
        source_ids=frozenset(source_ids),
        content_sha256s=frozenset(content_sha256s),
        manifest_sha256=sha256_bytes(manifest_bytes),
        train_sha256=train_sha256,
        validation_sha256=validation_sha256,
    )


def _sample_key(source_id: str, seed: int) -> str:
    return sha256_bytes(f"{seed}:routing:{source_id}".encode())


def _request_id(source_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{source_id}".encode()).hexdigest()
    return f"routing-{digest[:24]}"


def _with_request_id(example: BenchmarkExample, request_id: str) -> BenchmarkExample:
    return example.model_copy(update={"id": request_id}, deep=True)


def _request_inference_example(example: BenchmarkExample) -> InferenceExample:
    inference_example = to_inference_example(example)
    if isinstance(inference_example, KnowledgeInferenceExample):
        question_only = inference_example.input.model_copy(update={"context": []})
        return inference_example.model_copy(update={"input": question_only})
    return inference_example


def _output_contract(example: InferenceExample) -> OutputContract:
    if isinstance(example, MathInferenceExample):
        return OutputContract.NUMERIC
    if isinstance(example, CodeInferenceExample):
        return OutputContract.PYTHON_SOURCE
    if isinstance(example, LogicInferenceExample):
        return OutputContract.CHOICE_LABEL
    if isinstance(example, KnowledgeInferenceExample):
        return OutputContract.SHORT_ANSWER
    raise TypeError(f"unsupported inference example: {type(example).__name__}")


def _policy(config: RoutingConfig) -> RequestPolicyContext:
    defaults = config.policy_defaults
    return RequestPolicyContext(
        data_classification=defaults.data_classification,
        network_allowed=defaults.network_allowed,
        allowed_corpus_ids=defaults.allowed_corpus_ids,
        allowed_tool_ids=defaults.allowed_tool_ids,
        required_quality=defaults.required_quality,
        allow_unknown_output_safety=defaults.allow_unknown_output_safety,
    )


def _workflow_request(
    example: BenchmarkExample,
    catalog: PromptCatalog,
    config: RoutingConfig,
) -> WorkflowRequest:
    inference_example = _request_inference_example(example)
    attributes = (
        {"retrieval_query": inference_example.input.question}
        if isinstance(inference_example, KnowledgeInferenceExample)
        else None
    )
    return create_workflow_request(
        request_id=example.id,
        messages=tuple(render_messages(inference_example, catalog, PromptProfileName.GENERAL)),
        output_contract=_output_contract(inference_example),
        policy=_policy(config),
        attributes=attributes,
    )


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_evaluator_records(
    path: Path,
    records: Sequence[StrictModel],
) -> RoutingArtifact:
    if not records:
        raise ValueError("routing evaluator artifact requires at least one row")
    lines = [canonical_json(record.model_dump(mode="json")) for record in records]
    content = ("\n".join(lines) + "\n").encode("utf-8")
    _write_atomic(path, content)
    return RoutingArtifact(path=path, sha256=sha256_bytes(content), row_count=len(records))


def load_routing_evaluator_records(path: Path) -> list[RoutingEvaluatorRecord]:
    records: list[RoutingEvaluatorRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(RoutingEvaluatorRecord.model_validate_json(line))
            except ValueError as error:
                raise ValueError(
                    f"invalid routing evaluator row at {path}:{line_number}"
                ) from error
    if not records:
        raise ValueError(f"routing evaluator artifact contains no rows: {path}")
    request_ids = [record.request_id for record in records]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError(f"routing evaluator artifact contains duplicate request IDs: {path}")
    return records


def load_calculator_evaluator_records(path: Path) -> list[CalculatorEvaluatorRecord]:
    records: list[CalculatorEvaluatorRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(CalculatorEvaluatorRecord.model_validate_json(line))
            except ValueError as error:
                raise ValueError(
                    f"invalid calculator evaluator row at {path}:{line_number}"
                ) from error
    if not records:
        raise ValueError(f"calculator evaluator artifact contains no rows: {path}")
    request_ids = [record.request_id for record in records]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError(f"calculator evaluator artifact has duplicate request IDs: {path}")
    return records


def load_verified_routing_split(
    requests_path: Path,
    evaluator_path: Path,
    manifest_path: Path,
    split: RoutingSplit,
) -> VerifiedRoutingSplit:
    """Load one routing split only after verifying its paired manifest entries."""

    try:
        manifest = _mapping(
            json.loads(manifest_path.read_bytes()),
            "routing data manifest",
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid routing data manifest: {manifest_path}") from error
    files = _mapping(manifest.get("files"), "routing data files")
    request_metadata = _mapping(
        files.get(f"{split.value}_requests"),
        "routing request metadata",
    )
    evaluator_metadata = _mapping(
        files.get(f"{split.value}_evaluator"),
        "routing evaluator metadata",
    )
    try:
        request_bytes = requests_path.read_bytes()
        evaluator_bytes = evaluator_path.read_bytes()
    except OSError as error:
        raise ValueError("unable to read routing split artifacts") from error
    requests_sha256 = sha256_bytes(request_bytes)
    evaluator_sha256 = sha256_bytes(evaluator_bytes)
    checks = (
        (request_metadata, requests_path, requests_sha256, "request"),
        (evaluator_metadata, evaluator_path, evaluator_sha256, "evaluator"),
    )
    for metadata, path, actual_sha256, description in checks:
        if metadata.get("path") != path.name:
            raise ValueError(f"routing {description} path does not match manifest")
        if metadata.get("sha256") != actual_sha256:
            raise ValueError(f"routing {description} hash does not match manifest")

    requests = load_workflow_requests(requests_path)
    evaluators = load_routing_evaluator_records(evaluator_path)
    if request_metadata.get("row_count") != len(requests):
        raise ValueError("routing request count does not match manifest")
    if evaluator_metadata.get("row_count") != len(evaluators):
        raise ValueError("routing evaluator count does not match manifest")
    if any(record.split is not split for record in evaluators):
        raise ValueError("routing evaluator contains records from another split")
    request_ids = {request.request_id for request in requests}
    evaluator_ids = {record.request_id for record in evaluators}
    if request_ids != evaluator_ids:
        raise ValueError("routing requests and evaluator records have different IDs")
    routing_fingerprint = manifest.get("routing_config_fingerprint")
    prompt_fingerprint = manifest.get("prompt_catalog_fingerprint")
    if not isinstance(routing_fingerprint, str) or len(routing_fingerprint) != 64:
        raise ValueError("routing data manifest has an invalid config fingerprint")
    if not isinstance(prompt_fingerprint, str) or len(prompt_fingerprint) != 64:
        raise ValueError("routing data manifest has an invalid prompt fingerprint")
    return VerifiedRoutingSplit(
        split=split,
        requests=tuple(requests),
        evaluator_records=tuple(evaluators),
        requests_sha256=requests_sha256,
        evaluator_sha256=evaluator_sha256,
        routing_config_fingerprint=routing_fingerprint,
        prompt_catalog_fingerprint=prompt_fingerprint,
    )


def _calculator_case(
    split: RoutingSplit,
    expression: str,
    expected_output: str,
) -> CalculatorEvaluatorRecord:
    request_id = f"calculator-{split.value}-{sha256_bytes(expression.encode())[:16]}"
    values = {
        "schema_version": 1,
        "request_id": request_id,
        "split": split.value,
        "expression": expression,
        "expected_output": expected_output,
    }
    return CalculatorEvaluatorRecord.model_validate(
        {
            **values,
            "case_fingerprint": sha256_bytes(canonical_json(values).encode("utf-8")),
        }
    )


def _calculator_suite(
    config: RoutingConfig,
) -> tuple[
    dict[RoutingSplit, list[WorkflowRequest]], dict[RoutingSplit, list[CalculatorEvaluatorRecord]]
]:
    requests: dict[RoutingSplit, list[WorkflowRequest]] = {split: [] for split in RoutingSplit}
    evaluators: dict[RoutingSplit, list[CalculatorEvaluatorRecord]] = {
        split: [] for split in RoutingSplit
    }
    system_message = ChatMessage(
        role="system",
        content="Evaluate the arithmetic expression. Return only the exact numeric result.",
    )
    for split in RoutingSplit:
        for expression, expected_output in _CALCULATOR_CASES[split]:
            evaluator = _calculator_case(split, expression, expected_output)
            evaluators[split].append(evaluator)
            requests[split].append(
                create_workflow_request(
                    request_id=evaluator.request_id,
                    messages=(
                        system_message,
                        ChatMessage(role="user", content=expression),
                    ),
                    output_contract=OutputContract.NUMERIC,
                    policy=_policy(config),
                )
            )
    return requests, evaluators


def _expected_paths(destination: Path) -> tuple[Path, ...]:
    return (
        destination / "development.requests.jsonl",
        destination / "development.evaluator.jsonl",
        destination / "test.requests.jsonl",
        destination / "test.evaluator.jsonl",
        destination / "calculator.development.requests.jsonl",
        destination / "calculator.development.evaluator.jsonl",
        destination / "calculator.test.requests.jsonl",
        destination / "calculator.test.evaluator.jsonl",
        destination / "manifest.json",
    )


def _artifact_manifest(artifact: RoutingArtifact) -> dict[str, object]:
    return {
        "path": artifact.path.name,
        "sha256": artifact.sha256,
        "row_count": artifact.row_count,
    }


def prepare_routing_data(
    config: RoutingConfig,
    catalog: PromptCatalog,
    output_dir: Path | None = None,
    row_loader: RowLoader | None = None,
    *,
    overwrite: bool = False,
) -> PreparedRoutingData:
    """Prepare opaque requests and hidden evaluator records for routing research."""

    destination = output_dir or Path(config.data.output_dir)
    if any(path.exists() for path in _expected_paths(destination)) and not overwrite:
        raise FileExistsError("routing data artifacts already exist; use overwrite explicitly")

    benchmark = load_benchmark_leakage_index(
        Path(config.data.benchmark_path),
        Path(config.data.benchmark_manifest_path),
    )
    training = load_training_exclusion_index(
        Path(config.data.training_train_path),
        Path(config.data.training_validation_path),
        Path(config.data.training_manifest_path),
    )
    loader = row_loader or (
        lambda source: load_source_rows(
            source,
            local_files_only=config.data.local_files_only,
        )
    )

    requests_by_split: dict[RoutingSplit, list[WorkflowRequest]] = {
        split: [] for split in RoutingSplit
    }
    evaluators_by_split: dict[RoutingSplit, list[RoutingEvaluatorRecord]] = {
        split: [] for split in RoutingSplit
    }
    source_statistics: dict[str, dict[str, object]] = {}
    selected_source_ids: set[str] = set()
    selected_content_sha256s: set[str] = set()
    selected_request_ids: set[str] = set()

    for domain in Domain:
        source = config.data.sources[domain]
        normalized = [normalize_row(domain, row, index) for index, row in enumerate(loader(source))]
        qualified = [qualified_source_id(source, example.id) for example in normalized]
        if len(set(qualified)) != len(qualified):
            raise ValueError(f"{domain.value} routing source contains duplicate normalized IDs")
        ranked = sorted(
            zip(qualified, normalized, strict=True),
            key=lambda item: _sample_key(item[0], config.seed),
        )
        exclusions: Counter[str] = Counter()
        eligible: list[tuple[str, BenchmarkExample, str]] = []
        seen_content: set[str] = set()
        for source_id, example in ranked:
            content_sha256 = normalized_content_sha256(example)
            if content_sha256 in seen_content:
                exclusions["duplicate_source_content"] += 1
                continue
            seen_content.add(content_sha256)
            if source_id in benchmark.source_ids:
                exclusions["phase1_source_id"] += 1
                continue
            if content_sha256 in benchmark.content_sha256s:
                exclusions["phase1_content"] += 1
                continue
            if source_id in training.source_ids:
                exclusions["phase3_source_id"] += 1
                continue
            if content_sha256 in training.content_sha256s:
                exclusions["phase3_content"] += 1
                continue
            eligible.append((source_id, example, content_sha256))

        requested = config.data.development_size_per_domain + config.data.test_size_per_domain
        if len(eligible) < requested:
            raise ValueError(
                f"{domain.value} has {len(eligible)} eligible routing rows after exclusions; "
                f"{requested} requested"
            )
        selected = eligible[:requested]
        for index, (source_id, original_example, content_sha256) in enumerate(selected):
            split = (
                RoutingSplit.DEVELOPMENT
                if index < config.data.development_size_per_domain
                else RoutingSplit.TEST
            )
            request_id = _request_id(source_id, config.seed)
            if source_id in selected_source_ids or content_sha256 in selected_content_sha256s:
                raise ValueError("routing development and test selections overlap")
            if request_id in selected_request_ids:
                raise ValueError("routing request ID collision detected")
            selected_source_ids.add(source_id)
            selected_content_sha256s.add(content_sha256)
            selected_request_ids.add(request_id)

            evaluator_example = _with_request_id(original_example, request_id)
            requests_by_split[split].append(_workflow_request(evaluator_example, catalog, config))
            evaluators_by_split[split].append(
                RoutingEvaluatorRecord(
                    request_id=request_id,
                    split=split,
                    source_id=source_id,
                    source_content_sha256=content_sha256,
                    example=evaluator_example,
                )
            )

        source_statistics[domain.value] = {
            **source.model_dump(mode="json"),
            "available_rows": len(normalized),
            "eligible_rows": len(eligible),
            "selected_development_rows": config.data.development_size_per_domain,
            "selected_test_rows": config.data.test_size_per_domain,
            "excluded_rows": dict(sorted(exclusions.items())),
        }

    for split in RoutingSplit:
        requests_by_split[split].sort(key=lambda request: request.request_id)
        evaluators_by_split[split].sort(key=lambda record: record.request_id)

    development_requests = write_workflow_requests(
        destination / "development.requests.jsonl",
        requests_by_split[RoutingSplit.DEVELOPMENT],
    )
    development_evaluator = _write_evaluator_records(
        destination / "development.evaluator.jsonl",
        evaluators_by_split[RoutingSplit.DEVELOPMENT],
    )
    test_requests = write_workflow_requests(
        destination / "test.requests.jsonl",
        requests_by_split[RoutingSplit.TEST],
    )
    test_evaluator = _write_evaluator_records(
        destination / "test.evaluator.jsonl",
        evaluators_by_split[RoutingSplit.TEST],
    )
    calculator_requests, calculator_evaluators = _calculator_suite(config)
    calculator_development_requests = write_workflow_requests(
        destination / "calculator.development.requests.jsonl",
        calculator_requests[RoutingSplit.DEVELOPMENT],
    )
    calculator_development_evaluator = _write_evaluator_records(
        destination / "calculator.development.evaluator.jsonl",
        calculator_evaluators[RoutingSplit.DEVELOPMENT],
    )
    calculator_test_requests = write_workflow_requests(
        destination / "calculator.test.requests.jsonl",
        calculator_requests[RoutingSplit.TEST],
    )
    calculator_test_evaluator = _write_evaluator_records(
        destination / "calculator.test.evaluator.jsonl",
        calculator_evaluators[RoutingSplit.TEST],
    )
    manifest_path = destination / "manifest.json"
    manifest = {
        "schema_version": config.schema_version,
        "routing_config_fingerprint": config.fingerprint(),
        "prompt_catalog_fingerprint": catalog.fingerprint(),
        "seed": config.seed,
        "row_count": development_requests.row_count + test_requests.row_count,
        "rows_by_split": {
            RoutingSplit.DEVELOPMENT.value: development_requests.row_count,
            RoutingSplit.TEST.value: test_requests.row_count,
        },
        "exclusions": {
            "phase1_benchmark_sha256": benchmark.benchmark_sha256,
            "phase3_manifest_sha256": training.manifest_sha256,
            "phase3_train_sha256": training.train_sha256,
            "phase3_validation_sha256": training.validation_sha256,
        },
        "files": {
            "development_requests": _artifact_manifest(development_requests),
            "development_evaluator": _artifact_manifest(development_evaluator),
            "test_requests": _artifact_manifest(test_requests),
            "test_evaluator": _artifact_manifest(test_evaluator),
        },
        "companion_suites": {
            "calculator": {
                "included_in_four_domain_aggregate": False,
                "suite_version": 1,
                "rows_by_split": {
                    RoutingSplit.DEVELOPMENT.value: calculator_development_requests.row_count,
                    RoutingSplit.TEST.value: calculator_test_requests.row_count,
                },
                "files": {
                    "development_requests": _artifact_manifest(calculator_development_requests),
                    "development_evaluator": _artifact_manifest(calculator_development_evaluator),
                    "test_requests": _artifact_manifest(calculator_test_requests),
                    "test_evaluator": _artifact_manifest(calculator_test_evaluator),
                },
            }
        },
        "sources": source_statistics,
    }
    _write_atomic(manifest_path, (canonical_json(manifest) + "\n").encode("utf-8"))
    return PreparedRoutingData(
        development_requests_path=development_requests.path,
        development_evaluator_path=development_evaluator.path,
        test_requests_path=test_requests.path,
        test_evaluator_path=test_evaluator.path,
        calculator_development_requests_path=calculator_development_requests.path,
        calculator_development_evaluator_path=calculator_development_evaluator.path,
        calculator_test_requests_path=calculator_test_requests.path,
        calculator_test_evaluator_path=calculator_test_evaluator.path,
        manifest_path=manifest_path,
        development_row_count=development_requests.row_count,
        test_row_count=test_requests.row_count,
        development_requests_sha256=development_requests.sha256,
        test_requests_sha256=test_requests.sha256,
    )
