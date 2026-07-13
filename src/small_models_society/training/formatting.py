"""Prompt-completion formatting and exact completion-only token masks."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from small_models_society.data.prepare import canonical_json, sha256_bytes
from small_models_society.inference.contracts import ChatMessage, to_inference_example
from small_models_society.inference.prompts import (
    PromptCatalog,
    PromptProfileName,
    render_messages,
)
from small_models_society.schemas import (
    BenchmarkExample,
    CodeExample,
    KnowledgeExample,
    LogicExample,
    MathExample,
)
from small_models_society.training.config import TrainingConfig
from small_models_society.training.contracts import (
    SFTTrainingRecord,
    SourceTrainingRecord,
    TrainingSplit,
    validate_sft_training_record,
)
from small_models_society.training.prepare import load_source_training_records

IGNORE_INDEX = -100


class SFTFormatError(ValueError):
    """Raised when a source reference cannot form a valid SFT completion."""


class SFTLengthError(SFTFormatError):
    """Raised when a fully templated prompt-completion row exceeds its token budget."""


class ChatTemplateTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        **kwargs: Any,
    ) -> str: ...

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class SFTTokenization:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class PreparedSFTData:
    train_path: Path
    validation_path: Path
    manifest_path: Path
    train_sha256: str
    validation_sha256: str
    train_row_count: int
    validation_row_count: int


def _token_ids(value: Any) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise SFTFormatError("tokenizer did not return a one-dimensional integer input_ids list")
    return tuple(cast(list[int], value))


def _render_chat(
    tokenizer: ChatTemplateTokenizer,
    messages: list[ChatMessage],
    *,
    add_generation_prompt: bool,
) -> tuple[int, ...]:
    text = tokenizer.apply_chat_template(
        [message.model_dump(mode="json") for message in messages],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(text, str):
        raise SFTFormatError("chat template did not return text")
    encoded = tokenizer(text, add_special_tokens=False)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise SFTFormatError("tokenizer result is missing input_ids")
    return _token_ids(encoded["input_ids"])


def tokenize_prompt_completion(
    prompt: list[ChatMessage],
    completion: list[ChatMessage],
    tokenizer: ChatTemplateTokenizer,
    max_length: int,
) -> SFTTokenization:
    """Tokenize a conversation and mask every token before the assistant completion."""

    prompt_ids = _render_chat(tokenizer, prompt, add_generation_prompt=True)
    full_ids = _render_chat(
        tokenizer,
        [*prompt, *completion],
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise SFTFormatError(
            "full conversation tokens do not preserve the generation prompt prefix"
        )
    if len(full_ids) > max_length:
        raise SFTLengthError(
            f"fully templated conversation has {len(full_ids)} tokens; max_length is {max_length}"
        )
    completion_tokens = len(full_ids) - len(prompt_ids)
    if completion_tokens <= 0:
        raise SFTFormatError("assistant completion produced no trainable tokens")
    return SFTTokenization(
        input_ids=full_ids,
        labels=(IGNORE_INDEX,) * len(prompt_ids) + full_ids[len(prompt_ids) :],
        prompt_tokens=len(prompt_ids),
        completion_tokens=completion_tokens,
    )


def render_completion(example: BenchmarkExample) -> str:
    """Render only the gold assistant completion for one normalized source example."""

    if isinstance(example, MathExample):
        final_answer = f"Final answer: {example.reference.answer.strip()}"
        rationale = (example.reference.rationale or "").strip()
        return f"{rationale}\n\n{final_answer}" if rationale else final_answer
    if isinstance(example, CodeExample):
        solution = (example.reference.canonical_solution or "").strip()
        if not solution:
            raise SFTFormatError("code source example is missing a canonical solution")
        return solution
    if isinstance(example, LogicExample):
        return example.reference.answer_label.strip()
    if isinstance(example, KnowledgeExample):
        answer = example.reference.answers[0].strip()
        if not answer:
            raise SFTFormatError("knowledge source example has an empty accepted answer")
        return answer
    raise TypeError(f"unsupported source example: {type(example).__name__}")


def _conversation(
    example: BenchmarkExample,
    catalog: PromptCatalog,
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    prompt = render_messages(
        to_inference_example(example),
        catalog,
        PromptProfileName.GENERAL,
    )
    completion = [ChatMessage(role="assistant", content=render_completion(example))]
    return prompt, completion


def format_source_record(
    record: SourceTrainingRecord,
    catalog: PromptCatalog,
    tokenizer: ChatTemplateTokenizer,
    max_length: int,
) -> SFTTrainingRecord:
    prompt, completion = _conversation(record.example, catalog)
    tokenization = tokenize_prompt_completion(prompt, completion, tokenizer, max_length)
    return SFTTrainingRecord(
        source_id=record.source_id,
        domain=record.domain,
        split=record.split,
        source_content_sha256=record.content_sha256,
        prompt=prompt,
        completion=completion,
        prompt_tokens=tokenization.prompt_tokens,
        completion_tokens=tokenization.completion_tokens,
    )


def build_sft_eligibility_filter(
    catalog: PromptCatalog,
    tokenizer: ChatTemplateTokenizer,
    max_length: int,
) -> Callable[[BenchmarkExample], bool]:
    """Return an exact preselection filter that rejects only overlength conversations."""

    def is_eligible(example: BenchmarkExample) -> bool:
        prompt, completion = _conversation(example, catalog)
        try:
            tokenize_prompt_completion(prompt, completion, tokenizer, max_length)
        except SFTLengthError:
            return False
        return True

    return is_eligible


def _records_bytes(records: Iterable[SFTTrainingRecord]) -> bytes:
    lines = [canonical_json(record.model_dump(mode="json")) for record in records]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def _manifest_mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return cast(Mapping[str, Any], value)


def _verify_source_file(
    path: Path,
    file_manifest: object,
    expected_split: TrainingSplit,
) -> list[SourceTrainingRecord]:
    metadata = _manifest_mapping(file_manifest, f"source {expected_split.value} file metadata")
    actual_sha256 = sha256_bytes(path.read_bytes())
    if metadata.get("sha256") != actual_sha256:
        raise ValueError(f"source {expected_split.value} hash does not match source manifest")
    records = load_source_training_records(path)
    if any(record.split is not expected_split for record in records):
        raise ValueError(f"source {expected_split.value} file contains the wrong split")
    if metadata.get("row_count") != len(records):
        raise ValueError(f"source {expected_split.value} row count does not match source manifest")
    return records


def prepare_sft_data(
    config: TrainingConfig,
    catalog: PromptCatalog,
    tokenizer: ChatTemplateTokenizer,
    source_train_path: Path,
    source_validation_path: Path,
    source_manifest_path: Path,
    output_dir: Path | None = None,
) -> PreparedSFTData:
    """Format verified source records as deterministic conversational SFT artifacts."""

    try:
        source_manifest_bytes = source_manifest_path.read_bytes()
        source_manifest = _manifest_mapping(
            json.loads(source_manifest_bytes),
            "source training manifest",
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid source training manifest: {source_manifest_path}") from error
    if not config.accepts_fingerprint(source_manifest.get("training_config_sha256")):
        raise ValueError("source training manifest uses a different training configuration")
    files = _manifest_mapping(source_manifest.get("files"), "source training files")
    train_source_records = _verify_source_file(
        source_train_path,
        files.get("train"),
        TrainingSplit.TRAIN,
    )
    validation_source_records = _verify_source_file(
        source_validation_path,
        files.get("validation"),
        TrainingSplit.VALIDATION,
    )

    train_records = [
        format_source_record(record, catalog, tokenizer, config.data.max_length)
        for record in train_source_records
    ]
    validation_records = [
        format_source_record(record, catalog, tokenizer, config.data.max_length)
        for record in validation_source_records
    ]
    train_bytes = _records_bytes(train_records)
    validation_bytes = _records_bytes(validation_records)
    train_sha256 = sha256_bytes(train_bytes)
    validation_sha256 = sha256_bytes(validation_bytes)

    destination = output_dir or Path(config.data.output_dir) / "sft"
    train_path = destination / "train.jsonl"
    validation_path = destination / "validation.jsonl"
    manifest_path = destination / "manifest.json"
    _write_atomic(train_path, train_bytes)
    _write_atomic(validation_path, validation_bytes)

    counts = Counter(record.domain.value for record in [*train_records, *validation_records])
    manifest = {
        "schema_version": config.schema_version,
        "training_config_sha256": config.fingerprint(),
        "prompt_catalog_sha256": catalog.fingerprint(),
        "prompt_profile": PromptProfileName.GENERAL.value,
        "completion_only_loss": config.sft.completion_only_loss,
        "max_length": config.data.max_length,
        "tokenizer": {
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "class": type(tokenizer).__name__,
        },
        "source_manifest": {
            "path": str(source_manifest_path),
            "sha256": sha256_bytes(source_manifest_bytes),
        },
        "row_count": len(train_records) + len(validation_records),
        "rows_by_domain": dict(sorted(counts.items())),
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
    _write_atomic(manifest_path, (canonical_json(manifest) + "\n").encode("utf-8"))
    return PreparedSFTData(
        train_path=train_path,
        validation_path=validation_path,
        manifest_path=manifest_path,
        train_sha256=train_sha256,
        validation_sha256=validation_sha256,
        train_row_count=len(train_records),
        validation_row_count=len(validation_records),
    )


def load_sft_training_records(path: Path) -> list[SFTTrainingRecord]:
    """Read and validate model-facing SFT records."""

    records: list[SFTTrainingRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(validate_sft_training_record(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid SFT row at {path}:{line_number}") from error
    source_ids = [record.source_id for record in records]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError(f"SFT data contains duplicate source IDs: {path}")
    return records
