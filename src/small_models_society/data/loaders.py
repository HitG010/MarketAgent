"""Source-specific conversion into the normalized benchmark contracts."""

from __future__ import annotations

import ast
import warnings
from collections.abc import Iterable, Mapping
from typing import Any, cast

from datasets import DownloadConfig, load_dataset  # type: ignore[import-untyped]

from small_models_society.data.config import DatasetSource
from small_models_society.schemas import (
    BenchmarkExample,
    Choice,
    CodeExample,
    CodeInput,
    CodeReference,
    Domain,
    KnowledgeExample,
    KnowledgeInput,
    KnowledgeReference,
    LogicExample,
    LogicInput,
    LogicReference,
    MathExample,
    MathInput,
    MathReference,
    SupportingFact,
)

SourceRow = Mapping[str, Any]


def load_source_rows(
    source: DatasetSource,
    *,
    local_files_only: bool = False,
) -> Iterable[SourceRow]:
    """Load one immutable Hugging Face dataset split."""

    dataset = load_dataset(
        source.dataset,
        source.config,
        split=source.split,
        revision=source.revision,
        download_config=DownloadConfig(local_files_only=local_files_only),
    )
    return cast(Iterable[SourceRow], dataset)


def _entry_point(code: str) -> str | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return None
    return next(
        (
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
    )


def normalize_math(row: SourceRow, index: int) -> MathExample:
    raw_answer = str(row["answer"])
    rationale, separator, answer = raw_answer.rpartition("####")
    if not separator:
        rationale, answer = "", raw_answer
    return MathExample(
        id=f"gsm8k-{index:05d}",
        input=MathInput(question=str(row["question"])),
        reference=MathReference(answer=answer.strip(), rationale=rationale.strip() or None),
    )


def normalize_code(row: SourceRow, index: int) -> CodeExample:
    del index
    canonical_solution = str(row["code"])
    imports = [str(item) for item in row.get("test_imports", [])]
    return CodeExample(
        id=f"mbpp-{int(row['task_id'])}",
        input=CodeInput(
            prompt=str(row["prompt"]),
            entry_point=_entry_point(canonical_solution),
        ),
        reference=CodeReference(
            test_setup="\n".join(imports),
            tests=[str(item) for item in row["test_list"]],
            canonical_solution=canonical_solution,
        ),
        metadata={"source_file": str(row.get("source_file", ""))},
    )


def normalize_logic(row: SourceRow, index: int) -> LogicExample:
    del index
    raw_choices = cast(Mapping[str, list[Any]], row["choices"])
    choices = [
        Choice(label=str(label), text=str(text))
        for label, text in zip(raw_choices["label"], raw_choices["text"], strict=True)
    ]
    return LogicExample(
        id=f"arc-{row['id']}",
        input=LogicInput(question=str(row["question"]), choices=choices),
        reference=LogicReference(answer_label=str(row["answerKey"])),
    )


def normalize_knowledge(row: SourceRow, index: int) -> KnowledgeExample:
    del index
    raw_context = cast(Mapping[str, list[Any]], row["context"])
    context = [
        f"{title}\n{''.join(str(sentence) for sentence in sentences)}"
        for title, sentences in zip(raw_context["title"], raw_context["sentences"], strict=True)
    ]
    raw_facts = cast(Mapping[str, list[Any]], row["supporting_facts"])
    supporting_facts = [
        SupportingFact(title=str(title), sentence_index=int(sentence_index))
        for title, sentence_index in zip(raw_facts["title"], raw_facts["sent_id"], strict=True)
    ]
    return KnowledgeExample(
        id=f"hotpot-{row['id']}",
        input=KnowledgeInput(question=str(row["question"]), context=context),
        reference=KnowledgeReference(
            answers=[str(row["answer"])],
            supporting_facts=supporting_facts,
        ),
        metadata={"level": str(row["level"]), "type": str(row["type"])},
    )


def normalize_row(domain: Domain, row: SourceRow, index: int) -> BenchmarkExample:
    if domain is Domain.MATH:
        return normalize_math(row, index)
    if domain is Domain.CODE:
        return normalize_code(row, index)
    if domain is Domain.LOGIC:
        return normalize_logic(row, index)
    if domain is Domain.KNOWLEDGE:
        return normalize_knowledge(row, index)
    raise ValueError(f"unsupported domain: {domain}")
