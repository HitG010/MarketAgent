"""Versioned prompt profiles and deterministic domain rendering."""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from small_models_society.data.prepare import canonical_json
from small_models_society.inference.contracts import (
    ChatMessage,
    CodeInferenceExample,
    GenerationRequest,
    InferenceExample,
    KnowledgeInferenceExample,
    LogicInferenceExample,
    MathInferenceExample,
)
from small_models_society.schemas import Domain, StrictModel


class PromptProfileName(StrEnum):
    GENERAL = "general"
    MATH = "math"
    CODE = "code"
    LOGIC = "logic"
    KNOWLEDGE = "knowledge"


class PromptProfile(StrictModel):
    description: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)


class PromptCatalog(StrictModel):
    schema_version: Literal[1] = 1
    profiles: dict[PromptProfileName, PromptProfile]

    @model_validator(mode="after")
    def require_every_profile(self) -> Self:
        missing = set(PromptProfileName) - set(self.profiles)
        extra = set(self.profiles) - set(PromptProfileName)
        if missing or extra:
            raise ValueError(
                "profiles must contain exactly every prompt profile; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return self

    def fingerprint(self) -> str:
        payload = canonical_json(self.model_dump(mode="json")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, profile: PromptProfileName | str) -> PromptProfile:
        try:
            profile_name = PromptProfileName(profile)
        except ValueError as error:
            raise ValueError(f"unknown prompt profile: {profile}") from error
        return self.profiles[profile_name]


def load_prompt_catalog(path: Path) -> PromptCatalog:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return PromptCatalog.model_validate(value)


def _render_math(example: MathInferenceExample) -> str:
    return (
        "Problem:\n"
        f"{example.input.question}\n\n"
        "Solve the problem. End with the final numeric answer."
    )


def _render_code(example: CodeInferenceExample) -> str:
    entry_point = example.input.entry_point or "not specified"
    return (
        "Task:\n"
        f"{example.input.prompt}\n\n"
        f"Required entry point: {entry_point}\n"
        "Return only executable Python source code without Markdown fences or explanation."
    )


def _render_logic(example: LogicInferenceExample) -> str:
    choices = "\n".join(f"{choice.label}. {choice.text}" for choice in example.input.choices)
    labels = ", ".join(choice.label for choice in example.input.choices)
    return (
        "Question:\n"
        f"{example.input.question}\n\n"
        "Choices:\n"
        f"{choices}\n\n"
        f"Return exactly one final choice label from: {labels}."
    )


def _render_knowledge(example: KnowledgeInferenceExample) -> str:
    evidence = "\n\n".join(
        f"[{index}] {passage}" for index, passage in enumerate(example.input.context, start=1)
    )
    return (
        "Question:\n"
        f"{example.input.question}\n\n"
        "Answer the question using only the evidence below. Give a concise answer.\n\n"
        "Evidence:\n"
        f"{evidence}"
    )


def render_messages(
    example: InferenceExample,
    catalog: PromptCatalog,
    profile: PromptProfileName | str,
) -> list[ChatMessage]:
    prompt_profile = catalog.get(profile)
    if isinstance(example, MathInferenceExample):
        user_prompt = _render_math(example)
    elif isinstance(example, CodeInferenceExample):
        user_prompt = _render_code(example)
    elif isinstance(example, LogicInferenceExample):
        user_prompt = _render_logic(example)
    elif isinstance(example, KnowledgeInferenceExample):
        user_prompt = _render_knowledge(example)
    else:
        raise TypeError(f"unsupported inference example: {type(example).__name__}")
    return [
        ChatMessage(role="system", content=prompt_profile.system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]


def render_generation_request(
    example: InferenceExample,
    catalog: PromptCatalog,
    profile: PromptProfileName | str,
    max_new_tokens: int,
) -> GenerationRequest:
    profile_name = PromptProfileName(profile)
    return GenerationRequest(
        example=example,
        profile=profile_name.value,
        messages=render_messages(example, catalog, profile_name),
        max_new_tokens=max_new_tokens,
    )


_OUTER_CODE_FENCE = re.compile(
    r"\A\s*```(?:python|py)?[ \t]*\r?\n(?P<code>.*?)\r?\n```\s*\Z",
    flags=re.DOTALL | re.IGNORECASE,
)


def clean_response(domain: Domain, response: str) -> str:
    """Normalize transport formatting without altering answer semantics."""

    stripped = response.strip()
    if domain is not Domain.CODE:
        return stripped
    match = _OUTER_CODE_FENCE.fullmatch(response)
    return match.group("code").strip() if match else stripped
