"""Backend-neutral, reference-free contracts for model inference."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol, TypeAlias

from pydantic import Field, TypeAdapter

from small_models_society.schemas import (
    BenchmarkExample,
    CodeExample,
    CodeInput,
    Domain,
    KnowledgeExample,
    KnowledgeInput,
    LogicExample,
    LogicInput,
    MathExample,
    MathInput,
    StrictModel,
)


class MathInferenceExample(StrictModel):
    id: str = Field(min_length=1)
    domain: Literal[Domain.MATH] = Domain.MATH
    input: MathInput


class CodeInferenceExample(StrictModel):
    id: str = Field(min_length=1)
    domain: Literal[Domain.CODE] = Domain.CODE
    input: CodeInput


class LogicInferenceExample(StrictModel):
    id: str = Field(min_length=1)
    domain: Literal[Domain.LOGIC] = Domain.LOGIC
    input: LogicInput


class KnowledgeInferenceExample(StrictModel):
    id: str = Field(min_length=1)
    domain: Literal[Domain.KNOWLEDGE] = Domain.KNOWLEDGE
    input: KnowledgeInput


InferenceExample: TypeAlias = Annotated[
    MathInferenceExample | CodeInferenceExample | LogicInferenceExample | KnowledgeInferenceExample,
    Field(discriminator="domain"),
]
_INFERENCE_EXAMPLE_ADAPTER: TypeAdapter[InferenceExample] = TypeAdapter(InferenceExample)


def validate_inference_example(value: object) -> InferenceExample:
    """Validate unknown data as a model-facing example."""

    return _INFERENCE_EXAMPLE_ADAPTER.validate_python(value)


def to_inference_example(example: BenchmarkExample) -> InferenceExample:
    """Project a benchmark example into the only shape model backends may receive."""

    if isinstance(example, MathExample):
        return MathInferenceExample(id=example.id, input=example.input.model_copy(deep=True))
    if isinstance(example, CodeExample):
        return CodeInferenceExample(id=example.id, input=example.input.model_copy(deep=True))
    if isinstance(example, LogicExample):
        return LogicInferenceExample(id=example.id, input=example.input.model_copy(deep=True))
    if isinstance(example, KnowledgeExample):
        return KnowledgeInferenceExample(id=example.id, input=example.input.model_copy(deep=True))
    raise TypeError(f"unsupported example type: {type(example).__name__}")


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class AdapterReference(StrictModel):
    name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class GenerationRequest(StrictModel):
    request_id: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    adapter: str | None = Field(default=None, min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    max_new_tokens: int = Field(gt=0)


class GenerationOutput(StrictModel):
    text: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextGenerationBackend(Protocol):
    """Minimal interface shared by real and test generation backends."""

    def generate(self, request: GenerationRequest) -> GenerationOutput: ...
