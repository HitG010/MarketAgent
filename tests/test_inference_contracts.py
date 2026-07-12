from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from small_models_society.data.prepare import load_benchmark
from small_models_society.inference.contracts import (
    ChatMessage,
    GenerationOutput,
    GenerationRequest,
    TextGenerationBackend,
    to_inference_example,
    validate_inference_example,
)
from small_models_society.schemas import Domain

FIXTURE_BENCHMARK = Path(__file__).parent / "fixtures" / "benchmark.jsonl"


def test_projects_every_domain_without_reference_or_metadata() -> None:
    examples = load_benchmark(FIXTURE_BENCHMARK)

    projected = [to_inference_example(example) for example in examples]

    assert [example.domain for example in projected] == list(Domain)
    for example in projected:
        serialized = example.model_dump_json()
        assert '"reference"' not in serialized
        assert '"metadata"' not in serialized
        assert set(json.loads(serialized)) == {"id", "domain", "input"}


def test_projection_deep_copies_model_input() -> None:
    benchmark_example = load_benchmark(FIXTURE_BENCHMARK)[0]
    projected = to_inference_example(benchmark_example)

    assert projected.input is not benchmark_example.input


def test_inference_example_rejects_injected_reference() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_inference_example(
            {
                "id": "math-1",
                "domain": "math",
                "input": {"question": "What is 2 + 2?"},
                "reference": {"answer": "4"},
            }
        )


def test_domain_and_input_shape_must_agree() -> None:
    with pytest.raises(ValidationError):
        validate_inference_example(
            {
                "id": "mismatched",
                "domain": "code",
                "input": {"question": "What is 2 + 2?"},
            }
        )


class EchoBackend:
    def generate(self, request: GenerationRequest) -> GenerationOutput:
        return GenerationOutput(
            text=request.messages[-1].content,
            prompt_tokens=4,
            completion_tokens=4,
            latency_ms=1,
        )


def test_backend_protocol_is_independent_of_transformers() -> None:
    example = to_inference_example(load_benchmark(FIXTURE_BENCHMARK)[0])
    request = GenerationRequest(
        example=example,
        profile="general",
        messages=[ChatMessage(role="user", content="Answer this question.")],
        max_new_tokens=16,
    )
    backend: TextGenerationBackend = EchoBackend()

    output = backend.generate(request)

    assert output.text == "Answer this question."
    assert output.prompt_tokens == 4
