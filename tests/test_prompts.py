from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from small_models_society.data.prepare import load_benchmark
from small_models_society.inference.contracts import to_inference_example
from small_models_society.inference.prompts import (
    PromptCatalog,
    PromptProfileName,
    clean_response,
    load_prompt_catalog,
    render_generation_request,
    render_messages,
)
from small_models_society.schemas import Domain

FIXTURE_BENCHMARK = Path(__file__).parent / "fixtures" / "benchmark.jsonl"
PROMPT_CONFIG = Path(__file__).parents[1] / "configs" / "prompt_profiles.yaml"


@pytest.fixture(scope="module")
def catalog() -> PromptCatalog:
    return load_prompt_catalog(PROMPT_CONFIG)


def test_catalog_contains_every_versioned_profile(catalog: PromptCatalog) -> None:
    assert set(catalog.profiles) == set(PromptProfileName)
    assert len(catalog.fingerprint()) == 64


def test_every_profile_renders_every_domain(catalog: PromptCatalog) -> None:
    examples = [to_inference_example(example) for example in load_benchmark(FIXTURE_BENCHMARK)]

    for profile in PromptProfileName:
        for example in examples:
            request = render_generation_request(example, catalog, profile, 128)
            serialized = request.model_dump_json()
            assert request.profile == profile.value
            assert [message.role for message in request.messages] == ["system", "user"]
            assert '"reference"' not in serialized
            assert '"metadata"' not in serialized


def test_math_prompt_is_stable(catalog: PromptCatalog) -> None:
    example = to_inference_example(load_benchmark(FIXTURE_BENCHMARK)[0])

    messages = render_messages(example, catalog, PromptProfileName.GENERAL)

    assert messages[1].content == (
        "Problem:\nWhat is 20% of 50?\n\nSolve the problem. End with the final numeric answer."
    )


def test_code_prompt_requires_raw_python(catalog: PromptCatalog) -> None:
    example = to_inference_example(load_benchmark(FIXTURE_BENCHMARK)[1])

    messages = render_messages(example, catalog, PromptProfileName.CODE)

    assert "Required entry point: add" in messages[1].content
    assert "without Markdown fences or explanation" in messages[1].content


def test_logic_prompt_lists_labels_and_choices(catalog: PromptCatalog) -> None:
    example = to_inference_example(load_benchmark(FIXTURE_BENCHMARK)[2])

    messages = render_messages(example, catalog, PromptProfileName.LOGIC)

    assert "A. Paris\nB. Rome\nC. Madrid" in messages[1].content
    assert messages[1].content.endswith("Return exactly one final choice label from: A, B, C.")


def test_knowledge_prompt_places_question_before_evidence(catalog: PromptCatalog) -> None:
    example = to_inference_example(load_benchmark(FIXTURE_BENCHMARK)[3])

    messages = render_messages(example, catalog, PromptProfileName.KNOWLEDGE)
    user_prompt = messages[1].content

    assert user_prompt.index("What is the capital of France?") < user_prompt.index("Evidence:")
    assert "[1] France\nParis is the capital of France." in user_prompt


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("```python\ndef add(a, b):\n    return a + b\n```", "def add(a, b):\n    return a + b"),
        ("```py\nprint('ok')\n```", "print('ok')"),
        ("```\nprint('ok')\n```", "print('ok')"),
        ("Text before\n```python\nprint('ok')\n```", "Text before\n```python\nprint('ok')\n```"),
    ],
)
def test_code_cleanup_removes_only_an_outer_fence(response: str, expected: str) -> None:
    assert clean_response(Domain.CODE, response) == expected


def test_non_code_cleanup_only_strips_outer_whitespace() -> None:
    assert clean_response(Domain.MATH, "  Final answer: 4\n") == "Final answer: 4"


def test_catalog_rejects_missing_profile(catalog: PromptCatalog) -> None:
    value = catalog.model_dump(mode="json")
    del value["profiles"][PromptProfileName.CODE.value]

    with pytest.raises(ValidationError, match="exactly every prompt profile"):
        PromptCatalog.model_validate(value)


def test_unknown_profile_is_rejected(catalog: PromptCatalog) -> None:
    with pytest.raises(ValueError, match="unknown prompt profile"):
        catalog.get("unknown")
