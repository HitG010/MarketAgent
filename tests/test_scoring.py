from __future__ import annotations

import pytest

from small_models_society.scoring.knowledge import normalize_answer, score_knowledge
from small_models_society.scoring.logic import extract_choice_label, score_choice
from small_models_society.scoring.math import extract_numeric_answer, score_math


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("The answer is $1,234.00.", "1234", 1.0),
        ("After calculating, I get 1/2", "0.5", 1.0),
        ("I considered 3, but the final answer is 4.", "4", 1.0),
        ("5", "4", 0.0),
        ("I cannot determine it", "4", 0.0),
    ],
)
def test_math_numeric_exact_match(prediction: str, reference: str, expected: float) -> None:
    assert score_math(prediction, reference) == expected


def test_invalid_fraction_is_not_an_answer() -> None:
    assert extract_numeric_answer("1/0") is None


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("B", "B"),
        ("Answer: (B)", "B"),
        ("A is tempting, but final answer: B", "B"),
        ("It could be A or B", None),
    ],
)
def test_extracts_arc_choice_conservatively(response: str, expected: str | None) -> None:
    assert extract_choice_label(response, ["A", "B", "C", "D"]) == expected


def test_arc_choice_accuracy() -> None:
    assert score_choice("Final answer is C", "C", ["A", "B", "C", "D"]) == 1.0


def test_hotpot_normalization_matches_official_style() -> None:
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"


def test_hotpot_exact_match_and_f1() -> None:
    exact = score_knowledge("The Eiffel Tower", ["Eiffel Tower"])
    partial = score_knowledge("Eiffel Tower Paris", ["Eiffel Tower"])

    assert exact.exact_match == 1.0
    assert exact.f1 == 1.0
    assert partial.exact_match == 0.0
    assert partial.f1 == pytest.approx(0.8)


def test_hotpot_special_answers_do_not_get_partial_credit() -> None:
    score = score_knowledge("yes indeed", ["yes"])

    assert score.exact_match == 0.0
    assert score.f1 == 0.0
