"""Official-style HotpotQA answer exact match and token F1."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeScore:
    exact_match: float
    f1: float


def normalize_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _score_single(prediction: str, reference: str) -> KnowledgeScore:
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    exact_match = float(normalized_prediction == normalized_reference)

    special_answers = {"yes", "no", "noanswer"}
    if (
        normalized_prediction in special_answers or normalized_reference in special_answers
    ) and normalized_prediction != normalized_reference:
        return KnowledgeScore(exact_match=exact_match, f1=0.0)

    prediction_tokens = normalized_prediction.split()
    reference_tokens = normalized_reference.split()
    shared = Counter(prediction_tokens) & Counter(reference_tokens)
    matching_tokens = sum(shared.values())
    if matching_tokens == 0:
        return KnowledgeScore(exact_match=exact_match, f1=0.0)
    precision = matching_tokens / len(prediction_tokens)
    recall = matching_tokens / len(reference_tokens)
    return KnowledgeScore(
        exact_match=exact_match,
        f1=2 * precision * recall / (precision + recall),
    )


def score_knowledge(prediction: str, references: list[str]) -> KnowledgeScore:
    """Use the best official-style score across accepted reference variants."""

    if not references:
        raise ValueError("at least one knowledge reference is required")
    scores = [_score_single(prediction, reference) for reference in references]
    return KnowledgeScore(
        exact_match=max(score.exact_match for score in scores),
        f1=max(score.f1 for score in scores),
    )
