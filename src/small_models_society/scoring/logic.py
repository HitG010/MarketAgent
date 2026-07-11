"""ARC multiple-choice label scoring."""

from __future__ import annotations

import re
from collections.abc import Sequence


def extract_choice_label(response: str, labels: Sequence[str]) -> str | None:
    """Extract a choice only when it maps unambiguously to an allowed label."""

    label_lookup = {label.casefold(): label for label in labels}
    stripped = response.strip().strip("()[]{}.: ")
    if stripped.casefold() in label_lookup:
        return label_lookup[stripped.casefold()]

    alternatives = "|".join(sorted((re.escape(label) for label in labels), key=len, reverse=True))
    marked = re.findall(
        rf"(?i)(?:final\s+answer|answer|choice)\s*(?:is|:|=)?\s*[\(\[]?({alternatives})[\)\]]?(?!\w)",
        response,
    )
    if marked:
        unique = {match.casefold() for match in marked}
        return label_lookup[marked[-1].casefold()] if len(unique) == 1 else None

    standalone = re.findall(rf"(?i)(?<!\w)({alternatives})(?!\w)", response)
    unique = {match.casefold() for match in standalone}
    return label_lookup[standalone[-1].casefold()] if len(unique) == 1 else None


def score_choice(response: str, reference_label: str, labels: Sequence[str]) -> float:
    predicted_label = extract_choice_label(response, labels)
    return float(
        predicted_label is not None and predicted_label.casefold() == reference_label.casefold()
    )
