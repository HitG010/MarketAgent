"""GSM8K-style normalized numeric exact match."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

_NUMBER = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?:/\d+)?"
)


def extract_numeric_answer(value: str) -> Fraction | None:
    """Extract the final numeric expression and convert it to an exact value."""

    matches = _NUMBER.findall(value)
    if not matches:
        return None
    token = matches[-1].replace(",", "")
    try:
        if "/" in token:
            return Fraction(token)
        return Fraction(Decimal(token))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def score_math(prediction: str, reference: str) -> float:
    predicted_value = extract_numeric_answer(prediction)
    reference_value = extract_numeric_answer(reference)
    return float(
        predicted_value is not None
        and reference_value is not None
        and predicted_value == reference_value
    )
