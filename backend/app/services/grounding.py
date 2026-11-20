"""Heuristic checks that an extracted value is actually supported by the cited passage."""
from __future__ import annotations

from app.utils.text import extract_numbers, normalize_for_match, tokenize

NOT_FOUND_MARKERS = frozenset(
    {
        "", "null", "none", "n/a", "na", "nil", "-", "--", "not specified", "not found", "unknown",
        "not mentioned", "not provided", "not stated", "unable to determine", "no information",
    }
)


def is_blank_value(value: object) -> bool:
    if value is None:
        return True
    return str(value).strip().strip(".").lower() in NOT_FOUND_MARKERS


def is_grounded(value: str, evidence: str | None, passage: str, *, strict: bool = False) -> bool:
    """Is `value` supported by `passage`?

    * If the value contains numbers, every number must appear in the passage.
      (This is the check that catches invented prices, dates and durations.)
    * Otherwise a verbatim evidence quote found in the passage is accepted, or - unless
      `strict` - at least 60% of the value's content words appear in the passage.
    This is a heuristic safety net, not proof; the UI still shows the source passage.
    """
    numbers = extract_numbers(value)
    if numbers:
        passage_numbers = set(extract_numbers(passage))
        return all(n in passage_numbers for n in numbers)
    if evidence and len(evidence.strip()) >= 4 and normalize_for_match(evidence) in normalize_for_match(passage):
        return True
    if strict:
        return False
    value_tokens = set(tokenize(value))
    if not value_tokens:
        return False
    return len(value_tokens & set(tokenize(passage))) / len(value_tokens) >= 0.6
