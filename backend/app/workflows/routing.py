"""Deterministic request routing (no LLM involved, so it is fast, free and testable)."""
from __future__ import annotations

from app.schemas.analysis import AnalysisRequest

OPERATIONS = ("qa", "compare", "extract", "summarize")

_COMPARE = ("compare", "comparison", "difference", "differ", "versus", " vs ", "contrast", "side by side", "cheaper")
_EXTRACT = ("extract", "pull out", "list the fields", "all fields")
_SUMMARY = ("summar", "overview", "tl;dr", "key points", "main points", "executive", "risks", "concerns")


def route_operation(request: AnalysisRequest) -> str:
    """Decide which operation handles the request.

    An explicit operation always wins. For "auto" we look at what was supplied and at keywords in
    the question. Anything not recognised is treated as a question about the documents.
    """
    if request.operation != "auto":
        return request.operation
    text = f" {(request.question or '').lower()} "
    multiple = len(set(request.document_ids)) >= 2
    if not request.question:
        if request.fields or request.preset:
            return "extract"
        return "summarize"
    if multiple and any(w in text for w in _COMPARE):
        return "compare"
    if any(w in text for w in _EXTRACT):
        return "extract"
    if any(w in text for w in _SUMMARY):
        return "summarize"
    return "qa"


def infer_summary_type(text: str | None) -> str:
    lowered = (text or "").lower()
    if "executive" in lowered:
        return "executive"
    if "risk" in lowered or "concern" in lowered:
        return "risks"
    if any(w in lowered for w in ("deadline", "obligation", "dates", "values")):
        return "dates_values_obligations"
    if any(w in lowered for w in ("key point", "finding", "highlight", "main point")):
        return "key_points"
    return "general"
