"""Workflow state shared by all nodes."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from app.core.errors import AppError
from app.models.domain import DocumentRecord
from app.schemas.analysis import AnalysisRequest


class WorkflowState(TypedDict, total=False):
    request: AnalysisRequest
    operation: str  # resolved: qa | compare | extract | summarize
    documents: list[DocumentRecord]
    result: dict[str, Any]  # JSON-serialisable result of the chosen operation
    error: AppError | None
    steps: Annotated[list[str], operator.add]  # trace of executed nodes (useful for debugging/tests)
