"""Runs an analysis request through the workflow and persists the result."""
from __future__ import annotations

from typing import Any

from app.core.context import AppContext
from app.core.errors import NotFoundError
from app.schemas.analysis import AnalysisRecordOut, AnalysisRequest
from app.services.field_presets import PRESETS
from app.services.summary_service import SUMMARY_TYPES
from app.utils.text import truncate
from app.workflows import build_workflow


def _title(req: AnalysisRequest, operation: str, result: dict[str, Any], names: list[str]) -> str:
    if req.title and req.title.strip():
        return req.title.strip()[:200]
    n = len(names)
    docs = names[0] if n == 1 else f"{n} documents"
    if operation == "qa":
        return f"Q&A: {truncate(req.question or '', 70)}"
    if operation == "compare":
        if result.get("mode") == "versions":
            return f"Version comparison: {names[0]} vs {names[1]}"
        return f"Comparison of {docs}"
    if operation == "extract":
        label = PRESETS[req.preset]["label"] if req.preset in PRESETS else "custom fields"
        preset = result.get("preset")
        if preset in PRESETS:
            label = PRESETS[preset]["label"]
        return f"Extraction ({label}): {docs}"
    if operation == "summarize":
        return f"{SUMMARY_TYPES[result['summary_type']][0]}: {docs}"
    return f"Analysis of {docs}"


def run_analysis(ctx: AppContext, req: AnalysisRequest) -> AnalysisRecordOut:
    """Execute the request. Raises `AppError` subclasses for expected failures."""
    workflow = build_workflow(ctx)
    state = workflow.invoke({"request": req, "steps": []})
    if state.get("error"):
        raise state["error"]

    operation: str = state["operation"]
    result: dict[str, Any] = state["result"]
    documents = state["documents"]
    names = {d.id: d.filename for d in documents}
    params = req.model_dump(mode="json", exclude={"save", "title"}, exclude_none=True)
    params["resolved_operation"] = operation
    params["document_names"] = {str(i): n for i, n in names.items()}
    title = _title(req, operation, result, list(names.values()))
    document_ids = list(names.keys())

    record = AnalysisRecordOut(
        operation=operation,  # type: ignore[arg-type]
        title=title,
        document_ids=document_ids,
        document_names=names,
        params=params,
        result=result,
    )
    if req.save:
        saved = ctx.analyses.create(
            operation=operation, title=title, document_ids=document_ids, params=params, result=result
        )
        record.id, record.created_at = saved["id"], saved["created_at"]
    return record


def to_record(row: dict[str, Any]) -> AnalysisRecordOut:
    names_raw = (row.get("params") or {}).get("document_names", {})
    return AnalysisRecordOut(
        id=row["id"],
        operation=row["operation"],
        title=row["title"],
        document_ids=row["document_ids"],
        document_names={int(k): v for k, v in names_raw.items()},
        params=row["params"],
        result=row["result"],
        created_at=row["created_at"],
    )


def get_analysis(ctx: AppContext, analysis_id: int) -> AnalysisRecordOut:
    row = ctx.analyses.get(analysis_id)
    if row is None:
        raise NotFoundError(f"Analysis {analysis_id} not found.")
    return to_record(row)


def list_analyses(ctx: AppContext, *, limit: int = 100, operation: str | None = None) -> list[dict[str, Any]]:
    """Lightweight rows for the history list (no result payloads)."""
    out = []
    for row in ctx.analyses.list(limit=limit, operation=operation):
        rec = to_record(row)
        out.append(
            {
                "id": rec.id,
                "operation": rec.operation,
                "title": rec.title,
                "document_ids": rec.document_ids,
                "document_names": rec.document_names,
                "created_at": rec.created_at,
            }
        )
    return out


def delete_analysis(ctx: AppContext, analysis_id: int) -> None:
    if not ctx.analyses.delete(analysis_id):
        raise NotFoundError(f"Analysis {analysis_id} not found.")
