"""Analysis, history, export and report routes."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query, Response

from app.api.deps import get_ctx
from app.core.context import AppContext
from app.core.errors import InvalidRequestError
from app.schemas.analysis import AnalysisRecordOut, AnalysisRequest
from app.schemas.api import ReportRequest
from app.services import analysis_service, export_service, report_service
from app.services.field_presets import list_presets
from app.services.summary_service import SUMMARY_TYPES

router = APIRouter(prefix="/api", tags=["analysis"])


@router.get("/analysis/options")
def analysis_options():
    """Presets and summary types the UI offers."""
    return {
        "presets": list_presets(),
        "summary_types": [{"value": k, "label": v[0]} for k, v in SUMMARY_TYPES.items()],
    }


@router.post("/analyses/run", response_model=AnalysisRecordOut)
def run_analysis(body: AnalysisRequest, ctx: AppContext = Depends(get_ctx)):
    """Route the request through the workflow (Q&A / compare / extract / summarize) and save the result."""
    return analysis_service.run_analysis(ctx, body)


@router.get("/analyses")
def list_analyses(
    limit: int = Query(default=100, ge=1, le=500),
    operation: Literal["qa", "compare", "extract", "summarize"] | None = None,
    ctx: AppContext = Depends(get_ctx),
):
    return analysis_service.list_analyses(ctx, limit=limit, operation=operation)


@router.get("/analyses/{analysis_id}", response_model=AnalysisRecordOut)
def get_analysis(analysis_id: int, ctx: AppContext = Depends(get_ctx)):
    return analysis_service.get_analysis(ctx, analysis_id)


@router.delete("/analyses/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: int, ctx: AppContext = Depends(get_ctx)):
    analysis_service.delete_analysis(ctx, analysis_id)


@router.get("/analyses/{analysis_id}/export")
def export_analysis(
    analysis_id: int,
    format: Literal["json", "csv"] = "json",
    ctx: AppContext = Depends(get_ctx),
):
    record = analysis_service.get_analysis(ctx, analysis_id)
    content, media_type, ext = export_service.export_analysis(record.model_dump(), format)
    return Response(
        content=content.encode("utf-8-sig") if ext == "csv" else content.encode("utf-8"),
        media_type=f"{media_type}; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="analysis_{analysis_id}.{ext}"'},
    )


@router.post("/reports")
def create_report(body: ReportRequest, ctx: AppContext = Depends(get_ctx)):
    """Return a PDF.

    - `analysis_ids`: build the report from those saved analyses.
    - only `document_ids`: run an executive summary, extraction (and comparison for 2+ documents) first.
    """
    if body.analysis_ids:
        pdf, name = report_service.generate_report(ctx, analysis_ids=body.analysis_ids, title=body.title)
        ids: list[int] = body.analysis_ids
    elif body.document_ids:
        pdf, name, ids = report_service.generate_full_report(ctx, document_ids=body.document_ids, title=body.title)
    else:
        raise InvalidRequestError("Provide analysis_ids or document_ids to build a report.")
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "X-Analysis-Ids": ",".join(map(str, ids)),
            "Access-Control-Expose-Headers": "Content-Disposition, X-Analysis-Ids",
        },
    )
