"""Health, dashboard and settings routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.api.deps import get_ctx
from app.core.config import RUNTIME_EDITABLE
from app.core.context import AppContext
from app.core.errors import AppError, InvalidRequestError
from app.schemas.api import HealthOut, SettingsUpdate
from app.services import analysis_service

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthOut)
def health(ctx: AppContext = Depends(get_ctx)):
    """Liveness only - does NOT call the model (use /api/settings/test-llm for that)."""
    return HealthOut(version=__version__, details={"database": "ok"})


@router.get("/dashboard")
def dashboard(ctx: AppContext = Depends(get_ctx)):
    docs = ctx.documents.list()
    return {
        "documents": {
            "total": len(docs),
            "by_status": ctx.documents.counts_by_status(),
            "total_pages": sum(d.page_count for d in docs),
            "recent": [
                {"id": d.id, "filename": d.filename, "status": d.status, "created_at": d.created_at}
                for d in sorted(docs, key=lambda d: d.created_at, reverse=True)[:5]
            ],
        },
        "collections": len(ctx.collections.list()),
        "analyses": {
            "total": sum(ctx.analyses.counts_by_operation().values()),
            "by_operation": ctx.analyses.counts_by_operation(),
            "recent": analysis_service.list_analyses(ctx, limit=6),
        },
        "model": {
            "llm_provider": ctx.settings.llm_provider,
            "llm_model": ctx.settings.llm_model,
            "embedding_provider": ctx.settings.embedding_provider,
            "embedding_model": ctx.settings.embedding_model,
        },
    }


@router.get("/settings")
def get_settings(ctx: AppContext = Depends(get_ctx)):
    return {"settings": ctx.settings.public_dict(), "editable": list(RUNTIME_EDITABLE)}


@router.put("/settings")
def update_settings(body: SettingsUpdate, ctx: AppContext = Depends(get_ctx)):
    values = body.model_dump(exclude_none=True)
    if not values:
        raise InvalidRequestError("No settings were provided.")
    ctx.update_runtime_settings(values)
    return {"settings": ctx.settings.public_dict(), "editable": list(RUNTIME_EDITABLE)}


@router.post("/settings/reset")
def reset_settings(ctx: AppContext = Depends(get_ctx)):
    """Drop UI overrides and go back to the values from .env."""
    ctx.reset_runtime_settings()
    return {"settings": ctx.settings.public_dict(), "editable": list(RUNTIME_EDITABLE)}


@router.post("/settings/test-llm")
def test_llm(ctx: AppContext = Depends(get_ctx)):
    """Send a tiny prompt to the configured model and report whether it answered."""
    llm = ctx.get_llm()
    info = llm.describe()
    try:
        reply = llm.generate("You are a connectivity check.", "Reply with the single word: OK")
        return {"ok": True, "model": ctx.settings.llm_model, "reply": reply.strip()[:80], "info": info}
    except AppError as exc:
        return {"ok": False, "model": ctx.settings.llm_model, "error": exc.message, "code": exc.code, "info": info}
