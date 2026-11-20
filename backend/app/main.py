"""FastAPI application factory.  Run with:  uvicorn app.main:app --reload"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import analysis, documents, system
from app.core.config import Settings
from app.core.context import AppContext
from app.core.errors import AppError

logger = logging.getLogger(__name__)


def _error(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details or {}}})


def create_app(ctx: AppContext | None = None) -> FastAPI:
    """Build the app. Tests pass their own `ctx` (temp database, fake LLM)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ctx = ctx or AppContext(Settings.from_env())
    app = FastAPI(title="Document Intelligence Assistant", version=__version__)
    app.state.ctx = ctx

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(ctx.settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition", "X-Analysis-Ids"],
    )

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return _error(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        issues = [
            {"field": ".".join(str(p) for p in e.get("loc", []) if p != "body"), "message": e.get("msg", "")}
            for e in exc.errors()
        ]
        first = issues[0] if issues else {"field": "", "message": "Invalid request"}
        msg = f"Invalid request: {first['field']} - {first['message']}" if first["field"] else "Invalid request."
        return _error(422, "validation_error", msg, {"issues": issues})

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        logger.exception("Unhandled error")
        return _error(500, "internal_error", "An unexpected server error occurred. Check the backend log.")

    for module in (system, documents, analysis):
        app.include_router(module.router)
    return app


app = create_app()
