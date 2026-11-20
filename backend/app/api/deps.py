"""FastAPI dependencies."""
from __future__ import annotations

from fastapi import Request

from app.core.context import AppContext


def get_ctx(request: Request) -> AppContext:
    """The AppContext is created once in `create_app` and stored on `app.state`."""
    return request.app.state.ctx
