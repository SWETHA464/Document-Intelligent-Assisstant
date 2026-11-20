"""Shared test fixtures: temp data dir, offline embedder, scripted fake LLM, PDF builder."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from app.core.config import Settings
from app.core.context import AppContext
from app.core.errors import LLMUnavailableError
from app.rag.embeddings import HashingEmbedder
from app.services.llm.base import LLMClient


class ScriptedLLM(LLMClient):
    """Fake LLM: returns queued replies (str), raises queued Exceptions, or calls a function.

    Records every call so tests can assert the model was (or was not) used.
    """

    def __init__(self, replies: list[Any] | Callable[[str, str], str] | None = None) -> None:
        self.replies = replies if replies is not None else []
        self.calls: list[dict] = []

    def generate(self, system: str, prompt: str, *, json_mode: bool = False) -> str:
        self.calls.append({"system": system, "prompt": prompt, "json_mode": json_mode})
        if callable(self.replies):
            return self.replies(system, prompt)
        if not self.replies:
            raise AssertionError("ScriptedLLM has no more replies queued")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, (dict, list)):
            return json.dumps(reply)
        return reply

    def describe(self) -> dict:
        return {"provider": "fake", "model": "fake"}


class DownLLM(LLMClient):
    def generate(self, system: str, prompt: str, *, json_mode: bool = False) -> str:
        raise LLMUnavailableError("Model offline (test)")


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Create a small text PDF with one string of text per page (lines split on \\n)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        embedding_provider="hashing",
        chunk_size=400,
        chunk_overlap=60,
        max_upload_mb=1,
    )


@pytest.fixture
def ctx(settings: Settings) -> AppContext:
    return AppContext(settings, llm=ScriptedLLM(), embedder=HashingEmbedder())


@pytest.fixture
def pdf_bytes(tmp_path: Path):
    def _make(pages: list[str]) -> bytes:
        p = make_pdf(tmp_path / "tmp_make.pdf", pages)
        return p.read_bytes()

    return _make


@pytest.fixture
def add_doc(ctx: AppContext, pdf_bytes):
    """Ingest a PDF built from `pages`; returns the DocumentRecord (must be ready)."""

    def _add(filename: str, pages: list[str], collection_id: int | None = None):
        outcome = _ingest(ctx, filename, pdf_bytes(pages), collection_id)
        assert outcome.outcome == "ready", outcome
        return outcome.document

    return _add


def _ingest(ctx, filename, data, collection_id=None):
    from app.services.document_service import ingest_upload

    return ingest_upload(ctx, filename, data, collection_id)


VENDOR_A_PAGES = [
    "Acme Solutions Proposal\nProposal Date: 12 March 2026\nVendor: Acme Solutions Pvt Ltd",
    "Pricing\nTotal project price: USD 120,000 fixed fee.\nPayment terms: Net 30 days from invoice.\n"
    "Delivery timeline: 16 weeks from contract signing.",
    "Contract duration: 24 months.\nContact: sales@acme.example, +91 44 5555 0100",
]
VENDOR_B_PAGES = [
    "Beta Systems Proposal\nProposal Date: 2 April 2026\nVendor: Beta Systems Ltd",
    "Commercials\nTotal project price: USD 98,500 including support.\nPayment terms: Net 45 days from invoice.\n"
    "Delivery timeline: 20 weeks from kickoff.",
    "Contract duration: 12 months with optional renewal.",
]
