"""Request/response models that belong to the HTTP layer (documents, collections, settings, reports)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.models.domain import DocumentRecord


class DocumentOut(BaseModel):
    id: int
    collection_id: int | None
    filename: str
    file_type: str
    size_bytes: int
    status: Literal["processing", "ready", "failed"] | str
    error: str | None = None
    page_count: int = 0
    char_count: int = 0
    chunk_count: int = 0
    created_at: str

    @classmethod
    def from_record(cls, d: DocumentRecord) -> "DocumentOut":
        return cls(
            id=d.id, collection_id=d.collection_id, filename=d.filename, file_type=d.file_type,
            size_bytes=d.size_bytes, status=d.status, error=d.error, page_count=d.page_count,
            char_count=d.char_count, chunk_count=d.chunk_count, created_at=d.created_at,
        )


class UploadItemOut(BaseModel):
    filename: str
    outcome: Literal["ready", "failed", "rejected", "duplicate"]
    document: DocumentOut | None = None
    error: str | None = None
    warnings: list[str] = []


class UploadResponse(BaseModel):
    items: list[UploadItemOut]
    counts: dict[str, int]


class CollectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=300)


class CollectionOut(BaseModel):
    id: int
    name: str
    description: str = ""
    created_at: str
    document_count: int = 0


class MoveDocumentIn(BaseModel):
    collection_id: int | None = None


class SettingsUpdate(BaseModel):
    """Only runtime-editable values. Secrets (API keys) are never accepted or returned here."""

    llm_provider: Literal["ollama", "openai"] | None = None
    llm_model: str | None = Field(default=None, min_length=1, max_length=100)
    llm_base_url: str | None = Field(default=None, min_length=1, max_length=300)
    llm_temperature: float | None = Field(default=None, ge=0.0, le=1.5)
    embedding_provider: Literal["ollama", "sentence_transformers", "hashing"] | None = None
    embedding_model: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("llm_model", "llm_base_url", "embedding_model")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        return v.strip() if v else v


class ReportRequest(BaseModel):
    """Build a report from saved analyses, or (document_ids only) run a one-click full report."""

    analysis_ids: list[int] = Field(default_factory=list, max_length=50)
    document_ids: list[int] = Field(default_factory=list, max_length=20)
    title: str | None = Field(default=None, max_length=200)


class HealthOut(BaseModel):
    status: str = "ok"
    version: str
    details: dict[str, Any] = {}
