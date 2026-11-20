"""Schemas shared by several operations."""
from __future__ import annotations

from pydantic import BaseModel


class Source(BaseModel):
    """A traceable pointer to the passage that supports a statement.

    `page_number` always comes from stored document metadata - never from model output.
    """

    source_id: str  # "S1", "S2", ... unique within one result
    document_id: int
    document_name: str
    page_number: int | None = None  # None for formats without pages (TXT)
    chunk_id: int
    snippet: str = ""

    @property
    def label(self) -> str:
        return f"{self.document_name}, p. {self.page_number}" if self.page_number else self.document_name
