"""Extractor interface. Add an OCR extractor later by implementing `TextExtractor`."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractedPage:
    page_number: int | None  # None for formats without pages (plain text)
    text: str


@dataclass
class ExtractionResult:
    pages: list[ExtractedPage]
    page_count: int
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


class TextExtractor(ABC):
    """Turns a file on disk into text pages. Raise `ExtractionError` on failure."""

    extensions: tuple[str, ...] = ()
    file_type: str = ""

    @abstractmethod
    def extract(self, path: Path) -> ExtractionResult:  # pragma: no cover - interface
        ...
