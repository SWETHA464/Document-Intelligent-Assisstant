"""Extractor registry: pick the right extractor for a file name."""
from __future__ import annotations

from pathlib import PurePath

from app.core.errors import UnsupportedFileError
from app.services.text_extraction.base import ExtractedPage, ExtractionResult, TextExtractor
from app.services.text_extraction.pdf_extractor import PdfExtractor
from app.services.text_extraction.text_extractor import TextFileExtractor

# To add OCR or DOCX later: implement TextExtractor and append it here.
_EXTRACTORS: list[TextExtractor] = [PdfExtractor(), TextFileExtractor()]

SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(ext for e in _EXTRACTORS for ext in e.extensions)


def get_extractor(filename: str) -> TextExtractor:
    suffix = PurePath(filename).suffix.lower()
    for extractor in _EXTRACTORS:
        if suffix in extractor.extensions:
            return extractor
    raise UnsupportedFileError(
        f"Unsupported file type '{suffix or 'unknown'}'. Supported types: "
        + ", ".join(SUPPORTED_EXTENSIONS)
    )


__all__ = [
    "ExtractedPage",
    "ExtractionResult",
    "TextExtractor",
    "SUPPORTED_EXTENSIONS",
    "get_extractor",
]
