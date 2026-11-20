"""Plain-text (.txt) extraction with encoding detection."""
from __future__ import annotations

from pathlib import Path

from app.core.errors import ExtractionError
from app.services.text_extraction.base import ExtractedPage, ExtractionResult, TextExtractor
from app.services.text_extraction.cleaning import clean_text

_ENCODINGS = ("utf-8-sig", "utf-16", "cp1252", "latin-1")


class TextFileExtractor(TextExtractor):
    extensions = (".txt",)
    file_type = "txt"

    def extract(self, path: Path) -> ExtractionResult:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ExtractionError(f"The file could not be read: {exc}") from exc
        if b"\x00" in data[:4096] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
            raise ExtractionError("This looks like a binary file, not plain text.")
        text = None
        for encoding in _ENCODINGS:
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:  # pragma: no cover - latin-1 never fails
            raise ExtractionError("The text encoding could not be determined.")
        cleaned = clean_text(text)
        pages = [ExtractedPage(None, cleaned)] if cleaned else []
        return ExtractionResult(pages=pages, page_count=1 if pages else 0)
