"""PDF text extraction using pypdf (pure Python, no system dependencies)."""
from __future__ import annotations

import logging
from pathlib import Path

from app.core.errors import ExtractionError
from app.services.text_extraction.base import ExtractedPage, ExtractionResult, TextExtractor
from app.services.text_extraction.cleaning import clean_text

logger = logging.getLogger(__name__)


class PdfExtractor(TextExtractor):
    extensions = (".pdf",)
    file_type = "pdf"

    def extract(self, path: Path) -> ExtractionResult:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - pypdf is in requirements
            raise ExtractionError("PDF support requires the 'pypdf' package.") from exc

        try:
            with open(path, "rb") as fh:
                head = fh.read(1024)
            if b"%PDF" not in head:
                raise ExtractionError("This file is not a valid PDF (missing PDF header).")
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                # Many "encrypted" PDFs only use an owner password and open with an empty user password.
                if not reader.decrypt(""):
                    raise ExtractionError("The PDF is password-protected and cannot be read.")
            total = len(reader.pages)
        except ExtractionError:
            raise
        except Exception as exc:  # pypdf raises many different error types for broken files
            raise ExtractionError(f"The PDF appears to be corrupted or unreadable ({type(exc).__name__}).") from exc

        pages: list[ExtractedPage] = []
        warnings: list[str] = []
        blank_pages: list[int] = []
        for index in range(total):
            number = index + 1
            try:
                raw = reader.pages[index].extract_text() or ""
            except Exception as exc:
                logger.warning("Could not extract page %s of %s: %s", number, path.name, exc)
                warnings.append(f"Page {number} could not be read.")
                continue
            text = clean_text(raw)
            if text:
                pages.append(ExtractedPage(number, text))
            else:
                blank_pages.append(number)

        if blank_pages and pages:
            warnings.append(
                f"{len(blank_pages)} page(s) contained no extractable text (possibly scanned images): "
                + ", ".join(map(str, blank_pages[:10]))
                + ("..." if len(blank_pages) > 10 else "")
            )
        return ExtractionResult(pages=pages, page_count=total, warnings=warnings)
