"""Document ingestion and management: validate -> store -> extract -> chunk -> persist."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.core.context import AppContext
from app.core.errors import (
    AppError,
    DocumentNotReadyError,
    DuplicateDocumentError,
    EmptyFileError,
    ExtractionError,
    FileTooLargeError,
    InvalidRequestError,
    NotFoundError,
    UnsupportedFileError,
)
from app.models.domain import DocumentRecord, PageRecord
from app.rag.chunking import chunk_pages
from app.services.text_extraction import get_extractor
from app.utils.files import safe_filename, sha256_bytes

logger = logging.getLogger(__name__)


@dataclass
class IngestOutcome:
    """Result of one uploaded file (a batch upload returns one per file)."""

    filename: str
    outcome: str  # "ready" | "failed" | "rejected" | "duplicate"
    document: DocumentRecord | None = None
    error: str | None = None
    warnings: list[str] | None = None


def ingest_upload(ctx: AppContext, filename: str, data: bytes, collection_id: int | None = None) -> IngestOutcome:
    """Process one uploaded file. Never raises for bad input - returns an outcome instead."""
    name = safe_filename(filename or "document")
    try:
        _validate(ctx, name, data, collection_id)
        sha = sha256_bytes(data)
        existing = ctx.documents.get_by_sha256(sha)
        if existing:
            raise DuplicateDocumentError(
                f"'{name}' was already uploaded as '{existing.filename}'.",
                details={"existing_document_id": existing.id},
            )
    except DuplicateDocumentError as exc:
        existing_id = exc.details.get("existing_document_id")
        return IngestOutcome(name, "duplicate", ctx.documents.get(existing_id) if existing_id else None, exc.message)
    except AppError as exc:
        return IngestOutcome(name, "rejected", None, exc.message)

    extractor = get_extractor(name)
    stored_name = f"{sha[:16]}_{name}"
    path = ctx.settings.upload_dir / stored_name
    path.write_bytes(data)
    doc = ctx.documents.create(
        filename=name,
        stored_name=stored_name,
        file_type=extractor.file_type,
        size_bytes=len(data),
        sha256=sha,
        collection_id=collection_id,
    )
    return _process(ctx, doc, path)


def _validate(ctx: AppContext, name: str, data: bytes, collection_id: int | None) -> None:
    get_extractor(name)  # raises UnsupportedFileError
    if not data:
        raise EmptyFileError(f"'{name}' is empty (0 bytes).")
    if len(data) > ctx.settings.max_upload_bytes:
        raise FileTooLargeError(f"'{name}' is larger than the {ctx.settings.max_upload_mb} MB upload limit.")
    if collection_id is not None and ctx.collections.get(collection_id) is None:
        raise NotFoundError(f"Collection {collection_id} does not exist.")


def _process(ctx: AppContext, doc: DocumentRecord, path: Path) -> IngestOutcome:
    """Extract text and chunk it. Failures are recorded on the document, not raised."""
    try:
        result = get_extractor(doc.filename).extract(path)
        if not result.pages:
            raise ExtractionError(
                "No extractable text was found. The file may be empty or a scanned/image-only document "
                "(OCR is not enabled in this version)."
            )
        drafts = chunk_pages(result.pages, ctx.settings.chunk_size, ctx.settings.chunk_overlap)
        ctx.content.save_pages(doc.id, [PageRecord(doc.id, p.page_number, p.text) for p in result.pages])
        count = ctx.content.save_chunks(doc.id, [(d.chunk_index, d.page_number, d.text) for d in drafts])
        ctx.documents.mark_ready(
            doc.id, page_count=result.page_count, char_count=result.char_count, chunk_count=count
        )
        outcome, error = "ready", None
        warnings = result.warnings
    except AppError as exc:
        ctx.documents.mark_failed(doc.id, exc.message)
        outcome, error, warnings = "failed", exc.message, None
    except Exception as exc:  # defensive: never let one bad file crash a batch upload
        logger.exception("Unexpected error processing %s", doc.filename)
        message = f"Unexpected error while processing the file ({type(exc).__name__})."
        ctx.documents.mark_failed(doc.id, message)
        outcome, error, warnings = "failed", message, None
    return IngestOutcome(doc.filename, outcome, ctx.documents.get(doc.id), error, warnings)


def delete_document(ctx: AppContext, document_id: int) -> None:
    doc = ctx.documents.get(document_id)
    if doc is None:
        raise NotFoundError(f"Document {document_id} not found.")
    ctx.documents.delete(document_id)  # pages/chunks cascade
    (ctx.settings.upload_dir / doc.stored_name).unlink(missing_ok=True)
    for cached in ctx.settings.embedding_cache_dir.glob(f"doc{document_id}_*.npy"):
        cached.unlink(missing_ok=True)


def move_document(ctx: AppContext, document_id: int, collection_id: int | None) -> DocumentRecord:
    if ctx.documents.get(document_id) is None:
        raise NotFoundError(f"Document {document_id} not found.")
    if collection_id is not None and ctx.collections.get(collection_id) is None:
        raise NotFoundError(f"Collection {collection_id} does not exist.")
    ctx.documents.set_collection(document_id, collection_id)
    return ctx.documents.get(document_id)  # type: ignore[return-value]


def require_ready_documents(ctx: AppContext, document_ids: list[int], *, minimum: int = 1) -> list[DocumentRecord]:
    """Validate a selection: ids exist, are unique, are processed. Returns records in request order."""
    unique = list(dict.fromkeys(document_ids))
    if len(unique) < minimum:
        raise InvalidRequestError(
            f"Select at least {minimum} document{'s' if minimum > 1 else ''} for this operation."
        )
    docs = ctx.documents.get_many(unique)
    found = {d.id for d in docs}
    missing = [i for i in unique if i not in found]
    if missing:
        raise NotFoundError("Document(s) not found: " + ", ".join(map(str, missing)))
    not_ready = [d for d in docs if d.status != "ready"]
    if not_ready:
        names = ", ".join(f"'{d.filename}' ({d.status})" for d in not_ready)
        raise DocumentNotReadyError(f"These documents cannot be analysed yet: {names}.")
    return docs


def create_collection(ctx: AppContext, name: str, description: str = "") -> dict:
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        raise InvalidRequestError("Collection name cannot be empty.")
    if len(name) > 80:
        raise InvalidRequestError("Collection name is too long (max 80 characters).")
    if ctx.collections.get_by_name(name):
        raise InvalidRequestError(f"A collection named '{name}' already exists.")
    return ctx.collections.create(name, description.strip()[:300])
