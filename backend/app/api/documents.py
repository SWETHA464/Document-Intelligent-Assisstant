"""Document and collection routes. Thin: validation + delegation to `document_service`."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import get_ctx
from app.core.context import AppContext
from app.core.errors import InvalidRequestError, NotFoundError
from app.schemas.api import (
    CollectionIn, CollectionOut, DocumentOut, MoveDocumentIn, UploadItemOut, UploadResponse,
)
from app.services import document_service as svc

router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/collections", response_model=list[CollectionOut])
def list_collections(ctx: AppContext = Depends(get_ctx)):
    return ctx.collections.list()


@router.post("/collections", response_model=CollectionOut, status_code=201)
def create_collection(body: CollectionIn, ctx: AppContext = Depends(get_ctx)):
    return svc.create_collection(ctx, body.name, body.description)


@router.delete("/collections/{collection_id}", status_code=204)
def delete_collection(collection_id: int, ctx: AppContext = Depends(get_ctx)):
    """Removes the collection only; its documents are kept (they become uncategorised)."""
    if not ctx.collections.delete(collection_id):
        raise NotFoundError(f"Collection {collection_id} not found.")


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(collection_id: int | None = None, ctx: AppContext = Depends(get_ctx)):
    return [DocumentOut.from_record(d) for d in ctx.documents.list(collection_id)]


@router.post("/documents/upload", response_model=UploadResponse)
def upload_documents(
    files: list[UploadFile] = File(...),
    collection_id: int | None = Form(default=None),
    ctx: AppContext = Depends(get_ctx),
):
    """Upload one or more files. Each file succeeds or fails independently (HTTP 200 with per-file outcomes)."""
    if not files:
        raise InvalidRequestError("No files were uploaded.")
    limit = ctx.settings.max_upload_bytes
    items: list[UploadItemOut] = []
    for f in files:
        data = f.file.read(limit + 1)  # read one byte past the limit so oversize files are detected
        out = svc.ingest_upload(ctx, f.filename or "document", data, collection_id)
        items.append(
            UploadItemOut(
                filename=out.filename,
                outcome=out.outcome,  # type: ignore[arg-type]
                document=DocumentOut.from_record(out.document) if out.document else None,
                error=out.error,
                warnings=out.warnings or [],
            )
        )
    counts: dict[str, int] = {}
    for item in items:
        counts[item.outcome] = counts.get(item.outcome, 0) + 1
    return UploadResponse(items=items, counts=counts)


@router.get("/documents/{document_id}/preview")
def document_preview(document_id: int, ctx: AppContext = Depends(get_ctx)):
    """First ~1500 characters of extracted text, so users can check extraction quality."""
    doc = ctx.documents.get(document_id)
    if doc is None:
        raise NotFoundError(f"Document {document_id} not found.")
    pages = ctx.content.get_pages(document_id)
    text = "\n\n".join(p.text for p in pages[:3])
    return {"document": DocumentOut.from_record(doc), "preview": text[:1500], "truncated": len(text) > 1500}


@router.patch("/documents/{document_id}", response_model=DocumentOut)
def move_document(document_id: int, body: MoveDocumentIn, ctx: AppContext = Depends(get_ctx)):
    return DocumentOut.from_record(svc.move_document(ctx, document_id, body.collection_id))


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: int, ctx: AppContext = Depends(get_ctx)):
    svc.delete_document(ctx, document_id)
