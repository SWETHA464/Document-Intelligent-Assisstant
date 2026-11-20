"""Document ingestion, dedupe, failure handling, collections."""
from __future__ import annotations

import pytest

from app.core.errors import DocumentNotReadyError, InvalidRequestError, NotFoundError
from app.services import document_service as ds
from tests.conftest import VENDOR_A_PAGES


def test_ingest_pdf_ready_with_chunks_and_pages(ctx, pdf_bytes):
    out = ds.ingest_upload(ctx, "proposal.pdf", pdf_bytes(VENDOR_A_PAGES))
    assert out.outcome == "ready"
    doc = out.document
    assert doc.status == "ready" and doc.page_count == 3 and doc.chunk_count >= 3
    assert doc.file_type == "pdf" and doc.size_bytes > 0
    chunks = ctx.content.get_chunks([doc.id])
    assert {c.page_number for c in chunks} == {1, 2, 3}
    assert chunks[0].document_name == "proposal.pdf"
    assert len(ctx.content.get_pages(doc.id)) == 3


def test_ingest_txt(ctx):
    out = ds.ingest_upload(ctx, "notes.txt", b"Payment terms: Net 30.\nDelivery in 6 weeks.")
    assert out.outcome == "ready" and out.document.file_type == "txt"
    assert ctx.content.get_chunks([out.document.id])[0].page_number is None


def test_duplicate_upload_detected(ctx, pdf_bytes):
    data = pdf_bytes(VENDOR_A_PAGES)
    first = ds.ingest_upload(ctx, "a.pdf", data)
    second = ds.ingest_upload(ctx, "copy-of-a.pdf", data)
    assert second.outcome == "duplicate"
    assert second.document.id == first.document.id
    assert len(ctx.documents.list()) == 1


def test_unsupported_format_rejected_and_not_stored(ctx):
    out = ds.ingest_upload(ctx, "malware.exe", b"MZ....")
    assert out.outcome == "rejected" and "Unsupported" in out.error
    assert ctx.documents.list() == []


def test_empty_file_rejected(ctx):
    out = ds.ingest_upload(ctx, "empty.pdf", b"")
    assert out.outcome == "rejected" and "empty" in out.error.lower()


def test_oversized_file_rejected(ctx):
    out = ds.ingest_upload(ctx, "big.txt", b"a" * (ctx.settings.max_upload_bytes + 1))
    assert out.outcome == "rejected" and "limit" in out.error


def test_corrupted_pdf_is_stored_as_failed(ctx):
    out = ds.ingest_upload(ctx, "broken.pdf", b"%PDF-1.4 garbage garbage \x00\x01")
    assert out.outcome == "failed"
    assert out.document.status == "failed" and out.document.error
    assert ctx.content.get_chunks([out.document.id]) == []


def test_pdf_without_text_marked_failed_with_ocr_hint(ctx, pdf_bytes):
    out = ds.ingest_upload(ctx, "scan.pdf", pdf_bytes(["", ""]))
    assert out.outcome == "failed"
    assert "OCR" in out.document.error


def test_whitespace_only_txt_marked_failed(ctx):
    out = ds.ingest_upload(ctx, "blank.txt", b"   \n  \n")
    assert out.outcome == "failed" and "No extractable text" in out.error


def test_filename_is_sanitised(ctx):
    out = ds.ingest_upload(ctx, "../../etc/passwd.txt", b"hello there")
    assert out.outcome == "ready"
    assert "/" not in out.document.filename and ".." not in out.document.stored_name
    assert (ctx.settings.upload_dir / out.document.stored_name).exists()


def test_delete_removes_file_chunks_and_cache(ctx, pdf_bytes):
    doc = ds.ingest_upload(ctx, "a.pdf", pdf_bytes(VENDOR_A_PAGES)).document
    stored = ctx.settings.upload_dir / doc.stored_name
    cache = ctx.settings.embedding_cache_dir / f"doc{doc.id}_deadbeef.npy"
    cache.write_bytes(b"x")
    assert stored.exists()
    ds.delete_document(ctx, doc.id)
    assert not stored.exists() and not cache.exists()
    assert ctx.documents.get(doc.id) is None
    assert ctx.content.get_chunks([doc.id]) == []
    with pytest.raises(NotFoundError):
        ds.delete_document(ctx, doc.id)


def test_reupload_after_delete_allowed(ctx, pdf_bytes):
    data = pdf_bytes(VENDOR_A_PAGES)
    doc = ds.ingest_upload(ctx, "a.pdf", data).document
    ds.delete_document(ctx, doc.id)
    assert ds.ingest_upload(ctx, "a.pdf", data).outcome == "ready"


def test_collections_and_moving_documents(ctx, pdf_bytes):
    col = ds.create_collection(ctx, "  Vendor   proposals ")
    assert col["name"] == "Vendor proposals"
    with pytest.raises(InvalidRequestError):
        ds.create_collection(ctx, "vendor proposals".title())  # duplicate name
    with pytest.raises(InvalidRequestError):
        ds.create_collection(ctx, "   ")
    doc = ds.ingest_upload(ctx, "a.pdf", pdf_bytes(VENDOR_A_PAGES), collection_id=col["id"]).document
    assert doc.collection_id == col["id"]
    assert ctx.collections.get(col["id"])["document_count"] == 1
    assert [d.id for d in ctx.documents.list(collection_id=col["id"])] == [doc.id]
    moved = ds.move_document(ctx, doc.id, None)
    assert moved.collection_id is None
    with pytest.raises(NotFoundError):
        ds.move_document(ctx, doc.id, 999)


def test_upload_to_missing_collection_rejected(ctx):
    out = ds.ingest_upload(ctx, "a.txt", b"hello world", collection_id=42)
    assert out.outcome == "rejected"


def test_deleting_collection_keeps_documents(ctx):
    col = ds.create_collection(ctx, "Temp")
    doc = ds.ingest_upload(ctx, "a.txt", b"hello world", collection_id=col["id"]).document
    assert ctx.collections.delete(col["id"])
    assert ctx.documents.get(doc.id).collection_id is None


def test_require_ready_documents_validation(ctx, pdf_bytes):
    good = ds.ingest_upload(ctx, "a.pdf", pdf_bytes(VENDOR_A_PAGES)).document
    bad = ds.ingest_upload(ctx, "broken.pdf", b"%PDF-1.4 junk").document
    assert [d.id for d in ds.require_ready_documents(ctx, [good.id, good.id])] == [good.id]
    with pytest.raises(NotFoundError):
        ds.require_ready_documents(ctx, [good.id, 9999])
    with pytest.raises(DocumentNotReadyError):
        ds.require_ready_documents(ctx, [good.id, bad.id])
    with pytest.raises(InvalidRequestError):
        ds.require_ready_documents(ctx, [good.id], minimum=2)
    with pytest.raises(InvalidRequestError):
        ds.require_ready_documents(ctx, [])
