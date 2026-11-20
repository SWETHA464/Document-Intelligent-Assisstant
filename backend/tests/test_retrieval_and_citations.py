"""BM25, vector store, hybrid retrieval, evidence sufficiency, citation tracking."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import EmbeddingError
from app.rag.bm25 import BM25Index
from app.rag.citations import (
    block_by_id,
    build_blocks,
    format_context,
    normalize_source_id,
    resolve_citations,
)
from app.rag.embeddings import HashingEmbedder, create_embedder
from app.rag.retriever import open_session, retrieve
from app.rag.vector_store import FAISS_AVAILABLE, VectorStore
from app.utils.text import tokenize
from tests.conftest import VENDOR_A_PAGES, VENDOR_B_PAGES


# --------------------------------------------------------------- BM25 ----
def test_bm25_ranks_matching_doc_first():
    docs = [tokenize(t) for t in ["payment terms net thirty days", "delivery timeline sixteen weeks", "office address"]]
    scores = BM25Index(docs).scores(tokenize("payment terms"))
    assert int(np.argmax(scores)) == 0
    assert scores[2] == 0


def test_bm25_empty_and_unknown_terms():
    assert BM25Index([]).scores(["x"]).shape == (0,)
    scores = BM25Index([["a", "b"]]).scores(["zzz"])
    assert scores.tolist() == [0.0]


def test_tokenizer_handles_numbers_and_plurals():
    assert "120000" in tokenize("Price USD 120,000")
    assert tokenize("Payments") == tokenize("payment")


# --------------------------------------------------------- vector store ----
def test_vector_store_numpy_backend_ranks_by_cosine():
    emb = HashingEmbedder()
    texts = ["payment terms net 30", "delivery in sixteen weeks", "totally unrelated gardening notes"]
    store = VectorStore(emb.embed_documents(texts), use_faiss=False)
    assert store.backend == "numpy"
    best = store.search(emb.embed_query("what are the payment terms"), k=3)
    assert best[0][0] == 0
    assert len(best) == 3


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss-cpu not installed in this environment")
def test_faiss_matches_numpy_backend():
    emb = HashingEmbedder()
    texts = [f"document {i} about topic {i % 3} pricing" for i in range(20)]
    vecs = emb.embed_documents(texts)
    q = emb.embed_query("topic 1 pricing")
    a = VectorStore(vecs, use_faiss=True).search(q, 5)
    b = VectorStore(vecs, use_faiss=False).search(q, 5)
    assert [i for i, _ in a] == [i for i, _ in b]


def test_vector_store_validates_input():
    with pytest.raises(ValueError):
        VectorStore(np.zeros((0, 4), dtype=np.float32))
    store = VectorStore(np.eye(3, dtype=np.float32), use_faiss=False)
    with pytest.raises(ValueError):
        store.search(np.ones(5, dtype=np.float32), 1)


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder()
    a, b = emb.embed_query("payment terms"), emb.embed_query("payment terms")
    assert np.allclose(a, b) and abs(np.linalg.norm(a) - 1) < 1e-5


def test_create_embedder_unknown_provider(settings):
    from dataclasses import replace

    with pytest.raises(EmbeddingError):
        create_embedder(replace(settings, embedding_provider="nope"))


# ---------------------------------------------------- hybrid retrieval ----
def test_retrieval_finds_right_page_in_right_document(ctx, add_doc):
    a = add_doc("vendor_a.pdf", VENDOR_A_PAGES)
    b = add_doc("vendor_b.pdf", VENDOR_B_PAGES)
    res = retrieve(ctx, "What are the payment terms?", [a.id, b.id], top_k=4)
    assert res.mode == "hybrid" and res.sufficient
    top_by_doc = {}
    for rc in res.chunks:
        top_by_doc.setdefault(rc.chunk.document_id, rc)
    assert top_by_doc[a.id].chunk.page_number == 2 and "Net 30" in top_by_doc[a.id].chunk.text
    assert top_by_doc[b.id].chunk.page_number == 2 and "Net 45" in top_by_doc[b.id].chunk.text


def test_retrieval_restricted_to_selected_documents(ctx, add_doc):
    a = add_doc("vendor_a.pdf", VENDOR_A_PAGES)
    add_doc("vendor_b.pdf", VENDOR_B_PAGES)
    res = retrieve(ctx, "payment terms", [a.id])
    assert {rc.chunk.document_id for rc in res.chunks} == {a.id}


def test_session_search_can_filter_a_single_document(ctx, add_doc):
    a = add_doc("vendor_a.pdf", VENDOR_A_PAGES)
    b = add_doc("vendor_b.pdf", VENDOR_B_PAGES)
    session = open_session(ctx, [a.id, b.id])
    res = session.search("project price", only_document_ids=[b.id])
    assert res.chunks and {rc.chunk.document_id for rc in res.chunks} == {b.id}


def test_per_doc_min_guarantees_every_document_is_represented(ctx, add_doc):
    a = add_doc("vendor_a.pdf", VENDOR_A_PAGES)
    b = add_doc("vendor_b.pdf", ["Beta only talks about renewal options and warranty.", "More renewal text."])
    res = retrieve(ctx, "Net 30 payment invoice", [a.id, b.id], top_k=1, per_doc_min=1)
    assert a.id in {rc.chunk.document_id for rc in res.chunks}


def test_unrelated_question_is_not_sufficient(ctx, add_doc):
    a = add_doc("vendor_a.pdf", VENDOR_A_PAGES)
    res = retrieve(ctx, "What is the capital of France?", [a.id])
    assert res.sufficient is False


def test_keyword_only_fallback_when_embedder_fails(ctx, add_doc):
    class BrokenEmbedder(HashingEmbedder):
        name = "broken"

        def embed_documents(self, texts):
            raise EmbeddingError("Ollama is down")

        def embed_query(self, text):
            raise EmbeddingError("Ollama is down")

    a = add_doc("vendor_a.pdf", VENDOR_A_PAGES)
    ctx._embedder_override = BrokenEmbedder()
    res = retrieve(ctx, "payment terms Net 30", [a.id])
    assert res.mode == "keyword_only"
    assert any("Ollama is down" in w for w in res.warnings)
    assert res.chunks and res.sufficient
    assert "Net 30" in res.chunks[0].chunk.text


def test_embedding_cache_is_written_and_reused(ctx, add_doc):
    a = add_doc("vendor_a.pdf", VENDOR_A_PAGES)
    retrieve(ctx, "payment", [a.id])
    cached = list(ctx.settings.embedding_cache_dir.glob(f"doc{a.id}_*.npy"))
    assert len(cached) == 1

    class CountingEmbedder(HashingEmbedder):
        calls = 0

        def embed_documents(self, texts):
            CountingEmbedder.calls += 1
            return super().embed_documents(texts)

    counting = CountingEmbedder()
    counting.name = ctx.get_embedder().name  # same cache key
    ctx._embedder_override = counting
    retrieve(ctx, "payment", [a.id])
    assert CountingEmbedder.calls == 0


def test_stale_cache_with_wrong_row_count_is_recomputed(ctx, add_doc):
    import numpy as np

    a = add_doc("vendor_a.pdf", VENDOR_A_PAGES)
    retrieve(ctx, "payment", [a.id])
    path = next(ctx.settings.embedding_cache_dir.glob(f"doc{a.id}_*.npy"))
    np.save(path, np.zeros((1, 768), dtype=np.float32))
    res = retrieve(ctx, "payment terms", [a.id])
    assert res.mode == "hybrid" and res.chunks


def test_empty_query_and_no_documents(ctx):
    res = retrieve(ctx, "anything", [])
    assert res.chunks == [] and res.sufficient is False and res.warnings


# ----------------------------------------------------------- citations ----
def test_sources_carry_document_and_page_from_metadata(ctx, add_doc):
    a = add_doc("vendor_a.pdf", VENDOR_A_PAGES)
    res = retrieve(ctx, "contract duration", [a.id])
    blocks = build_blocks(res.chunks)
    assert [b.source.source_id for b in blocks] == [f"S{i}" for i in range(1, len(blocks) + 1)]
    top = blocks[0].source
    assert top.document_name == "vendor_a.pdf" and top.page_number == 3
    assert "24 months" in blocks[0].text and top.snippet


def test_resolve_citations_drops_invented_ids_and_dedupes(ctx, add_doc):
    a = add_doc("vendor_a.pdf", VENDOR_A_PAGES)
    blocks = build_blocks(retrieve(ctx, "payment", [a.id]).chunks)
    n = len(blocks)
    resolved = resolve_citations(["S1", "[s1]", f"S{n + 5}", "banana", None, 1], blocks)
    assert [s.source_id for s in resolved] == ["S1"]


def test_normalize_source_id_variants():
    assert normalize_source_id("S3") == "S3"
    assert normalize_source_id("[s12]") == "S12"
    assert normalize_source_id(7) == "S7"
    assert normalize_source_id("none") is None and normalize_source_id(None) is None


def test_format_context_lists_names_pages_and_ids(ctx, add_doc):
    a = add_doc("vendor_a.pdf", VENDOR_A_PAGES)
    ctx_text = format_context(build_blocks(retrieve(ctx, "pricing", [a.id]).chunks))
    assert '[S1] Document: "vendor_a.pdf" | Page' in ctx_text


def test_txt_sources_have_no_page_number(ctx):
    from app.services.document_service import ingest_upload

    doc = ingest_upload(ctx, "n.txt", b"Payment terms: Net 30 days.").document
    blocks = build_blocks(retrieve(ctx, "payment terms", [doc.id]).chunks)
    assert blocks[0].source.page_number is None
    assert "Page n/a" in format_context(blocks)
    assert blocks[0].source.label == "n.txt"


def test_block_by_id(ctx, add_doc):
    a = add_doc("vendor_a.pdf", VENDOR_A_PAGES)
    blocks = build_blocks(retrieve(ctx, "payment", [a.id]).chunks)
    assert block_by_id(blocks, "[S1]") is blocks[0]
    assert block_by_id(blocks, "S99") is None
