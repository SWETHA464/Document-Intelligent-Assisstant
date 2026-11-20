"""Hybrid retrieval: BM25 keyword scores fused with FAISS semantic scores.

Flow for a `RetrievalSession` (one per operation, over the selected documents):
  1. Load the stored chunks of the selected documents.
  2. Build a BM25 index over them.
  3. Load (or compute and cache on disk) chunk embeddings, then build a FAISS index in memory.
  4. For each query: combined = alpha * semantic + (1 - alpha) * keyword.
If embeddings are unavailable (e.g. Ollama is not running) the session degrades to
keyword-only retrieval and says so in `warnings`, instead of failing.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from app.core.context import AppContext
from app.models.domain import ChunkRecord, RetrievalResult, RetrievedChunk
from app.rag.bm25 import BM25Index
from app.rag.vector_store import VectorStore
from app.utils.text import tokenize

logger = logging.getLogger(__name__)


def _cache_path(cache_dir: Path, document_id: int, embedder_name: str) -> Path:
    sig = hashlib.sha1(embedder_name.encode("utf-8")).hexdigest()[:8]
    return cache_dir / f"doc{document_id}_{sig}.npy"


class RetrievalSession:
    def __init__(self, ctx: AppContext, document_ids: list[int]) -> None:
        self.ctx = ctx
        self.settings = ctx.settings
        self.document_ids = list(dict.fromkeys(document_ids))
        self.chunks: list[ChunkRecord] = ctx.content.get_chunks(self.document_ids)
        self._tokens = [tokenize(c.text) for c in self.chunks]
        self._token_sets = [set(t) for t in self._tokens]
        self._bm25 = BM25Index(self._tokens) if self.chunks else None
        self._store: VectorStore | None = None
        self._semantic_loaded = False
        self._semantic_error: str | None = None
        self.vector_backend: str | None = None

    # ------------------------------------------------------ embeddings --
    def _document_vectors(self, doc_id: int, chunks: list[ChunkRecord]) -> np.ndarray:
        embedder = self.ctx.get_embedder()
        path = _cache_path(self.settings.embedding_cache_dir, doc_id, embedder.name)
        if path.exists():
            try:
                cached = np.load(path)
                if cached.shape[0] == len(chunks):
                    return cached
            except Exception:  # corrupted cache file: recompute
                logger.warning("Ignoring unreadable embedding cache %s", path)
        vectors = embedder.embed_documents([c.text for c in chunks])
        try:
            self.settings.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)
        except OSError:  # caching is an optimisation only
            logger.warning("Could not write embedding cache %s", path)
        return vectors

    def _ensure_semantic(self) -> None:
        if self._semantic_loaded or not self.chunks:
            return
        self._semantic_loaded = True
        try:
            parts = []
            for doc_id in self.document_ids:
                doc_chunks = [c for c in self.chunks if c.document_id == doc_id]
                if doc_chunks:
                    parts.append(self._document_vectors(doc_id, doc_chunks))
            self._store = VectorStore(np.vstack(parts))
            self.vector_backend = self._store.backend
        except Exception as exc:  # embedder down, dimension mismatch, ...
            self._semantic_error = getattr(exc, "message", None) or str(exc)
            logger.warning("Semantic retrieval unavailable, using keyword-only: %s", self._semantic_error)
            self._store = None

    # ----------------------------------------------------------- search --
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        per_doc_min: int = 0,
        only_document_ids: list[int] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.settings.retrieval_top_k
        result = RetrievalResult()
        allowed = [
            i for i, c in enumerate(self.chunks) if only_document_ids is None or c.document_id in only_document_ids
        ]
        if not allowed or not query.strip():
            result.warnings.append("No indexed content to search.")
            return result

        q_tokens = tokenize(query)
        keyword = self._bm25.scores(q_tokens)  # type: ignore[union-attr]
        top_kw = float(keyword.max()) if len(keyword) else 0.0
        keyword_norm = keyword / top_kw if top_kw > 0 else keyword

        self._ensure_semantic()
        semantic = np.zeros(len(self.chunks), dtype=np.float32)
        use_semantic = False
        if self._store is not None:
            try:
                q_vec = self.ctx.get_embedder().embed_query(query)
                for idx, score in self._store.search(q_vec, k=len(self.chunks)):
                    semantic[idx] = max(score, 0.0)
                use_semantic = True
            except Exception as exc:
                self._semantic_error = getattr(exc, "message", None) or str(exc)
                logger.warning("Query embedding failed, keyword-only: %s", self._semantic_error)

        if use_semantic:
            alpha = self.settings.hybrid_alpha
            combined = alpha * semantic + (1 - alpha) * keyword_norm
            result.mode = "hybrid"
        else:
            combined = keyword_norm
            result.mode = "keyword_only"
            result.warnings.append(
                "Semantic search was unavailable, so keyword (BM25) search was used"
                + (f": {self._semantic_error}" if self._semantic_error else ".")
            )

        ranked = sorted(allowed, key=lambda i: (-float(combined[i]), i))
        chosen = [i for i in ranked if combined[i] > 0][:top_k]
        if per_doc_min > 0:
            doc_ids = only_document_ids if only_document_ids is not None else self.document_ids
            for doc_id in doc_ids:
                have = [i for i in chosen if self.chunks[i].document_id == doc_id]
                extra = [
                    i for i in ranked
                    if self.chunks[i].document_id == doc_id and i not in chosen and combined[i] > 0
                ]
                chosen.extend(extra[: max(0, per_doc_min - len(have))])

        q_set = set(q_tokens)
        for i in chosen:
            coverage = len(q_set & self._token_sets[i]) / len(q_set) if q_set else 0.0
            result.chunks.append(
                RetrievedChunk(
                    chunk=self.chunks[i],
                    score=float(combined[i]),
                    semantic_score=float(semantic[i]),
                    keyword_score=float(keyword_norm[i]),
                    coverage=coverage,
                )
            )
        result.chunks.sort(key=lambda rc: -rc.score)
        result.sufficient = any(
            rc.coverage >= self.settings.min_term_coverage
            or (use_semantic and rc.semantic_score >= self.settings.min_semantic_score)
            for rc in result.chunks
        )
        if not result.chunks:
            result.warnings.append("No relevant passages were found in the selected documents.")
        return result


def open_session(ctx: AppContext, document_ids: list[int]) -> RetrievalSession:
    return RetrievalSession(ctx, document_ids)


def retrieve(
    ctx: AppContext,
    query: str,
    document_ids: list[int],
    *,
    top_k: int | None = None,
    per_doc_min: int = 0,
) -> RetrievalResult:
    """One-shot convenience wrapper around `RetrievalSession.search`."""
    return open_session(ctx, document_ids).search(query, top_k=top_k, per_doc_min=per_doc_min)
