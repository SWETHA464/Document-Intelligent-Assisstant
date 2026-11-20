"""Vector search: FAISS when available, exact numpy search otherwise (identical results)."""
from __future__ import annotations

import numpy as np

try:  # faiss-cpu is optional at runtime; the numpy fallback keeps the app working
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except Exception:  # ImportError, or a broken binary wheel
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


class VectorStore:
    """In-memory exact inner-product index over L2-normalised vectors (cosine similarity)."""

    def __init__(self, vectors: np.ndarray, *, use_faiss: bool | None = None) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("VectorStore needs a non-empty 2-D array of vectors")
        self.size, self.dim = vectors.shape
        want_faiss = FAISS_AVAILABLE if use_faiss is None else (use_faiss and FAISS_AVAILABLE)
        self.backend = "faiss" if want_faiss else "numpy"
        if want_faiss:
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(vectors)
        else:
            self._matrix = vectors

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to k (row_index, cosine_similarity) pairs, best first."""
        k = max(1, min(k, self.size))
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"Query dimension {q.shape[1]} does not match index dimension {self.dim}")
        if self.backend == "faiss":
            scores, ids = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        scores = (self._matrix @ q[0]).astype(np.float32)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order]
