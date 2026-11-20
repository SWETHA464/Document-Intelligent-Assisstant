"""Configurable embedding models.

All embedders return float32, L2-normalised vectors so that inner product == cosine similarity.
"""
from __future__ import annotations

import logging
import zlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from app.core.errors import EmbeddingError
from app.utils.text import tokenize

if TYPE_CHECKING:  # pragma: no cover
    from app.core.config import Settings

logger = logging.getLogger(__name__)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder(ABC):
    #: Identifies provider+model; part of the on-disk cache key.
    name: str = "embedder"

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Return a vector of shape (dim,)."""


class HashingEmbedder(Embedder):
    """Offline lexical 'embedding' using the hashing trick (unigrams + bigrams).

    It needs no model download and is deterministic, but it is NOT semantic: it only
    captures word overlap. Use it for demos, tests and as an offline fallback.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            h = zlib.crc32(feature.encode("utf-8"))
            sign = 1.0 if (h >> 31) & 1 else -1.0
            vec[h % self.dim] += sign
        return np.sign(vec) * np.sqrt(np.abs(vec))  # sub-linear term frequency

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vector(text))[0]


class OllamaEmbedder(Embedder):
    """Embeddings from a local Ollama server via langchain-ollama."""

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url
        self.name = f"ollama:{model}"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError as exc:
                raise EmbeddingError(
                    "langchain-ollama is not installed. Run: pip install langchain-ollama"
                ) from exc
            self._client = OllamaEmbeddings(model=self.model, base_url=self.base_url)
        return self._client

    def _wrap(self, exc: Exception) -> EmbeddingError:
        return EmbeddingError(
            f"Could not get embeddings from Ollama model '{self.model}' at {self.base_url}. "
            f"Is Ollama running and is the model pulled (ollama pull {self.model})? Details: {exc}"
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        client = self._get_client()
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), 32):
                vectors.extend(client.embed_documents(texts[start : start + 32]))
        except EmbeddingError:
            raise
        except Exception as exc:
            raise self._wrap(exc) from exc
        return _normalize(np.array(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        client = self._get_client()
        try:
            return _normalize(np.array(client.embed_query(text)))[0]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise self._wrap(exc) from exc


class SentenceTransformerEmbedder(Embedder):
    """Local sentence-transformers model (optional dependency)."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.name = f"sentence_transformers:{model}"
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Run: pip install sentence-transformers"
                ) from exc
            try:
                self._model = SentenceTransformer(self.model)
            except Exception as exc:
                raise EmbeddingError(f"Could not load embedding model '{self.model}': {exc}") from exc
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        try:
            return _normalize(self._get_model().encode(texts, normalize_embeddings=True))
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def create_embedder(settings: "Settings") -> Embedder:
    provider = settings.embedding_provider
    if provider == "hashing":
        return HashingEmbedder()
    if provider == "ollama":
        return OllamaEmbedder(settings.embedding_model, settings.effective_embedding_base_url)
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    raise EmbeddingError(
        f"Unknown EMBEDDING_PROVIDER '{provider}'. Use: ollama, sentence_transformers or hashing."
    )
