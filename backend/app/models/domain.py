"""Internal domain objects (plain dataclasses, independent of the API layer)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DocumentRecord:
    id: int
    collection_id: int | None
    filename: str
    stored_name: str
    file_type: str
    size_bytes: int
    sha256: str
    status: str
    error: str | None
    page_count: int
    char_count: int
    chunk_count: int
    created_at: str


@dataclass
class PageRecord:
    document_id: int
    page_number: int | None
    text: str


@dataclass
class ChunkRecord:
    """A stored chunk plus the name of the document it came from."""

    id: int
    document_id: int
    document_name: str
    chunk_index: int
    page_number: int | None
    text: str


@dataclass
class RetrievedChunk:
    chunk: ChunkRecord
    score: float  # combined score used for ranking
    semantic_score: float = 0.0
    keyword_score: float = 0.0  # BM25 normalised to 0..1
    coverage: float = 0.0  # fraction of query terms present in the chunk


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    mode: str = "hybrid"  # "hybrid" | "keyword_only"
    sufficient: bool = False
    warnings: list[str] = field(default_factory=list)
