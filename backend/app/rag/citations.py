"""Source tracking: number the passages shown to the model and map its citations back.

The model only ever sees opaque ids like [S3]. Document names and page numbers are attached
here from stored metadata, so a model cannot invent a page number or a document.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.models.domain import ChunkRecord, RetrievedChunk
from app.schemas.common import Source
from app.utils.text import truncate

_ID = re.compile(r"(\d+)")


@dataclass
class ContextBlock:
    source: Source
    text: str


def build_blocks(chunks: Iterable[ChunkRecord | RetrievedChunk], *, start: int = 1) -> list[ContextBlock]:
    """Number chunks S1, S2, ... (deduplicating by chunk id)."""
    blocks: list[ContextBlock] = []
    seen: set[int] = set()
    for item in chunks:
        chunk = item.chunk if isinstance(item, RetrievedChunk) else item
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        source = Source(
            source_id=f"S{start + len(blocks)}",
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            page_number=chunk.page_number,
            chunk_id=chunk.id,
            snippet=truncate(chunk.text, 280),
        )
        blocks.append(ContextBlock(source, chunk.text))
    return blocks


def format_context(blocks: list[ContextBlock]) -> str:
    """Render blocks as the numbered excerpt list placed in a prompt."""
    parts = []
    for b in blocks:
        page = f"Page {b.source.page_number}" if b.source.page_number else "Page n/a"
        parts.append(f'[{b.source.source_id}] Document: "{b.source.document_name}" | {page}\n{b.text}')
    return "\n\n".join(parts)


def normalize_source_id(raw: object) -> str | None:
    """Accept 'S3', 's3', '[S3]', 3 ... and return 'S3' (or None if unusable)."""
    if raw is None:
        return None
    match = _ID.search(str(raw))
    return f"S{int(match.group(1))}" if match else None


def resolve_citations(raw_ids: Iterable[object] | None, blocks: list[ContextBlock]) -> list[Source]:
    """Keep only ids that really exist in `blocks`, de-duplicated, in the order given."""
    by_id = {b.source.source_id: b.source for b in blocks}
    resolved: list[Source] = []
    for raw in raw_ids or []:
        sid = normalize_source_id(raw)
        source = by_id.get(sid) if sid else None
        if source and source not in resolved:
            resolved.append(source)
    return resolved


def block_by_id(blocks: list[ContextBlock], source_id: object) -> ContextBlock | None:
    sid = normalize_source_id(source_id)
    return next((b for b in blocks if b.source.source_id == sid), None)
