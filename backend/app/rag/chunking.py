"""Page-aware chunking.

Each chunk comes from exactly ONE page so that citations can name a precise page.
Text is packed line by line (PDF text is line-structured, and pricing tables rely on
line breaks) up to `chunk_size` characters, with `overlap` characters of trailing
context carried into the next chunk of the same page.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.services.text_extraction.base import ExtractedPage

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkDraft:
    chunk_index: int
    page_number: int | None
    text: str


def _hard_split(text: str, size: int) -> list[str]:
    """Split a very long run of text on word boundaries."""
    parts: list[str] = []
    current = ""
    for word in text.split(" "):
        while len(word) > size:  # a single "word" longer than the chunk (e.g. a long URL)
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:size])
            word = word[size:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= size:
            current = f"{current} {word}"
        else:
            parts.append(current)
            current = word
    if current:
        parts.append(current)
    return parts


def _split_long_line(line: str, size: int) -> list[str]:
    if len(line) <= size:
        return [line]
    pieces: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_END.split(line):
        if len(sentence) > size:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.extend(_hard_split(sentence, size))
        elif not buffer:
            buffer = sentence
        elif len(buffer) + 1 + len(sentence) <= size:
            buffer = f"{buffer} {sentence}"
        else:
            pieces.append(buffer)
            buffer = sentence
    if buffer:
        pieces.append(buffer)
    return pieces


def _joined_len(lines: list[str]) -> int:
    return sum(len(x) for x in lines) + max(len(lines) - 1, 0)


def _tail_lines(lines: list[str], overlap: int) -> list[str]:
    """Trailing lines whose combined length fits within `overlap` characters."""
    if overlap <= 0:
        return []
    tail: list[str] = []
    for line in reversed(lines):
        if _joined_len([line, *tail]) > overlap:
            break
        tail.insert(0, line)
    return tail


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into chunks of at most ~chunk_size characters."""
    if chunk_size < 50:
        raise ValueError("chunk_size must be at least 50")
    overlap = max(0, min(overlap, chunk_size // 2))
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line:
            lines.extend(_split_long_line(line, chunk_size))

    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        if current and _joined_len([*current, line]) > chunk_size:
            chunks.append("\n".join(current))
            current = _tail_lines(current, overlap)
            while current and _joined_len([*current, line]) > chunk_size:
                current.pop(0)
        current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_pages(pages: Iterable[ExtractedPage], chunk_size: int = 1000, overlap: int = 150) -> list[ChunkDraft]:
    """Chunk every page separately; chunk_index runs across the whole document."""
    drafts: list[ChunkDraft] = []
    for page in pages:
        for piece in chunk_text(page.text, chunk_size, overlap):
            drafts.append(ChunkDraft(len(drafts), page.page_number, piece))
    return drafts
