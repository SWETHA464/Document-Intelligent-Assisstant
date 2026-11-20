"""SQLite connection handling and schema creation (stdlib sqlite3, no ORM)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER REFERENCES collections(id) ON DELETE SET NULL,
    filename      TEXT NOT NULL,
    stored_name   TEXT NOT NULL,
    file_type     TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    sha256        TEXT NOT NULL UNIQUE,
    status        TEXT NOT NULL,            -- processing | ready | failed
    error         TEXT,
    page_count    INTEGER NOT NULL DEFAULT 0,
    char_count    INTEGER NOT NULL DEFAULT 0,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER,                    -- NULL for formats without pages (TXT)
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pages_document ON pages(document_id);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE TABLE IF NOT EXISTS analyses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    operation    TEXT NOT NULL,             -- qa | compare | extract | summarize
    title        TEXT NOT NULL,
    document_ids TEXT NOT NULL,             -- JSON list
    params       TEXT NOT NULL,             -- JSON object (the request)
    result       TEXT NOT NULL,             -- JSON object (the result)
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Database:
    """Thin wrapper that opens a short-lived connection per unit of work."""

    def __init__(self, path: Path | str) -> None:
        self.path = str(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
