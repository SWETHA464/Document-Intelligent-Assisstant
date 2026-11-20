"""Data-access layer. All SQL lives here; services never write SQL."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from app.database.db import Database
from app.models.domain import ChunkRecord, DocumentRecord, PageRecord


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _doc(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(**{k: row[k] for k in row.keys()})


class CollectionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, name: str, description: str = "") -> dict[str, Any]:
        with self.db.connect() as conn:
            cur = conn.execute(
                "INSERT INTO collections (name, description, created_at) VALUES (?, ?, ?)",
                (name, description, utcnow()),
            )
            return self._get(conn, cur.lastrowid)  # type: ignore[arg-type]

    def _get(self, conn: sqlite3.Connection, collection_id: int) -> dict[str, Any]:
        row = conn.execute(
            """SELECT c.*, (SELECT COUNT(*) FROM documents d WHERE d.collection_id = c.id)
                      AS document_count FROM collections c WHERE c.id = ?""",
            (collection_id,),
        ).fetchone()
        return dict(row) if row else {}

    def get(self, collection_id: int) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            return self._get(conn, collection_id) or None

    def get_by_name(self, name: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT id FROM collections WHERE name = ? COLLATE NOCASE", (name,)
            ).fetchone()
            return self._get(conn, row["id"]) if row else None

    def list(self) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT c.*, (SELECT COUNT(*) FROM documents d WHERE d.collection_id = c.id)
                          AS document_count FROM collections c ORDER BY c.name COLLATE NOCASE"""
            ).fetchall()
            return [dict(r) for r in rows]

    def delete(self, collection_id: int) -> bool:
        """Delete a collection. Its documents are kept (collection_id becomes NULL)."""
        with self.db.connect() as conn:
            cur = conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
            return cur.rowcount > 0


class DocumentRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        *,
        filename: str,
        stored_name: str,
        file_type: str,
        size_bytes: int,
        sha256: str,
        collection_id: int | None,
        status: str = "processing",
    ) -> DocumentRecord:
        with self.db.connect() as conn:
            cur = conn.execute(
                """INSERT INTO documents
                   (collection_id, filename, stored_name, file_type, size_bytes, sha256, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (collection_id, filename, stored_name, file_type, size_bytes, sha256, status, utcnow()),
            )
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (cur.lastrowid,)).fetchone()
            return _doc(row)

    def get(self, document_id: int) -> DocumentRecord | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
            return _doc(row) if row else None

    def get_by_sha256(self, sha256: str) -> DocumentRecord | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
            return _doc(row) if row else None

    def get_many(self, document_ids: Sequence[int]) -> list[DocumentRecord]:
        """Return the documents that exist, in the order requested (missing ids are skipped)."""
        if not document_ids:
            return []
        marks = ",".join("?" * len(document_ids))
        with self.db.connect() as conn:
            rows = conn.execute(f"SELECT * FROM documents WHERE id IN ({marks})", tuple(document_ids)).fetchall()
        by_id = {r["id"]: _doc(r) for r in rows}
        return [by_id[i] for i in document_ids if i in by_id]

    def list(self, collection_id: int | None = None) -> list[DocumentRecord]:
        with self.db.connect() as conn:
            if collection_id is None:
                rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC, id DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE collection_id = ? ORDER BY created_at DESC, id DESC",
                    (collection_id,),
                ).fetchall()
            return [_doc(r) for r in rows]

    def mark_ready(self, document_id: int, *, page_count: int, char_count: int, chunk_count: int) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """UPDATE documents SET status='ready', error=NULL, page_count=?, char_count=?, chunk_count=?
                   WHERE id=?""",
                (page_count, char_count, chunk_count, document_id),
            )

    def mark_failed(self, document_id: int, error: str, *, page_count: int = 0) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE documents SET status='failed', error=?, page_count=? WHERE id=?",
                (error, page_count, document_id),
            )

    def set_collection(self, document_id: int, collection_id: int | None) -> None:
        with self.db.connect() as conn:
            conn.execute("UPDATE documents SET collection_id=? WHERE id=?", (collection_id, document_id))

    def delete(self, document_id: int) -> bool:
        with self.db.connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            return cur.rowcount > 0

    def counts_by_status(self) -> dict[str, int]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM documents GROUP BY status").fetchall()
            return {r["status"]: r["n"] for r in rows}


class ContentRepository:
    """Pages and chunks belonging to documents."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def save_pages(self, document_id: int, pages: Iterable[PageRecord]) -> None:
        with self.db.connect() as conn:
            conn.executemany(
                "INSERT INTO pages (document_id, page_number, text) VALUES (?, ?, ?)",
                [(document_id, p.page_number, p.text) for p in pages],
            )

    def save_chunks(self, document_id: int, chunks: Iterable[tuple[int, int | None, str]]) -> int:
        """chunks: (chunk_index, page_number, text). Returns number saved."""
        rows = [(document_id, i, p, t) for i, p, t in chunks]
        with self.db.connect() as conn:
            conn.executemany(
                "INSERT INTO chunks (document_id, chunk_index, page_number, text) VALUES (?, ?, ?, ?)", rows
            )
        return len(rows)

    def get_pages(self, document_id: int) -> list[PageRecord]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT document_id, page_number, text FROM pages WHERE document_id=? ORDER BY id",
                (document_id,),
            ).fetchall()
            return [PageRecord(r["document_id"], r["page_number"], r["text"]) for r in rows]

    def get_chunks(self, document_ids: Sequence[int]) -> list[ChunkRecord]:
        """Chunks for the given documents, ordered by (document order requested, chunk_index)."""
        if not document_ids:
            return []
        marks = ",".join("?" * len(document_ids))
        with self.db.connect() as conn:
            rows = conn.execute(
                f"""SELECT c.id, c.document_id, d.filename AS document_name, c.chunk_index,
                           c.page_number, c.text
                    FROM chunks c JOIN documents d ON d.id = c.document_id
                    WHERE c.document_id IN ({marks})""",
                tuple(document_ids),
            ).fetchall()
        order = {doc_id: pos for pos, doc_id in enumerate(document_ids)}
        chunks = [
            ChunkRecord(r["id"], r["document_id"], r["document_name"], r["chunk_index"], r["page_number"], r["text"])
            for r in rows
        ]
        chunks.sort(key=lambda c: (order[c.document_id], c.chunk_index))
        return chunks


class AnalysisRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self, *, operation: str, title: str, document_ids: list[int], params: dict, result: dict
    ) -> dict[str, Any]:
        with self.db.connect() as conn:
            cur = conn.execute(
                """INSERT INTO analyses (operation, title, document_ids, params, result, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (operation, title, json.dumps(document_ids), json.dumps(params), json.dumps(result), utcnow()),
            )
            row = conn.execute("SELECT * FROM analyses WHERE id = ?", (cur.lastrowid,)).fetchone()
            return self._row(row)

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "operation": row["operation"],
            "title": row["title"],
            "document_ids": json.loads(row["document_ids"]),
            "params": json.loads(row["params"]),
            "result": json.loads(row["result"]),
            "created_at": row["created_at"],
        }

    def get(self, analysis_id: int) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
            return self._row(row) if row else None

    def get_many(self, analysis_ids: Sequence[int]) -> list[dict[str, Any]]:
        found = [self.get(i) for i in analysis_ids]
        return [a for a in found if a]

    def list(self, limit: int = 100, operation: str | None = None) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            if operation:
                rows = conn.execute(
                    "SELECT * FROM analyses WHERE operation=? ORDER BY id DESC LIMIT ?", (operation, limit)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [self._row(r) for r in rows]

    def delete(self, analysis_id: int) -> bool:
        with self.db.connect() as conn:
            cur = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
            return cur.rowcount > 0

    def counts_by_operation(self) -> dict[str, int]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT operation, COUNT(*) AS n FROM analyses GROUP BY operation").fetchall()
            return {r["operation"]: r["n"] for r in rows}


class SettingsRepository:
    """Key/value store for runtime-editable settings."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def get_all(self) -> dict[str, Any]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    def set_many(self, values: dict[str, Any]) -> None:
        with self.db.connect() as conn:
            conn.executemany(
                "INSERT INTO app_settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [(k, json.dumps(v)) for k, v in values.items()],
            )

    def clear(self) -> None:
        with self.db.connect() as conn:
            conn.execute("DELETE FROM app_settings")
