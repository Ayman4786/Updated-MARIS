"""SQLite-backed metadata and conversation persistence for MARIS."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


STORAGE_ROOT = Path(os.getenv("MARIS_STORAGE_DIR", "storage"))
DOCUMENTS_ROOT = STORAGE_ROOT / "documents"
DATABASE_PATH = STORAGE_ROOT / "maris.db"
CONTEXT_MESSAGE_LIMIT = int(os.getenv("MARIS_CONTEXT_MESSAGE_LIMIT", "8"))
DIAGRAM_ENRICHMENT_VERSION = 1
CURRENT_UPLOAD_DOCUMENT: ContextVar[str | None] = ContextVar(
    "current_upload_document", default=None
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MarisStore:
    """Small transaction-scoped repository; callers never share connections."""

    def __init__(self, database_path: Path | str = DATABASE_PATH) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    pages INTEGER,
                    document_folder TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_status
                    ON documents(status);
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_document
                    ON conversations(document_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, created_at);
                """
            )
            if connection.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
                connection.execute("INSERT INTO schema_version(version) VALUES (1)")
        self._register_legacy_documents()

    def _register_legacy_documents(self) -> None:
        if not DOCUMENTS_ROOT.exists():
            return
        for folder in DOCUMENTS_ROOT.iterdir():
            if not folder.is_dir():
                continue
            pdfs = list(folder.glob("*.pdf"))
            if not pdfs:
                continue
            pdf_path = pdfs[0]
            digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            with self.connection() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO documents
                    (document_id, content_hash, filename, uploaded_at, status,
                     error, pages, document_folder)
                    VALUES (?, ?, ?, ?, 'success', NULL, ?, ?)
                    """,
                    (
                        folder.name,
                        digest,
                        pdf_path.name,
                        datetime.fromtimestamp(
                            pdf_path.stat().st_mtime, timezone.utc
                        ).isoformat(),
                        self._legacy_page_count(folder),
                        str(folder),
                    ),
                )

    @staticmethod
    def _legacy_page_count(folder: Path) -> int | None:
        pages_path = folder / "pages.json"
        if not pages_path.exists():
            return None
        try:
            pages = json.loads(pages_path.read_text(encoding="utf-8"))
            return len(pages) if isinstance(pages, list) else None
        except (OSError, ValueError):
            return None

    def find_document_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return dict(row) if row else None

    def begin_document(
        self, content_hash: str, filename: str, document_id: str
    ) -> tuple[str, dict[str, Any] | None]:
        """Reserve a hash and return ``(action, row)`` for upload processing."""
        now = utc_now()
        folder = str(DOCUMENTS_ROOT / document_id)
        with self.connection() as connection:
            existing = connection.execute(
                "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
            ).fetchone()
            if existing and existing["status"] == "success":
                return "reuse", dict(existing)
            if existing:
                connection.execute(
                    """
                    UPDATE documents SET filename=?, uploaded_at=?, status='processing',
                    error=NULL WHERE content_hash=?
                    """,
                    (filename, now, content_hash),
                )
                row = connection.execute(
                    "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
                ).fetchone()
                return "process", dict(row)
            try:
                connection.execute(
                    """
                    INSERT INTO documents
                    (document_id, content_hash, filename, uploaded_at, status,
                     document_folder)
                    VALUES (?, ?, ?, ?, 'processing', ?)
                    """,
                    (document_id, content_hash, filename, now, folder),
                )
            except sqlite3.IntegrityError:
                row = connection.execute(
                    "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
                ).fetchone()
                if row and row["status"] == "success":
                    return "reuse", dict(row)
                raise
        return "process", None

    def complete_document(self, document_id: str, pages: int | None) -> None:
        with self.connection() as connection:
            connection.execute(
                "UPDATE documents SET status='success', error=NULL, pages=? "
                "WHERE document_id=?",
                (pages, document_id),
            )

    def fail_document(self, document_id: str, error: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "UPDATE documents SET status='failed', error=? WHERE document_id=?",
                (error[:2000], document_id),
            )

    def mark_for_reprocessing(self, document_id: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "UPDATE documents SET status='processing', error=NULL "
                "WHERE document_id=?",
                (document_id,),
            )

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id=?", (document_id,)
            ).fetchone()
        return dict(row) if row else None

    def create_conversation(self, document_id: str, title: str = "New chat") -> dict[str, Any]:
        conversation_id = uuid4().hex
        now = utc_now()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO conversations
                (conversation_id, title, document_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, title[:200] or "New chat", document_id, now, now),
            )
        return self.get_conversation(conversation_id)  # type: ignore[return-value]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_conversations(self, document_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM conversations"
        values: tuple[Any, ...] = ()
        if document_id:
            query += " WHERE document_id=?"
            values = (document_id,)
        query += " ORDER BY updated_at DESC"
        with self.connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return [dict(row) for row in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        with self.connection() as connection:
            result = connection.execute(
                "DELETE FROM conversations WHERE conversation_id=?", (conversation_id,)
            )
        return result.rowcount > 0

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, Any] | None:
        clean_title = title.strip().replace("\n", " ")[:200] or "New chat"
        with self.connection() as connection:
            connection.execute(
                "UPDATE conversations SET title=? WHERE conversation_id=?",
                (clean_title, conversation_id),
            )
        return self.get_conversation(conversation_id)

    def add_message(
        self,
        conversation_id: str,
        role: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message_id = uuid4().hex
        now = utc_now()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO messages
                (message_id, conversation_id, role, text, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, conversation_id, role, text, now, json.dumps(metadata or {})),
            )
            connection.execute(
                "UPDATE conversations SET updated_at=? WHERE conversation_id=?",
                (now, conversation_id),
            )
            if role == "user":
                connection.execute(
                    """
                    UPDATE conversations SET title=?
                    WHERE conversation_id=? AND title='New chat'
                    """,
                    (text.strip().replace("\n", " ")[:80] or "New chat", conversation_id),
                )
        return {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "text": text,
            "created_at": now,
            "metadata": metadata or {},
        }

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item.pop("metadata_json"))
            except (TypeError, ValueError):
                item["metadata"] = {}
                item.pop("metadata_json", None)
            result.append(item)
        return result

    def recent_messages(self, conversation_id: str, limit: int = CONTEXT_MESSAGE_LIMIT) -> list[dict[str, Any]]:
        messages = self.list_messages(conversation_id)
        return messages[-max(0, limit):]
