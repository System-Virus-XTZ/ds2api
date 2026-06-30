"""
SQLite Chat History Store

Python port of Go SQLite-based chat history storage.
Handles persistent storage of chat history, messages, citations, files, and responses.
Supports TTL-based cleanup and file attachment management.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config.logger import get_logger

logger = get_logger("chathistory.store")


# Constants
FILE_VERSION = 2
DISABLED_LIMIT = 0
DEFAULT_LIMIT = 20
MAX_LIMIT = 50
DEFAULT_PREVIEW_AT = 160

ALLOWED_LIMITS = {DISABLED_LIMIT, 10, 20, 50}

# Error types
class ChatHistoryError(Exception):
    """Base exception for chat history errors."""
    pass


class ErrDisabled(ChatHistoryError):
    """Chat history is disabled."""
    pass


class ErrNotFound(ChatHistoryError):
    """Entry not found."""
    pass


class ErrConflict(ChatHistoryError):
    """Conflict error."""
    pass


# Data classes
@dataclass
class Entry:
    """Chat history entry."""
    id: str
    revision: int = 1
    title: str = ""
    preview: str = ""
    created_at: int = 0
    updated_at: int = 0
    sources: List[str] = field(default_factory=list)
    model: str = ""
    disabled: bool = False


@dataclass
class Message:
    """Chat message."""
    id: int = 0
    history_id: str = ""
    revision: int = 1
    role: str = ""
    content: str = ""
    thinking: str = ""
    created_at: int = 0
    seq: int = 0


@dataclass
class Citation:
    """Citation reference."""
    id: int = 0
    message_id: int = 0
    index: int = 0
    url: str = ""
    title: str = ""
    text: str = ""


@dataclass
class MessageFile:
    """File attached to message."""
    id: int = 0
    message_id: int = 0
    file_id: str = ""
    file_name: str = ""


@dataclass
class Attachment:
    """File attachment."""
    id: int = 0
    history_id: str = ""
    file_id: str = ""
    file_name: str = ""
    file_path: str = ""
    file_size: int = 0
    mime_type: str = ""
    created_at: int = 0


@dataclass
class Response:
    """Saved response."""
    id: int = 0
    history_id: str = ""
    message_id: int = 0
    model: str = ""
    content: str = ""
    thinking: str = ""
    finish_reason: str = ""
    usage: str = ""
    created_at: int = 0


class ChatHistoryStore:
    """
    SQLite-based chat history store.

    Provides persistent storage for chat history with:
    - TTL-based cleanup
    - File attachment handling
    - Citation tracking
    - Message versioning
    """

    def __init__(
        self,
        path: Optional[str] = None,
        cleanup_interval: int = 3600,
        ttl_seconds: int = 0,
    ):
        """
        Initialize chat history store.

        Args:
            path: Path to SQLite database file
            cleanup_interval: Interval between cleanup runs (seconds)
            ttl_seconds: TTL for entries (0 = no expiry)
        """
        self._path = path
        self._cleanup_interval = cleanup_interval
        self._ttl_seconds = ttl_seconds
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._last_cleanup = 0.0
        self._closed = False

        # Initialize on disk or in-memory
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._conn = self._open(path)
        else:
            self._conn = self._open_in_memory()

        self._init_schema()

    def _open(self, path: str) -> sqlite3.Connection:
        """Open SQLite database."""
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _open_in_memory(self) -> sqlite3.Connection:
        """Open in-memory database."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._lock:
            cursor = self._conn.cursor()

            # chat_history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id TEXT PRIMARY KEY,
                    revision INTEGER DEFAULT 1,
                    title TEXT NOT NULL DEFAULT '',
                    preview TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    sources TEXT NOT NULL DEFAULT '[]',
                    model TEXT NOT NULL DEFAULT '',
                    disabled INTEGER NOT NULL DEFAULT 0
                )
            """)

            # messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    history_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    thinking TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL,
                    seq INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (history_id) REFERENCES chat_history(id) ON DELETE CASCADE
                )
            """)

            # citations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS citations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    idx INTEGER NOT NULL,
                    url TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
                )
            """)

            # message_files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS message_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    file_id TEXT NOT NULL DEFAULT '',
                    file_name TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
                )
            """)

            # attachments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attachments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    history_id TEXT NOT NULL,
                    file_id TEXT NOT NULL DEFAULT '',
                    file_name TEXT NOT NULL DEFAULT '',
                    file_path TEXT NOT NULL DEFAULT '',
                    file_size INTEGER NOT NULL DEFAULT 0,
                    mime_type TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (history_id) REFERENCES chat_history(id) ON DELETE CASCADE
                )
            """)

            # responses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    history_id TEXT NOT NULL,
                    message_id INTEGER NOT NULL,
                    model TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '',
                    thinking TEXT NOT NULL DEFAULT '',
                    finish_reason TEXT NOT NULL DEFAULT '',
                    usage TEXT NOT NULL DEFAULT '{}',
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (history_id) REFERENCES chat_history(id) ON DELETE CASCADE
                )
            """)

            # Indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_history_id
                ON messages(history_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_created_at
                ON messages(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_citations_message_id
                ON citations(message_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_responses_history_id
                ON responses(history_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_attachments_history_id
                ON attachments(history_id)
            """)

            self._conn.commit()

    def _maybe_cleanup(self) -> None:
        """Run cleanup if interval elapsed."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        threading.Thread(target=self._cleanup_task, daemon=True).start()

    def _cleanup_task(self) -> None:
        """Background cleanup task."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def cleanup(self, ttl_seconds: Optional[int] = None) -> int:
        """
        Clean up expired entries.

        Args:
            ttl_seconds: TTL in seconds (uses default if None)

        Returns:
            Number of entries deleted
        """
        ttl = ttl_seconds or self._ttl_seconds
        if ttl <= 0:
            return 0

        cutoff = int(time.time()) - ttl

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT id FROM chat_history WHERE updated_at < ?",
                (cutoff,)
            )
            ids = [row[0] for row in cursor.fetchall()]

            if ids:
                placeholders = ",".join("?" * len(ids))
                cursor.execute(f"DELETE FROM chat_history WHERE id IN ({placeholders})", ids)

                # Clean orphaned attachments
                cursor.execute(
                    "DELETE FROM attachments WHERE history_id NOT IN (SELECT id FROM chat_history)"
                )

            deleted = len(ids)
            self._conn.commit()

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} expired history entries")

            return deleted

    # === History Operations ===

    def list(
        self,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> List[Entry]:
        """
        List chat history entries.

        Args:
            limit: Maximum entries to return
            offset: Offset for pagination

        Returns:
            List of Entry objects
        """
        if limit not in ALLOWED_LIMITS:
            limit = DEFAULT_LIMIT

        if limit == DISABLED_LIMIT:
            return []

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT id, revision, title, preview, created_at, updated_at,
                       sources, model, disabled
                FROM chat_history
                WHERE disabled = 0
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

            entries = []
            for row in cursor.fetchall():
                entries.append(Entry(
                    id=row["id"],
                    revision=row["revision"],
                    title=row["title"],
                    preview=row["preview"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    sources=json.loads(row["sources"]),
                    model=row["model"],
                    disabled=bool(row["disabled"]),
                ))

            return entries

    def get(self, history_id: str) -> Optional[Entry]:
        """
        Get a single history entry.

        Args:
            history_id: History ID

        Returns:
            Entry or None
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT id, revision, title, preview, created_at, updated_at,
                       sources, model, disabled
                FROM chat_history
                WHERE id = ?
            """, (history_id,))

            row = cursor.fetchone()
            if not row:
                return None

            return Entry(
                id=row["id"],
                revision=row["revision"],
                title=row["title"],
                preview=row["preview"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                sources=json.loads(row["sources"]),
                model=row["model"],
                disabled=bool(row["disabled"]),
            )

    def save(
        self,
        history_id: str,
        title: str = "",
        model: str = "",
        sources: Optional[List[str]] = None,
        preview: str = "",
    ) -> Entry:
        """
        Save or update a history entry.

        Args:
            history_id: History ID
            title: Entry title
            model: Model used
            sources: Source URLs
            preview: Preview text

        Returns:
            Saved Entry
        """
        sources = sources or []
        now = int(time.time())

        with self._lock:
            cursor = self._conn.cursor()

            # Check if exists
            cursor.execute("SELECT revision FROM chat_history WHERE id = ?", (history_id,))
            row = cursor.fetchone()

            if row:
                # Update existing
                revision = row["revision"] + 1
                cursor.execute("""
                    UPDATE chat_history
                    SET revision = ?, title = ?, model = ?, sources = ?,
                        preview = ?, updated_at = ?
                    WHERE id = ?
                """, (revision, title, model, json.dumps(sources), preview, now, history_id))
            else:
                # Insert new
                revision = 1
                cursor.execute("""
                    INSERT INTO chat_history (id, revision, title, model, sources, preview, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (history_id, revision, title, model, json.dumps(sources), preview, now, now))

            self._conn.commit()

            return Entry(
                id=history_id,
                revision=revision,
                title=title,
                preview=preview,
                created_at=now,
                updated_at=now,
                sources=sources,
                model=model,
                disabled=False,
            )

    def delete(self, history_id: str) -> bool:
        """
        Delete a history entry and all associated data.

        Args:
            history_id: History ID

        Returns:
            True if deleted
        """
        with self._lock:
            cursor = self._conn.cursor()

            # Get attachments to clean up files
            cursor.execute("SELECT file_path FROM attachments WHERE history_id = ?", (history_id,))
            file_paths = [row["file_path"] for row in cursor.fetchall()]

            # Delete from DB (cascades to related tables)
            cursor.execute("DELETE FROM chat_history WHERE id = ?", (history_id,))
            deleted = cursor.rowcount > 0
            self._conn.commit()

            # Clean up files
            for path in file_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass

            if deleted:
                logger.info(f"Deleted history: {history_id}")

            return deleted

    def touch(self, history_id: str) -> None:
        """Update the updated_at timestamp."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "UPDATE chat_history SET updated_at = ? WHERE id = ?",
                (int(time.time()), history_id)
            )
            self._conn.commit()

    # === Message Operations ===

    def append_message(
        self,
        history_id: str,
        role: str,
        content: str,
        thinking: str = "",
        attachments: Optional[List[Dict]] = None,
    ) -> Message:
        """
        Append a message to history.

        Args:
            history_id: History ID
            role: Message role (user/assistant)
            content: Message content
            thinking: Thinking content
            attachments: File attachments

        Returns:
            Created Message
        """
        now = int(time.time())
        attachments = attachments or []

        with self._lock:
            cursor = self._conn.cursor()

            # Get next sequence number
            cursor.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE history_id = ?",
                (history_id,)
            )
            seq = cursor.fetchone()[0]

            # Get current revision
            cursor.execute("SELECT revision FROM chat_history WHERE id = ?", (history_id,))
            row = cursor.fetchone()
            revision = row["revision"] if row else 1

            # Insert message
            cursor.execute("""
                INSERT INTO messages (history_id, revision, role, content, thinking, created_at, seq)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (history_id, revision, role, content, thinking, now, seq))

            message_id = cursor.lastrowid

            # Insert attachments
            for att in attachments:
                cursor.execute("""
                    INSERT INTO message_files (message_id, file_id, file_name)
                    VALUES (?, ?, ?)
                """, (message_id, att.get("file_id", ""), att.get("file_name", "")))

            # Update history timestamp
            cursor.execute(
                "UPDATE chat_history SET updated_at = ? WHERE id = ?",
                (now, history_id)
            )

            self._conn.commit()

            return Message(
                id=message_id,
                history_id=history_id,
                revision=revision,
                role=role,
                content=content,
                thinking=thinking,
                created_at=now,
                seq=seq,
            )

    def get_messages(self, history_id: str) -> List[Message]:
        """Get all messages for a history entry."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT id, history_id, revision, role, content, thinking, created_at, seq
                FROM messages
                WHERE history_id = ?
                ORDER BY seq ASC
            """, (history_id,))

            messages = []
            for row in cursor.fetchall():
                messages.append(Message(
                    id=row["id"],
                    history_id=row["history_id"],
                    revision=row["revision"],
                    role=row["role"],
                    content=row["content"],
                    thinking=row["thinking"],
                    created_at=row["created_at"],
                    seq=row["seq"],
                ))

            return messages

    # === Response Operations ===

    def save_response(
        self,
        history_id: str,
        message_id: int,
        model: str,
        content: str,
        thinking: str = "",
        finish_reason: str = "",
        usage: Optional[Dict] = None,
    ) -> Response:
        """
        Save a response.

        Args:
            history_id: History ID
            message_id: Message ID
            model: Model used
            content: Response content
            thinking: Thinking content
            finish_reason: Finish reason
            usage: Usage statistics

        Returns:
            Saved Response
        """
        now = int(time.time())
        usage = usage or {}

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT INTO responses
                (history_id, message_id, model, content, thinking, finish_reason, usage, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (history_id, message_id, model, content, thinking, finish_reason, json.dumps(usage), now))

            resp_id = cursor.lastrowid

            # Update history timestamp and preview
            preview = content[:DEFAULT_PREVIEW_AT] if content else ""
            cursor.execute(
                "UPDATE chat_history SET updated_at = ?, preview = ? WHERE id = ?",
                (now, preview, history_id)
            )

            self._conn.commit()

            return Response(
                id=resp_id,
                history_id=history_id,
                message_id=message_id,
                model=model,
                content=content,
                thinking=thinking,
                finish_reason=finish_reason,
                usage=json.dumps(usage),
                created_at=now,
            )

    def get_responses(self, history_id: str) -> List[Response]:
        """Get all responses for a history entry."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT id, history_id, message_id, model, content, thinking,
                       finish_reason, usage, created_at
                FROM responses
                WHERE history_id = ?
                ORDER BY created_at ASC
            """, (history_id,))

            responses = []
            for row in cursor.fetchall():
                responses.append(Response(
                    id=row["id"],
                    history_id=row["history_id"],
                    message_id=row["message_id"],
                    model=row["model"],
                    content=row["content"],
                    thinking=row["thinking"],
                    finish_reason=row["finish_reason"],
                    usage=row["usage"],
                    created_at=row["created_at"],
                ))

            return responses

    # === Citation Operations ===

    def add_citation(
        self,
        message_id: int,
        index: int,
        url: str = "",
        title: str = "",
        text: str = "",
    ) -> Citation:
        """Add a citation to a message."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT INTO citations (message_id, idx, url, title, text)
                VALUES (?, ?, ?, ?, ?)
            """, (message_id, index, url, title, text))

            cit_id = cursor.lastrowid
            self._conn.commit()

            return Citation(
                id=cit_id,
                message_id=message_id,
                index=index,
                url=url,
                title=title,
                text=text,
            )

    def get_citations(self, message_id: int) -> List[Citation]:
        """Get citations for a message."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT id, message_id, idx, url, title, text
                FROM citations
                WHERE message_id = ?
                ORDER BY idx ASC
            """, (message_id,))

            citations = []
            for row in cursor.fetchall():
                citations.append(Citation(
                    id=row["id"],
                    message_id=row["message_id"],
                    index=row["idx"],
                    url=row["url"],
                    title=row["title"],
                    text=row["text"],
                ))

            return citations

    # === Attachment Operations ===

    def add_attachment(
        self,
        history_id: str,
        file_id: str,
        file_name: str,
        file_path: str,
        file_size: int = 0,
        mime_type: str = "",
    ) -> Attachment:
        """Add a file attachment to a history entry."""
        now = int(time.time())

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT INTO attachments
                (history_id, file_id, file_name, file_path, file_size, mime_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (history_id, file_id, file_name, file_path, file_size, mime_type, now))

            att_id = cursor.lastrowid
            self._conn.commit()

            return Attachment(
                id=att_id,
                history_id=history_id,
                file_id=file_id,
                file_name=file_name,
                file_path=file_path,
                file_size=file_size,
                mime_type=mime_type,
                created_at=now,
            )

    def get_attachments(self, history_id: str) -> List[Attachment]:
        """Get attachments for a history entry."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT id, history_id, file_id, file_name, file_path,
                       file_size, mime_type, created_at
                FROM attachments
                WHERE history_id = ?
                ORDER BY created_at ASC
            """, (history_id,))

            attachments = []
            for row in cursor.fetchall():
                attachments.append(Attachment(
                    id=row["id"],
                    history_id=row["history_id"],
                    file_id=row["file_id"],
                    file_name=row["file_name"],
                    file_path=row["file_path"],
                    file_size=row["file_size"],
                    mime_type=row["mime_type"],
                    created_at=row["created_at"],
                ))

            return attachments

    def get_attachment(self, attachment_id: int) -> Optional[Attachment]:
        """Get a single attachment by ID."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT id, history_id, file_id, file_name, file_path,
                       file_size, mime_type, created_at
                FROM attachments
                WHERE id = ?
            """, (attachment_id,))

            row = cursor.fetchone()
            if not row:
                return None

            return Attachment(
                id=row["id"],
                history_id=row["history_id"],
                file_id=row["file_id"],
                file_name=row["file_name"],
                file_path=row["file_path"],
                file_size=row["file_size"],
                mime_type=row["mime_type"],
                created_at=row["created_at"],
            )

    # === History Queries ===

    def get_history(
        self,
        history_id: str,
        include_messages: bool = True,
        include_citations: bool = True,
        include_attachments: bool = True,
    ) -> Dict[str, Any]:
        """
        Get complete history with related data.

        Args:
            history_id: History ID
            include_messages: Include messages
            include_citations: Include citations
            include_attachments: Include attachments

        Returns:
            Dict with history and related data
        """
        entry = self.get(history_id)
        if not entry:
            return {}

        result = {
            "entry": entry,
            "messages": [],
            "citations": {},
            "attachments": [],
        }

        if include_messages:
            messages = self.get_messages(history_id)
            result["messages"] = messages

            if include_citations:
                for msg in messages:
                    result["citations"][msg.id] = self.get_citations(msg.id)

        if include_attachments:
            result["attachments"] = self.get_attachments(history_id)

        return result

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._closed = True
            if self._conn:
                self._conn.close()
                self._conn = None


# Global store instance
_store: Optional[ChatHistoryStore] = None
_store_lock = threading.Lock()


def get_store() -> Optional[ChatHistoryStore]:
    """Get the global store instance."""
    return _store


def init_store(
    path: Optional[str] = None,
    cleanup_interval: int = 3600,
    ttl_seconds: int = 0,
) -> ChatHistoryStore:
    """Initialize the global store."""
    global _store
    with _store_lock:
        if _store:
            return _store
        _store = ChatHistoryStore(
            path=path,
            cleanup_interval=cleanup_interval,
            ttl_seconds=ttl_seconds,
        )
        return _store


def close_store() -> None:
    """Close the global store."""
    global _store
    with _store_lock:
        if _store:
            _store.close()
            _store = None
