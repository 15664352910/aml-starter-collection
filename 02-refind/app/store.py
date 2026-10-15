from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .models import AddRequest


@dataclass(frozen=True)
class MemoryChunk:
    id: str
    user_id: str
    session_id: str
    request_id: str
    sequence: int
    messages: tuple[dict, ...]
    timestamp_ms: int | None
    created_at: str

    @property
    def content(self) -> str:
        lines: list[str] = []
        for message in self.messages:
            timestamp = message.get("timestamp")
            prefix = _timestamp_label(timestamp) if timestamp is not None else ""
            role = str(message.get("role", "unknown")).upper()
            lines.append(f"{prefix}{role}: {message.get('content', '')}")
        return "\n".join(lines)


def _timestamp_label(timestamp_ms: int) -> str:
    try:
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return ""
    return f"[{dt.isoformat().replace('+00:00', 'Z')}] "


def _memory_id(user_id: str, request_id: str) -> str:
    digest = hashlib.sha256(f"{user_id}\0{request_id}".encode()).hexdigest()
    return f"mem_{digest[:24]}"


def _request_order(request_id: str) -> int | None:
    """Extract the platform chunk/turn ordinal when one is present."""
    matches = re.findall(
        r"(?:chunk|turn|message|msg)[-_:]?(\d+)", request_id, flags=re.IGNORECASE
    )
    if matches:
        return int(matches[-1])
    trailing = re.search(r"(?:^|[-_:])(\d+)$", request_id)
    return int(trailing.group(1)) if trailing else None


def _conversation_memory_id(
    user_id: str, session_id: str, source_positions: list[tuple[str, int]]
) -> str:
    canonical = json.dumps(source_positions, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(
        f"{user_id}\0{session_id}\0{canonical}".encode("utf-8")
    ).hexdigest()
    return f"mem_{digest[:24]}"


def _payload_hash(request: AddRequest) -> str:
    payload = request.model_dump(mode="json")
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ConflictingRequestError(ValueError):
    pass


class MemoryStore:
    """SQLite-backed, idempotent storage scoped strictly by ``user_id``."""

    def __init__(self, path: Path, retention_days: int = 30):
        self.path = path
        self.retention_days = retention_days
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ingestions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL UNIQUE,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ingestions_user
                    ON ingestions(user_id, sequence);

                CREATE TABLE IF NOT EXISTS messages (
                    ingestion_sequence INTEGER NOT NULL,
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp_ms INTEGER,
                    PRIMARY KEY (ingestion_sequence, message_index),
                    FOREIGN KEY (ingestion_sequence)
                        REFERENCES ingestions(sequence) ON DELETE CASCADE
                );
                """
            )
        self.delete_expired()

    def add(self, request: AddRequest) -> bool:
        """Synchronously persist all messages. Returns False for an exact retry."""
        digest = _payload_hash(request)
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._write_lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT payload_hash FROM ingestions WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
            if existing:
                if existing["payload_hash"] != digest:
                    raise ConflictingRequestError(
                        "request_id already exists with a different payload"
                    )
                return False

            cursor = connection.execute(
                """
                INSERT INTO ingestions
                    (request_id, user_id, session_id, memory_id, payload_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request.request_id,
                    request.user_id,
                    request.session_id,
                    _memory_id(request.user_id, request.request_id),
                    digest,
                    now,
                ),
            )
            sequence = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO messages
                    (ingestion_sequence, message_index, role, content, timestamp_ms)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (sequence, index, message.role, message.content, message.timestamp)
                    for index, message in enumerate(request.messages)
                ],
            )
        return True

    def list_chunks(self, user_id: str) -> list[MemoryChunk]:
        """Return paper-style conversation records for one exact user.

        The challenge sends one Add request per memory turn.  The original
        ReFind index, however, searches adjacent user/assistant turns as one
        conversation.  Reconstruct those records at Search time after ordering
        every source turn within its session by source timestamp and the stable
        chunk ordinal embedded in ``request_id``.  Database insertion order is
        only the final fallback, so concurrent Add delivery cannot silently
        scramble a timestamped conversation.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT i.sequence, i.request_id, i.user_id, i.session_id,
                       i.memory_id, i.created_at, m.message_index, m.role,
                       m.content, m.timestamp_ms
                FROM ingestions AS i
                JOIN messages AS m ON m.ingestion_sequence = i.sequence
                WHERE i.user_id = ?
                ORDER BY i.sequence, m.message_index
                """,
                (user_id,),
            ).fetchall()

        turns_by_session: dict[str, list[dict]] = {}
        first_sequence_by_session: dict[str, int] = {}
        for row in rows:
            sequence = int(row["sequence"])
            session_id = str(row["session_id"])
            request_id = str(row["request_id"])
            message_index = int(row["message_index"])
            timestamp = (
                int(row["timestamp_ms"])
                if row["timestamp_ms"] is not None
                else None
            )
            request_order = _request_order(request_id)
            first_sequence_by_session[session_id] = min(
                sequence, first_sequence_by_session.get(session_id, sequence)
            )
            turns_by_session.setdefault(session_id, []).append(
                {
                    "request_id": request_id,
                    "message_index": message_index,
                    "sequence": sequence,
                    "created_at": str(row["created_at"]),
                    "message": {
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": timestamp,
                    },
                    "timestamp": timestamp,
                    "request_order": request_order,
                }
            )

        chunks: list[MemoryChunk] = []
        ordered_sessions = sorted(
            turns_by_session,
            key=lambda session_id: (
                first_sequence_by_session[session_id],
                session_id,
            ),
        )
        for session_id in ordered_sessions:
            turns = turns_by_session[session_id]
            if all(item["timestamp"] is not None for item in turns):
                turns.sort(
                    key=lambda item: (
                        item["timestamp"],
                        item["request_order"]
                        if item["request_order"] is not None
                        else item["sequence"],
                        item["message_index"],
                        item["sequence"],
                    )
                )
            elif all(item["request_order"] is not None for item in turns):
                turns.sort(
                    key=lambda item: (
                        item["request_order"],
                        item["message_index"],
                        item["sequence"],
                    )
                )
            else:
                turns.sort(key=lambda item: (item["sequence"], item["message_index"]))
            for start in range(0, len(turns), 2):
                pair = turns[start : start + 2]
                source_positions = [
                    (item["request_id"], item["message_index"]) for item in pair
                ]
                timestamps = [
                    item["message"]["timestamp"]
                    for item in pair
                    if item["message"]["timestamp"] is not None
                ]
                conversation_id = _conversation_memory_id(
                    user_id, session_id, source_positions
                )
                chunks.append(
                    MemoryChunk(
                        id=conversation_id,
                        user_id=user_id,
                        session_id=session_id,
                        request_id=f"conversation:{conversation_id}",
                        sequence=len(chunks) + 1,
                        messages=tuple(item["message"] for item in pair),
                        timestamp_ms=min(timestamps) if timestamps else None,
                        created_at=min(item["created_at"] for item in pair),
                    )
                )
        return chunks

    def delete_expired(self) -> int:
        """Delete evaluation data older than the configured, at-most-30-day window."""
        modifier = f"-{self.retention_days} days"
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM ingestions WHERE julianday(created_at) < julianday('now', ?)",
                (modifier,),
            )
            return max(cursor.rowcount, 0)

    def count_users(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(DISTINCT user_id) AS count FROM ingestions"
            ).fetchone()
        return int(row["count"])
