from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .schemas import AddRequest
from .text import make_search_text


@dataclass(frozen=True)
class MemoryRow:
    id: str
    row_id: int
    content: str
    role: str
    session_id: str
    request_id: str
    ordinal: int
    occurred_at: int | None
    created_at: str
    search_text: str
    fts_rank: float

    def source_order(self) -> tuple[int, int, int, int]:
        chunk_match = re.search(r"(?:^|[-_:])chunk[-_:]?(\d+)(?:$|[-_:])", self.request_id)
        chunk_index = int(chunk_match.group(1)) if chunk_match else 1_000_000_000
        timestamp = self.occurred_at if self.occurred_at is not None else 0
        # Chunk/ordinal repairs ordering when concurrent Add calls carry a
        # shared or missing timestamp; row_id remains the deterministic fallback.
        return (timestamp, chunk_index, self.ordinal, self.row_id)


class MemoryDatabase:
    def __init__(self, path: Path):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    occurred_at INTEGER,
                    created_at TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    UNIQUE(request_id, ordinal),
                    FOREIGN KEY(request_id) REFERENCES requests(request_id)
                );

                CREATE INDEX IF NOT EXISTS idx_memories_user_time
                    ON memories(user_id, occurred_at, ordinal);

                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts_v3 USING fts5(
                    id UNINDEXED,
                    user_id UNINDEXED,
                    content,
                    search_text,
                    tokenize='porter unicode61 remove_diacritics 2'
                );
                """
            )
            indexed = connection.execute("SELECT count(*) FROM memory_fts_v3").fetchone()[0]
            stored = connection.execute("SELECT count(*) FROM memories").fetchone()[0]
            if indexed == 0 and stored:
                connection.execute(
                    """
                    INSERT INTO memory_fts_v3(id, user_id, content, search_text)
                    SELECT id, user_id, content, search_text FROM memories
                    """
                )

    def add(self, request: AddRequest, annotations: list[str]) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT user_id, session_id FROM requests WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
            if existing:
                if existing["user_id"] != request.user_id or existing["session_id"] != request.session_id:
                    raise ValueError("request_id was already used with a different scope")
                connection.execute("COMMIT")
                return

            connection.execute(
                "INSERT INTO requests(request_id, user_id, session_id, created_at) VALUES (?, ?, ?, ?)",
                (request.request_id, request.user_id, request.session_id, created_at),
            )
            for ordinal, message in enumerate(request.messages):
                digest = hashlib.sha256(f"{request.request_id}:{ordinal}".encode()).hexdigest()[:24]
                memory_id = f"mem_{digest}"
                annotation = annotations[ordinal] if ordinal < len(annotations) else ""
                search_text = make_search_text(message.content, annotation)
                connection.execute(
                    """
                    INSERT INTO memories(
                        id, user_id, session_id, request_id, ordinal, role, content,
                        occurred_at, created_at, search_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        request.user_id,
                        request.session_id,
                        request.request_id,
                        ordinal,
                        message.role,
                        message.content,
                        message.timestamp,
                        created_at,
                        search_text,
                    ),
                )
                connection.execute(
                    "INSERT INTO memory_fts_v3(id, user_id, content, search_text) VALUES (?, ?, ?, ?)",
                    (memory_id, request.user_id, message.content, search_text),
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def search(self, user_id: str, query: str, limit: int) -> list[MemoryRow]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.id, m.rowid AS row_id, m.content, m.role, m.session_id,
                       m.request_id, m.ordinal, m.occurred_at, m.created_at,
                       m.search_text, bm25(memory_fts_v3, 0.0, 0.0, 1.0, 2.0) AS fts_rank
                FROM memory_fts_v3
                JOIN memories AS m ON m.id = memory_fts_v3.id
                WHERE memory_fts_v3 MATCH ? AND memory_fts_v3.user_id = ?
                ORDER BY fts_rank ASC
                LIMIT ?
                """,
                (query, user_id, limit),
            ).fetchall()
        return [MemoryRow(**dict(row)) for row in rows]

    def recent(self, user_id: str, limit: int) -> list[MemoryRow]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, rowid AS row_id, content, role, session_id, request_id,
                       ordinal, occurred_at, created_at, search_text, 1000.0 AS fts_rank
                FROM memories
                WHERE user_id = ?
                ORDER BY COALESCE(occurred_at, 0) DESC, rowid DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [MemoryRow(**dict(row)) for row in rows]

    def neighbors(
        self,
        user_id: str,
        seed_ids: list[str],
        radius: int = 2,
    ) -> list[tuple[MemoryRow, int, int]]:
        """Return source turns around ranked seed rows without crossing a session."""
        if not seed_ids or radius < 1:
            return []
        placeholders = ",".join("?" for _ in seed_ids)
        with self._connect() as connection:
            seed_rows = connection.execute(
                f"""
                SELECT id, session_id
                FROM memories
                WHERE user_id = ? AND id IN ({placeholders})
                """,
                (user_id, *seed_ids),
            ).fetchall()
            seed_sessions = {row["session_id"] for row in seed_rows}
            if not seed_sessions:
                return []
            session_placeholders = ",".join("?" for _ in seed_sessions)
            rows = connection.execute(
                f"""
                SELECT id, rowid AS row_id, content, role, session_id, request_id,
                       ordinal, occurred_at, created_at, search_text, 1000.0 AS fts_rank
                FROM memories
                WHERE user_id = ? AND session_id IN ({session_placeholders})
                ORDER BY session_id, COALESCE(occurred_at, 0), rowid
                """,
                (user_id, *sorted(seed_sessions)),
            ).fetchall()

        by_session: dict[str, list[MemoryRow]] = {}
        for raw in rows:
            row = MemoryRow(**dict(raw))
            by_session.setdefault(row.session_id, []).append(row)
        for session_rows in by_session.values():
            session_rows.sort(key=MemoryRow.source_order)
        positions = {
            row.id: (session_id, index)
            for session_id, session_rows in by_session.items()
            for index, row in enumerate(session_rows)
        }
        output: dict[str, tuple[MemoryRow, int, int]] = {}
        for seed_rank, seed_id in enumerate(seed_ids):
            location = positions.get(seed_id)
            if location is None:
                continue
            session_id, seed_position = location
            session_rows = by_session[session_id]
            start = max(0, seed_position - radius)
            stop = min(len(session_rows), seed_position + radius + 1)
            for position in range(start, stop):
                distance = abs(position - seed_position)
                row = session_rows[position]
                previous = output.get(row.id)
                candidate = (row, distance, seed_rank)
                if previous is None or (distance, seed_rank) < (previous[1], previous[2]):
                    output[row.id] = candidate
        return list(output.values())
