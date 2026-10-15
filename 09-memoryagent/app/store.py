"""记忆存储：ChromaDB per-user 集合 + 写入去重 + request_id 幂等账本。

改编自 AML 2026-08 学术文本榜 #9 开源实现 MemoryAgent
（<https://github.com/llLAlisa/memory-agent-submission>），按 AML 契约改造。
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

COLLECTION_PREFIX = "user_mem_"


def _iso_from_ms(ms: Optional[int]) -> str:
    if ms is None:
        return datetime.now(timezone.utc).isoformat()
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


class RequestLedger:
    """request_id 幂等账本（sqlite3 标准库，无额外依赖）。

    平台会对同一 request_id 重试，账本保证重试安全返回、不重复写入。
    """

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS request_ids ("
            "request_id TEXT PRIMARY KEY, user_id TEXT, created_at TEXT)"
        )
        self._conn.commit()

    def seen(self, request_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM request_ids WHERE request_id = ?", (request_id,)
            ).fetchone()
            return row is not None

    def record(self, request_id: str, user_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO request_ids VALUES (?, ?, ?)",
                (request_id, user_id, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()


class MemoryStore:
    """每个 user_id 一个独立 Chroma 集合（哈希命名），天然隔离。"""

    def __init__(
        self,
        persist_path: str,
        embedder,
        ledger: RequestLedger,
        dedup_threshold: float = 0.95,
        abstention_threshold: float = 0.0,
    ) -> None:
        self._embedder = embedder
        self._ledger = ledger
        self.dedup_threshold = dedup_threshold
        self.abstention_threshold = abstention_threshold
        self._write_lock = threading.Lock()
        self._memories = 0
        self._client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    @staticmethod
    def _collection_name(user_id: str) -> str:
        digest = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:32]
        return f"{COLLECTION_PREFIX}{digest}"

    def _collection_for(self, user_id: str):
        return self._client.get_or_create_collection(
            name=self._collection_name(user_id),
            metadata={"hnsw:space": "cosine"},
        )

    def add_messages(
        self, request_id: str, user_id: str, messages: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """幂等摄入一批消息，返回 {stored, deduplicated, skipped}。"""
        if self._ledger.seen(request_id):
            return {"stored": 0, "deduplicated": 0, "skipped": len(messages)}

        stored = deduped = 0
        with self._write_lock:
            collection = self._collection_for(user_id)
            for msg in messages:
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                embedding = self._embedder.embed(content)
                if self._is_duplicate(collection, embedding):
                    deduped += 1
                    continue
                collection.add(
                    ids=[str(uuid.uuid4())],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[{
                        "user_id": user_id,
                        "role": msg.get("role", "user"),
                        "created_at": _iso_from_ms(msg.get("timestamp")),
                    }],
                )
                stored += 1
            self._memories += stored
        self._ledger.record(request_id, user_id)
        return {"stored": stored, "deduplicated": deduped, "skipped": 0}

    def search(self, user_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        collection = self._collection_for(user_id)
        embedding = self._embedder.embed(query)
        result = collection.query(
            query_embeddings=[embedding],
            n_results=max(top_k, 1),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        if not ids or top_k == 0:
            return []

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        items: List[Dict[str, Any]] = []
        for memory_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            score = round(max(0.0, min(1.0, 1.0 - float(distance))), 4)
            if score < self.abstention_threshold:  # 弃权：证据不足就返回空数组
                continue
            items.append({
                "id": memory_id,
                "content": document,
                "score": score,
                "created_at": metadata.get("created_at", ""),
            })
        return items[:top_k]

    def _is_duplicate(self, collection, embedding: List[float]) -> bool:
        if self.dedup_threshold >= 1.0:
            return False
        result = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        if not ids:
            return False
        distance = float((result.get("distances") or [[]])[0][0])
        return (1.0 - distance) > self.dedup_threshold

    def stats(self) -> Dict[str, int]:
        return {
            "users": len(self._client.list_collections()),
            "memories": self._memories,
        }
