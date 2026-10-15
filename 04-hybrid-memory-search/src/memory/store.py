from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass
class MemoryRecord:
    id: str
    bucket_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    has_embedding: bool = False


class MemoryStore:
    """Bucketed local store: text JSON + embedding matrix (same memory id)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._global = threading.Lock()

    def _bucket_dir(self, bucket_id: str) -> Path:
        # Windows forbids ':' in paths; keep a reversible-ish safe name.
        safe = (
            bucket_id.replace("\\", "_")
            .replace("/", "_")
            .replace(":", "__")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
        )
        path = self.root / safe
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _bucket_meta_path(self, bucket_id: str) -> Path:
        return self._bucket_dir(bucket_id) / "bucket.json"

    def _ensure_bucket_meta(self, bucket_id: str) -> None:
        path = self._bucket_meta_path(bucket_id)
        if not path.exists():
            path.write_text(
                json.dumps({"bucket_id": bucket_id}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _lock(self, bucket_id: str) -> threading.Lock:
        with self._global:
            if bucket_id not in self._locks:
                self._locks[bucket_id] = threading.Lock()
            return self._locks[bucket_id]

    def _records_path(self, bucket_id: str) -> Path:
        return self._bucket_dir(bucket_id) / "records.json"

    def _embeddings_path(self, bucket_id: str) -> Path:
        return self._bucket_dir(bucket_id) / "embeddings.npz"

    def _embedding_ids_path(self, bucket_id: str) -> Path:
        return self._bucket_dir(bucket_id) / "embedding_ids.json"

    def _load_records(self, bucket_id: str) -> dict[str, MemoryRecord]:
        path = self._records_path(bucket_id)
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, MemoryRecord] = {}
        for mid, obj in raw.get("memories", {}).items():
            out[mid] = MemoryRecord(
                id=obj["id"],
                bucket_id=obj["bucket_id"],
                text=obj["text"],
                metadata=obj.get("metadata") or {},
                has_embedding=bool(obj.get("has_embedding", False)),
            )
        return out

    def _save_records(self, bucket_id: str, records: dict[str, MemoryRecord]) -> None:
        path = self._records_path(bucket_id)
        payload = {
            "bucket_id": bucket_id,
            "memories": {mid: asdict(rec) for mid, rec in records.items()},
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _load_embeddings(self, bucket_id: str) -> tuple[list[str], np.ndarray]:
        vec_path = self._embeddings_path(bucket_id)
        ids_path = self._embedding_ids_path(bucket_id)
        if not vec_path.exists() or not ids_path.exists():
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [str(x) for x in json.loads(ids_path.read_text(encoding="utf-8"))]
        data = np.load(vec_path, allow_pickle=False)
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        if len(ids) != len(vectors):
            raise ValueError(
                f"embedding id/vector size mismatch in bucket={bucket_id}: "
                f"{len(ids)} ids vs {len(vectors)} vectors"
            )
        return ids, vectors

    def _save_embeddings(self, bucket_id: str, ids: list[str], vectors: np.ndarray) -> None:
        vec_path = self._embeddings_path(bucket_id)
        ids_path = self._embedding_ids_path(bucket_id)
        tmp_vec = vec_path.with_suffix(".tmp.npz")
        tmp_ids = ids_path.with_suffix(".tmp.json")
        np.savez_compressed(tmp_vec, vectors=vectors.astype(np.float32))
        tmp_ids.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
        tmp_vec.replace(vec_path)
        tmp_ids.replace(ids_path)

    def exists(self, memory_id: str, bucket_id: str) -> bool:
        with self._lock(bucket_id):
            return memory_id in self._load_records(bucket_id)

    def get(self, memory_id: str, bucket_id: str) -> MemoryRecord | None:
        with self._lock(bucket_id):
            return self._load_records(bucket_id).get(memory_id)

    def list_bucket(self, bucket_id: str) -> list[MemoryRecord]:
        with self._lock(bucket_id):
            records = self._load_records(bucket_id)
            return list(records.values())

    def list_buckets(self) -> list[str]:
        if not self.root.exists():
            return []
        names: list[str] = []
        for p in self.root.iterdir():
            if not p.is_dir():
                continue
            meta = p / "bucket.json"
            if meta.exists():
                try:
                    names.append(json.loads(meta.read_text(encoding="utf-8"))["bucket_id"])
                    continue
                except Exception:
                    pass
            # fallback for older stores
            names.append(p.name.replace("__", ":"))
        return sorted(names)

    def upsert_text(
        self,
        memory_id: str,
        bucket_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        *,
        skip_if_exists: bool = True,
    ) -> tuple[MemoryRecord, bool]:
        """Phase-1 write: store raw text. Returns (record, created)."""
        with self._lock(bucket_id):
            self._ensure_bucket_meta(bucket_id)
            records = self._load_records(bucket_id)
            if memory_id in records and skip_if_exists:
                return records[memory_id], False

            prev = records.get(memory_id)
            rec = MemoryRecord(
                id=memory_id,
                bucket_id=bucket_id,
                text=text,
                metadata=metadata or {},
                has_embedding=bool(prev.has_embedding) if prev else False,
            )
            records[memory_id] = rec
            self._save_records(bucket_id, records)
            return rec, True

    def set_embedding(self, memory_id: str, bucket_id: str, vector: np.ndarray) -> MemoryRecord:
        """Phase-2 write: attach embedding to the same memory id."""
        with self._lock(bucket_id):
            records = self._load_records(bucket_id)
            if memory_id not in records:
                raise KeyError(f"memory not found: {memory_id}")

            ids, matrix = self._load_embeddings(bucket_id)
            vec = np.asarray(vector, dtype=np.float32).reshape(-1)
            if memory_id in ids:
                idx = ids.index(memory_id)
                if matrix.size == 0:
                    matrix = vec.reshape(1, -1)
                else:
                    if matrix.shape[1] != vec.shape[0]:
                        raise ValueError(
                            f"embedding dim mismatch: store={matrix.shape[1]} new={vec.shape[0]}"
                        )
                    matrix[idx] = vec
            else:
                ids.append(memory_id)
                if matrix.size == 0:
                    matrix = vec.reshape(1, -1)
                else:
                    if matrix.shape[1] != vec.shape[0]:
                        raise ValueError(
                            f"embedding dim mismatch: store={matrix.shape[1]} new={vec.shape[0]}"
                        )
                    matrix = np.vstack([matrix, vec.reshape(1, -1)])

            self._save_embeddings(bucket_id, ids, matrix)
            records[memory_id].has_embedding = True
            self._save_records(bucket_id, records)
            return records[memory_id]

    def get_bucket_embeddings(self, bucket_id: str) -> dict[str, np.ndarray]:
        with self._lock(bucket_id):
            ids, matrix = self._load_embeddings(bucket_id)
            if not ids:
                return {}
            return {mid: matrix[i] for i, mid in enumerate(ids)}

    def iter_missing_embeddings(self, bucket_id: str) -> Iterator[MemoryRecord]:
        with self._lock(bucket_id):
            records = self._load_records(bucket_id)
            for rec in records.values():
                if not rec.has_embedding:
                    yield rec
