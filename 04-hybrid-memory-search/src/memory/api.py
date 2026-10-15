from __future__ import annotations

import concurrent.futures
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from tqdm import tqdm

from .config import Settings, load_settings
from .embedding import EmbeddingClient
from .hybrid import SearchHit, hybrid_search
from .sessions import ConversationSample, load_samples_from_path, memory_id_for
from .store import MemoryRecord, MemoryStore


def _ms_to_iso(ms: int | float) -> str:
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_platform_messages(messages: Sequence[dict[str, Any]]) -> tuple[str, str]:
    """Build searchable text and created_at from a platform message chunk."""
    lines: list[str] = []
    timestamps: list[int | float] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = str(msg.get("content") or "")
        ts = msg.get("timestamp")
        if ts is not None:
            try:
                timestamps.append(float(ts))
                lines.append(f"[{_ms_to_iso(ts)}] {role}: {content}")
            except (TypeError, ValueError):
                lines.append(f"{role}: {content}")
        else:
            lines.append(f"{role}: {content}")
    text = "\n".join(lines).strip()
    created_at = _ms_to_iso(max(timestamps)) if timestamps else _now_iso()
    return text, created_at


@dataclass
class UpsertResult:
    id: str
    created: bool
    embedded: bool
    skipped: bool
    text_len: int = 0


@dataclass
class AddReport:
    data_path: str
    samples: int = 0
    sessions_total: int = 0
    created: int = 0
    skipped: int = 0
    embedded: int = 0
    buckets: list[str] = field(default_factory=list)


class MemorySystem:
    """Minimal memory API: upsert / embed / search (+ dataset ingest)."""

    def __init__(
        self,
        settings: Settings | None = None,
        store_dir: str | Path | None = None,
    ):
        self.settings = settings or load_settings()
        root = Path(store_dir) if store_dir is not None else self.settings.store_dir
        self.store = MemoryStore(root)
        self.embedder = EmbeddingClient(self.settings)

    # ---------- core APIs ----------

    def upsert(
        self,
        id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        *,
        bucket_id: str | None = None,
        embed: bool = True,
        skip_if_exists: bool = True,
        async_embed: bool = False,
    ) -> UpsertResult:
        """upsert(id, text, metadata) → optional embed(id).

        Idempotent: if id already exists and skip_if_exists=True, skip text rewrite.
        Two-phase: text first, then embedding attached to the same id.
        """
        meta = dict(metadata or {})
        bkt = bucket_id or meta.get("bucket_id")
        if not bkt:
            # derive bucket from id prefix before ":session_"
            if ":session_" in id:
                bkt = id.rsplit(":session_", 1)[0]
            else:
                bkt = meta.get("conversation_id") or "default"
        meta.setdefault("bucket_id", bkt)

        if skip_if_exists and self.store.exists(id, bkt):
            rec = self.store.get(id, bkt)
            embedded = False
            if embed and rec and not rec.has_embedding:
                if async_embed:
                    self.embed_async(id, bkt)
                    embedded = True
                else:
                    self.embed(id, bkt)
                    embedded = True
            return UpsertResult(
                id=id,
                created=False,
                embedded=embedded,
                skipped=True,
                text_len=len(rec.text) if rec else 0,
            )

        _, created = self.store.upsert_text(
            memory_id=id,
            bucket_id=bkt,
            text=text,
            metadata=meta,
            skip_if_exists=skip_if_exists,
        )

        embedded = False
        if embed:
            if async_embed:
                self.embed_async(id, bkt)
                embedded = True
            else:
                self.embed(id, bkt)
                embedded = True

        return UpsertResult(
            id=id,
            created=created,
            embedded=embedded,
            skipped=not created,
            text_len=len(text),
        )

    def embed(self, id: str, bucket_id: str | None = None) -> MemoryRecord:
        """Compute and cache embedding for an existing memory id."""
        bkt = bucket_id
        if bkt is None:
            if ":session_" in id:
                bkt = id.rsplit(":session_", 1)[0]
            else:
                raise ValueError("bucket_id is required when it cannot be inferred from id")
        rec = self.store.get(id, bkt)
        if rec is None:
            raise KeyError(f"memory not found: {id}")
        vector = self.embedder.embed(rec.text)
        return self.store.set_embedding(id, bkt, vector)

    def embed_async(
        self,
        id: str,
        bucket_id: str | None = None,
        executor: concurrent.futures.Executor | None = None,
    ) -> concurrent.futures.Future:
        own = executor is None
        pool = executor or concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(self.embed, id, bucket_id)
        if own:
            fut.add_done_callback(lambda _: pool.shutdown(wait=False))
        return fut

    def search(
        self,
        bucket_id: str,
        query: str,
        top_k: int = 5,
        alpha: float | None = None,
    ) -> list[SearchHit]:
        """Search within one bucket only: hybrid BM25 + dense."""
        a = self.settings.hybrid_alpha if alpha is None else float(alpha)
        records = self.store.list_bucket(bucket_id)
        embeddings = self.store.get_bucket_embeddings(bucket_id)
        return hybrid_search(
            records=records,
            embeddings=embeddings,
            query=query,
            embedder=self.embedder,
            top_k=top_k,
            alpha=a,
        )

    # ---------- platform eval Add / Search ----------

    def add_platform_chunk(
        self,
        *,
        request_id: str,
        messages: Sequence[dict[str, Any]],
        user_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Store one platform memory chunk under user_id; sync-embed before return."""
        if not messages:
            raise ValueError("messages must be non-empty")

        text, created_at = format_platform_messages(messages)
        if not text:
            raise ValueError("messages content is empty")

        meta = {
            "bucket_id": user_id,
            "user_id": user_id,
            "session_id": session_id,
            "request_id": request_id,
            "messages": [dict(m) for m in messages],
            "created_at": created_at,
        }
        # Idempotent: same request_id already stored → ensure embedding then succeed.
        self.upsert(
            id=request_id,
            text=text,
            metadata=meta,
            bucket_id=user_id,
            embed=True,
            skip_if_exists=True,
            async_embed=False,
        )
        return {
            "success": True,
            "request_id": request_id,
            "user_id": user_id,
            "session_id": session_id,
        }

    def platform_search(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int,
        options: Sequence[str] | None = None,
        alpha: float | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search scoped by user_id; return platform data[] items."""
        search_query = query
        if options:
            search_query = query + "\n" + "\n".join(str(o) for o in options)
        hits = self.search(
            bucket_id=user_id,
            query=search_query,
            top_k=top_k,
            alpha=alpha,
        )
        return self.to_platform_hits(hits)

    def to_platform_hits(self, hits: list[SearchHit]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for h in hits:
            created = (h.metadata or {}).get("created_at") or _now_iso()
            out.append(
                {
                    "id": h.id,
                    "content": h.text,
                    "score": float(h.score),
                    "created_at": created,
                }
            )
        return out

    # ---------- dataset ingest ----------

    def add_from_data(
        self,
        data_path: str | Path,
        *,
        bucket_id: str | None = None,
        embed: bool = True,
        skip_if_exists: bool = True,
        show_progress: bool = True,
        on_sample: Callable[[ConversationSample], None] | None = None,
    ) -> AddReport:
        """Ingest sessions from a data path (file or directory).

        data_path is required and never hardcoded — e.g. ``data`` or
        ``data/LoCoMo_refined/.../locomo_refined.json``.
        """
        samples = load_samples_from_path(data_path)
        report = AddReport(data_path=str(data_path), samples=len(samples))
        buckets: set[str] = set()
        multi = len(samples) > 1

        # flatten work items
        jobs: list[tuple[str, str, str, dict[str, Any]]] = []
        for sample in samples:
            # One conversation sample = one retrieval bucket.
            # If caller forces a prefix on multi-sample data, keep samples isolated:
            #   {prefix}:{sample_id}
            if bucket_id is None:
                bkt = sample.bucket_id
            elif multi and sample.sample_id and sample.sample_id != bucket_id:
                bkt = f"{bucket_id}:{sample.sample_id}"
            else:
                bkt = bucket_id
            buckets.add(bkt)
            if on_sample:
                on_sample(sample)
            for sess in sample.sessions:
                mid = memory_id_for(bkt, sess.session_index)
                meta = {
                    "bucket_id": bkt,
                    "sample_id": sample.sample_id,
                    "session_index": sess.session_index,
                    "date_time": sess.date_time,
                    "source_path": sample.source_path,
                }
                jobs.append((mid, bkt, sess.text, meta))

        report.sessions_total = len(jobs)
        report.buckets = sorted(buckets)

        iterator = tqdm(jobs, desc="add sessions") if show_progress else jobs
        pending_embed: list[tuple[str, str]] = []

        for mid, bkt, text, meta in iterator:
            if skip_if_exists and self.store.exists(mid, bkt):
                report.skipped += 1
                rec = self.store.get(mid, bkt)
                if embed and rec and not rec.has_embedding:
                    pending_embed.append((mid, bkt))
                continue

            self.store.upsert_text(
                memory_id=mid,
                bucket_id=bkt,
                text=text,
                metadata=meta,
                skip_if_exists=False,
            )
            report.created += 1
            if embed:
                pending_embed.append((mid, bkt))

        if embed and pending_embed:
            emb_iter = (
                tqdm(pending_embed, desc="embed sessions") if show_progress else pending_embed
            )
            for mid, bkt in emb_iter:
                self.embed(mid, bkt)
                report.embedded += 1

        return report

    def search_texts(
        self,
        bucket_id: str,
        query: str,
        top_k: int = 5,
        alpha: float | None = None,
    ) -> list[str]:
        """Convenience: return only ranked memory texts for downstream generation."""
        return [h.text for h in self.search(bucket_id, query, top_k=top_k, alpha=alpha)]

    def to_dict_hits(self, hits: list[SearchHit]) -> list[dict[str, Any]]:
        return [asdict(h) for h in hits]
