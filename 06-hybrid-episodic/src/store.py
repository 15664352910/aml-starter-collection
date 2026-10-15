"""Per-user memory store: write path, retrieval path, cross-channel fusion.

``user_id`` partitions everything. The contract forbids returning a memory
outside the requested ``user_id``, and ``session_id`` is explicitly *not* a
retrieval filter — it is kept only as provenance.

Two channels are maintained per user and fused at read time:

  * **extracted** — mem0's LLM-distilled facts (third-party, see README).
  * **raw** — our own BM25/dense index over the original turns, with each unit's
    timestamp and role baked into the text.

The second exists because extraction is lossy in exactly the ways this benchmark
scores. The platform's answer prompt asks the model to treat memories as
"episodic raw observations", to preserve specific names and titles verbatim, and
to resolve relative dates *from the memory timestamp* — all of which a condensed
fact statement tends to drop. Fusing recovers the surface form without giving up
the semantic abstraction.
"""

from __future__ import annotations

import re
import threading

from . import chunker, config
from .embedder import EMBEDDER
from .index import UserIndex, apply_recency, rrf
from .mem0_backend import MEM0, Mem0Unavailable
from .models import AddRequest, MemoryItem, SearchRequest

# Channel tags for fused candidate keys.
RAW = "raw"
EXTRACTED = "extracted"

_NORMALISE = re.compile(r"[^a-z0-9]+")


def _dedup_key(text: str) -> str:
    return _NORMALISE.sub(" ", text.lower()).strip()


class MemoryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._users: dict[str, UserIndex] = {}
        self._counters: dict[str, int] = {}

    def _index_for(self, user_id: str) -> UserIndex:
        with self._lock:
            index = self._users.get(user_id)
            if index is None:
                index = UserIndex()
                self._users[user_id] = index
                self._counters[user_id] = 0
            return index

    def _next_ids(self, user_id: str, count: int) -> list[str]:
        with self._lock:
            start = self._counters.get(user_id, 0)
            self._counters[user_id] = start + count
        return [f"mem_{n}" for n in range(start, start + count)]

    # ── write ───────────────────────────────────────────────────────────────

    def add(self, request: AddRequest) -> int:
        """Index one Add payload. Returns the number of units written.

        Synchronous by contract: when this returns, the memory is queryable.
        Raises :class:`Mem0Unavailable` when the optional extraction channel is
        enabled and fails, so the caller can answer with a retryable status
        instead of reporting success for a write that did not happen.
        """
        index = self._index_for(request.user_id)

        # Ownership is claimed before any work, not after. Add is retried on
        # 408/409/425/429/5xx and on transport errors, and the platform's timeout
        # is independent of ours — so a retry can arrive while the original write
        # is still embedding. Checking a "seen" set before the work and setting it
        # after would let both copies through that window.
        is_owner, event = index.begin(request.request_id)
        if not is_owner:
            if event is None:
                return 0  # Already completed; nothing to do.
            # Another caller is mid-write. Wait for it, so that returning 200 still
            # means the memory is retrievable.
            if not event.wait(timeout=config.DUPLICATE_WAIT_SECONDS):
                raise Mem0Unavailable(
                    f"concurrent write for {request.request_id} did not settle in time"
                )
            if not index.is_completed(request.request_id):
                # The owner failed. Surface it so the platform retries rather than
                # accepting a success for a write nobody completed.
                raise Mem0Unavailable(
                    f"concurrent write for {request.request_id} failed"
                )
            return 0

        try:
            units = chunker.build_units(request.messages)
            if not units:
                index.finish(request.request_id, success=True)
                return 0

            # Extraction first when enabled: a failure must leave the request
            # un-completed so the platform's retry actually redoes the work.
            if MEM0.enabled:
                stamps = [m.timestamp for m in request.messages if m.timestamp is not None]
                try:
                    MEM0.add(
                        messages=[
                            {"role": m.role, "content": m.content}
                            for m in request.messages
                            if m.content and m.content.strip()
                        ],
                        user_id=request.user_id,
                        timestamp=min(stamps) if stamps else None,
                    )
                except Mem0Unavailable:
                    if config.MEM0_FAIL_ADD_ON_ERROR:
                        raise
                    # Explicitly configured to tolerate extraction loss.

            rendered = [chunker.render(role, ts, text) for role, ts, text in units]
            vectors = EMBEDDER.encode(rendered, is_query=False)
            memory_ids = self._next_ids(request.user_id, len(units))

            for position, ((_, timestamp_ms, _), content) in enumerate(zip(units, rendered)):
                index.add(
                    memory_id=memory_ids[position],
                    content=content,
                    created_at=chunker.iso_timestamp(timestamp_ms),
                    timestamp_ms=timestamp_ms,
                    vector=vectors[position] if vectors is not None else None,
                )
        except BaseException:
            index.finish(request.request_id, success=False)
            raise

        index.finish(request.request_id, success=True)
        return len(units)

    # ── read ────────────────────────────────────────────────────────────────

    def search(self, request: SearchRequest) -> list[MemoryItem]:
        limit = max(0, min(request.top_k, config.MAX_RETURN))
        if limit == 0:
            return []

        index = self._users.get(request.user_id)
        depth = max(config.CANDIDATE_DEPTH, limit)

        # Query variants. Option text carries the discriminating content on
        # multiple-choice items, but appending it to the primary query would let a
        # hundred words of options swamp a ten-word question in the term
        # statistics — so it gets its own weighted channel instead.
        queries: list[tuple[str, float]] = [(request.query, 1.0)]
        if request.options:
            queries.append(
                (request.query + " " + " ".join(request.options), config.OPTION_CHANNEL_WEIGHT)
            )

        channels: list[tuple[list[tuple[str, int]], float]] = []

        if index is not None and len(index):
            query_vectors = EMBEDDER.encode([q for q, _ in queries], is_query=True)
            for position, (text, weight) in enumerate(queries):
                channels.append(
                    ([(RAW, p) for p in index.bm25(text, depth)], weight)
                )
                if query_vectors is not None:
                    channels.append(
                        ([(RAW, p) for p in index.dense(query_vectors[position], depth)], weight)
                    )

        extracted: list[dict] = []
        if MEM0.enabled:
            extracted = MEM0.search(request.query, request.user_id, depth)
            if extracted:
                channels.append(
                    (
                        [(EXTRACTED, n) for n in range(len(extracted))],
                        config.MEM0_CHANNEL_WEIGHT,
                    )
                )

        if not channels:
            return []

        fused = rrf(channels, config.RRF_K)
        if not fused:
            return []

        if index is not None and len(index):
            raw_only = {key[1]: score for key, score in fused.items() if key[0] == RAW}
            nudged = apply_recency(raw_only, index.timestamps, config.RECENCY_WEIGHT)
            for position, score in nudged.items():
                fused[(RAW, position)] = score

        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))

        items: list[MemoryItem] = []
        seen: set[str] = set()
        for (channel, position), score in ordered:
            if len(items) >= limit:
                break
            if channel == RAW:
                content = index.contents[position]  # type: ignore[union-attr]
                identifier = index.ids[position]  # type: ignore[union-attr]
                created = index.created_at[position]  # type: ignore[union-attr]
            else:
                row = extracted[position]
                content = row["content"]
                identifier = row["id"]
                created = row["created_at"]
            key = _dedup_key(content)
            # An extracted fact often restates a raw turn almost verbatim; a
            # duplicate wastes a top_k slot and reads to the answer model as
            # independent corroboration of the same claim.
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(
                MemoryItem(
                    id=identifier,
                    content=content,
                    score=round(float(score), 6),
                    created_at=created,
                )
            )
        return items

    # ── introspection (local debugging only; not part of the contract) ──────

    def stats(self) -> dict[str, object]:
        with self._lock:
            users = list(self._users.items())
        return {
            "users": len(users),
            "memories": sum(len(index) for _, index in users),
            "extraction": "on" if MEM0.enabled else "off",
        }


STORE = MemoryStore()
