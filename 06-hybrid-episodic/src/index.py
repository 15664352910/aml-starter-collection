"""Hybrid retrieval: incremental BM25 + dense cosine, fused with RRF.

Why hybrid rather than dense alone: these benchmarks turn on names, titles,
places and labels ("use 'Rob' not 'a colleague'"), and lexical matching is what
reliably pins those. Why RRF rather than score blending: BM25 and cosine live on
incomparable scales, and rank fusion needs no per-corpus normalisation.

Everything is per-user. ``user_id`` is a hard partition — the contract forbids
returning memories across it.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict

import numpy as np

from . import config

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[一-鿿]")


def tokenize(text: str) -> list[str]:
    """Lowercase word/CJK-character tokens. CJK is split per character so that
    Chinese content is still matchable without a segmenter dependency."""
    return _TOKEN.findall(text.lower())


class UserIndex:
    """Append-only hybrid index for a single ``user_id``.

    Add is synchronous per the contract, so writes must be visible to the very
    next Search. Every mutation happens under ``_lock`` and the derived BM25
    statistics are updated in the same critical section — there is no deferred
    build step that a subsequent read could race.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.ids: list[str] = []
        self.contents: list[str] = []
        self.created_at: list[str | None] = []
        self.timestamps: list[int | None] = []
        self._tokens: list[list[str]] = []
        self._term_freq: list[Counter[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._length_sum: int = 0
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._vectors: list[np.ndarray] = []
        # request_id bookkeeping for idempotent Add. A retry can arrive while the
        # original is still being embedded — the platform's timeout is its own, not
        # ours — so a plain "seen" set checked before the work and set after it
        # leaves a window in which both copies get indexed. Ownership is claimed
        # atomically up front instead, and late arrivals wait on the owner.
        self._completed: set[str] = set()
        self._inflight: dict[str, threading.Event] = {}

    # ── idempotency ─────────────────────────────────────────────────────────

    def begin(self, request_id: str) -> tuple[bool, threading.Event | None]:
        """Claim a request_id.

        Returns ``(is_owner, wait_event)``:
          * ``(True, event)``  — caller owns the write and must call :meth:`finish`.
          * ``(False, None)``  — already completed; nothing to do.
          * ``(False, event)`` — another caller is mid-write; wait on ``event``.
        """
        with self._lock:
            if request_id in self._completed:
                return False, None
            existing = self._inflight.get(request_id)
            if existing is not None:
                return False, existing
            event = threading.Event()
            self._inflight[request_id] = event
            return True, event

    def finish(self, request_id: str, *, success: bool) -> None:
        """Release ownership, marking the request completed only if it succeeded.

        On failure the id is left un-completed so the platform's retry redoes the
        work instead of being skipped as a duplicate.
        """
        with self._lock:
            event = self._inflight.pop(request_id, None)
            if success:
                self._completed.add(request_id)
        if event is not None:
            event.set()

    def is_completed(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._completed

    # ── writes ──────────────────────────────────────────────────────────────

    def add(
        self,
        memory_id: str,
        content: str,
        created_at: str | None,
        timestamp_ms: int | None,
        vector: np.ndarray | None,
    ) -> None:
        tokens = tokenize(content)
        with self._lock:
            position = len(self.ids)
            self.ids.append(memory_id)
            self.contents.append(content)
            self.created_at.append(created_at)
            self.timestamps.append(timestamp_ms)
            self._tokens.append(tokens)
            frequencies = Counter(tokens)
            self._term_freq.append(frequencies)
            for term in frequencies:
                self._doc_freq[term] += 1
                self._postings[term].append(position)
            self._length_sum += len(tokens)
            self._vectors.append(
                vector if vector is not None else np.zeros(0, dtype=np.float32)
            )

    # ── reads ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self.ids)

    def bm25(self, query: str, depth: int) -> list[int]:
        """Return document positions ranked by BM25, best first."""
        with self._lock:
            total = len(self.ids)
            if not total:
                return []
            avg_len = (self._length_sum / total) or 1.0
            scores: dict[int, float] = defaultdict(float)
            for term in set(tokenize(query)):
                postings = self._postings.get(term)
                if not postings:
                    continue
                df = self._doc_freq[term]
                idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
                for position in postings:
                    tf = self._term_freq[position][term]
                    doc_len = len(self._tokens[position]) or 1
                    denominator = tf + config.BM25_K1 * (
                        1.0 - config.BM25_B + config.BM25_B * doc_len / avg_len
                    )
                    scores[position] += idf * tf * (config.BM25_K1 + 1.0) / denominator
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [position for position, _ in ranked[:depth]]

    def dense(self, query_vector: np.ndarray | None, depth: int) -> list[int]:
        """Return document positions ranked by cosine similarity, best first."""
        if query_vector is None or not query_vector.size:
            return []
        with self._lock:
            if not self._vectors:
                return []
            usable = [
                (position, vector)
                for position, vector in enumerate(self._vectors)
                if vector.size == query_vector.size
            ]
            if not usable:
                return []
            positions = np.fromiter((p for p, _ in usable), dtype=np.int64, count=len(usable))
            matrix = np.vstack([v for _, v in usable])
        # Vectors are stored L2-normalised, so a dot product is the cosine.
        similarities = matrix @ query_vector
        if similarities.size > depth:
            top = np.argpartition(-similarities, depth - 1)[:depth]
            top = top[np.argsort(-similarities[top])]
        else:
            top = np.argsort(-similarities)
        return [int(positions[i]) for i in top]


def rrf(ranked_lists: list[tuple[list[int], float]], damping: float) -> dict[int, float]:
    """Weighted reciprocal rank fusion: score = Σ weight / (damping + rank)."""
    fused: dict[int, float] = defaultdict(float)
    for positions, weight in ranked_lists:
        if weight <= 0.0:
            continue
        for rank, position in enumerate(positions, start=1):
            fused[position] += weight / (damping + rank)
    return fused


def apply_recency(
    fused: dict[int, float], timestamps: list[int | None], weight: float
) -> dict[int, float]:
    """Nudge scores toward newer memories.

    The platform's answer prompt resolves conflicts by preferring the most recent
    supported memory, so a mild recency preference aligns retrieval with how the
    answer is actually produced. Kept mild on purpose: over-weighting recency
    destroys explicit-fact recall on older turns.
    """
    if weight <= 0.0 or not fused:
        return fused
    known = [(p, timestamps[p]) for p in fused if timestamps[p] is not None]
    if len(known) < 2:
        return fused
    values = [t for _, t in known]
    low, high = min(values), max(values)
    if high == low:
        return fused
    span = float(high - low)
    best = max(fused.values()) or 1.0
    for position, stamp in known:
        fused[position] += weight * best * ((stamp - low) / span)
    return fused
