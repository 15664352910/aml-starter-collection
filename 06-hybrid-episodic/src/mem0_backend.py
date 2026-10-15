"""Mem0 memory core, configured for the Academic division's model constraint.

Disclosure (see README): mem0 is third-party, Apache-2.0, used as an unmodified
library dependency — https://github.com/mem0ai/mem0. Its contribution here is
LLM-based fact extraction on write and semantic recall on read. Everything in
``src/index.py`` and the fusion in ``src/store.py`` is ours.

Model policy: the Academic board requires the model used inside Add/Search to be
``gpt-4o-mini``, so the extraction LLM is pinned there and the temperature is 0
for reproducibility — the platform re-runs submissions and invalidates results
that drift. The embedder is declared explicitly rather than left to a default.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from . import config

_log = logging.getLogger(__name__)


class Mem0Unavailable(RuntimeError):
    """Raised when the memory core cannot serve a request.

    Surfaced as a retryable HTTP status so the platform's own backoff applies —
    Add retries on 408/409/425/429/5xx, which is the correct response to a
    transient upstream LLM failure. Silently dropping the write instead would
    return 200 for a memory that was never stored.
    """


def _build_config() -> dict[str, Any]:
    vector_config: dict[str, Any] = {"collection_name": config.MEM0_COLLECTION}
    if config.MEM0_VECTOR_STORE == "faiss":
        vector_config["path"] = config.MEM0_VECTOR_PATH
        # Cosine, to match how the embeddings are normalised.
        vector_config["distance_strategy"] = "cosine"
    elif config.MEM0_VECTOR_STORE == "chroma":
        vector_config["path"] = config.MEM0_VECTOR_PATH
    elif config.MEM0_VECTOR_STORE == "qdrant":
        vector_config["path"] = config.MEM0_VECTOR_PATH

    settings: dict[str, Any] = {
        "llm": {
            "provider": "openai",
            "config": {
                # Pinned by the Academic division's rules, not a default.
                "model": config.MEM0_LLM_MODEL,
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": config.MEM0_EMBEDDER_PROVIDER,
            "config": {"model": config.MEM0_EMBEDDER_MODEL},
        },
        "vector_store": {
            "provider": config.MEM0_VECTOR_STORE,
            "config": vector_config,
        },
    }
    if config.MEM0_HISTORY_DB:
        settings["history_db_path"] = config.MEM0_HISTORY_DB
    return settings


class Mem0Backend:
    """Thin, lazily-initialised wrapper around ``mem0.Memory``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._memory: Any = None
        self._failed_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return config.MEM0_ENABLED

    def _ensure(self) -> Any:
        if self._memory is not None:
            return self._memory
        if self._failed_reason:
            raise Mem0Unavailable(self._failed_reason)
        with self._lock:
            if self._memory is not None:
                return self._memory
            if self._failed_reason:
                raise Mem0Unavailable(self._failed_reason)
            if not os.environ.get("OPENAI_API_KEY"):
                # Fail loudly at first use rather than silently degrading: a run
                # that quietly loses the extraction channel would post a score
                # that does not describe the submitted system.
                self._failed_reason = "OPENAI_API_KEY is not set"
                raise Mem0Unavailable(self._failed_reason)
            try:
                from mem0 import Memory

                self._memory = Memory.from_config(_build_config())
                _log.info(
                    "mem0 core ready: llm=%s embedder=%s/%s store=%s",
                    config.MEM0_LLM_MODEL,
                    config.MEM0_EMBEDDER_PROVIDER,
                    config.MEM0_EMBEDDER_MODEL,
                    config.MEM0_VECTOR_STORE,
                )
                return self._memory
            except Mem0Unavailable:
                raise
            except Exception as error:  # noqa: BLE001
                self._failed_reason = f"mem0 init failed: {error}"
                _log.exception("mem0 init failed")
                raise Mem0Unavailable(self._failed_reason) from error

    # ── write ───────────────────────────────────────────────────────────────

    def add(self, messages: list[dict[str, Any]], user_id: str, timestamp: int | None) -> None:
        """Extract and store facts. Raises Mem0Unavailable so Add can 503."""
        memory = self._ensure()
        kwargs: dict[str, Any] = {"user_id": user_id, "infer": True}
        if timestamp is not None:
            # mem0 takes one timestamp per call, while the payload carries one per
            # message. The earliest is the anchor for any relative time expression
            # inside the batch; per-message stamps survive in our raw channel.
            kwargs["timestamp"] = int(timestamp / 1000) if abs(timestamp) > 1e11 else int(timestamp)
        try:
            memory.add(messages, **kwargs)
        except Exception as error:  # noqa: BLE001
            raise Mem0Unavailable(f"mem0 add failed: {error}") from error

    # ── read ────────────────────────────────────────────────────────────────

    def search(self, query: str, user_id: str, top_k: int) -> list[dict[str, Any]]:
        """Return mem0 hits as ``{id, content, score, created_at}`` dicts.

        Read-side failures degrade to an empty channel rather than raising: the
        raw lexical channel can still answer, and a malformed Search response
        fails the whole evaluation stage outright.
        """
        try:
            memory = self._ensure()
        except Mem0Unavailable as error:
            _log.warning("mem0 unavailable for search: %s", error)
            return []
        try:
            result = memory.search(
                query,
                top_k=top_k,
                filters={"user_id": user_id},
                # mem0 defaults to 0.1, which silently drops weak-but-correct
                # evidence. Fusion downstream handles ranking, so keep everything.
                threshold=config.MEM0_SEARCH_THRESHOLD,
            )
        except Exception:  # noqa: BLE001
            _log.exception("mem0 search failed; continuing with the raw channel only")
            return []

        rows = result.get("results", result) if isinstance(result, dict) else result
        if not isinstance(rows, list):
            return []

        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = row.get("memory") or row.get("content") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            out.append(
                {
                    "id": str(row.get("id") or f"mem0_{len(out)}"),
                    "content": text.strip(),
                    "score": row.get("score"),
                    "created_at": row.get("created_at") or row.get("updated_at"),
                }
            )
        return out


MEM0 = Mem0Backend()
