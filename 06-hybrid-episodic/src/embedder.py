"""Sentence-embedding backend with a lexical-only fallback.

The model is loaded lazily and once. If ``EMBED_ENABLED`` is off, or the model
cannot be loaded, ``encode`` returns None and the retriever degrades to BM25
alone rather than failing the run — a degraded score beats a failed stage, and
the platform aborts the whole phase on a malformed response.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from . import config

_log = logging.getLogger(__name__)


class Embedder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._loaded = False
        self._failed = False

    @property
    def available(self) -> bool:
        if not config.EMBED_ENABLED or self._failed:
            return False
        self._ensure()
        return self._model is not None

    def _ensure(self) -> None:
        if self._loaded or self._failed or not config.EMBED_ENABLED:
            return
        with self._lock:
            if self._loaded or self._failed:
                return
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
                self._loaded = True
                _log.info("loaded embedding model %s", config.EMBED_MODEL)
            except Exception:  # noqa: BLE001 - degrade, never crash the service
                self._failed = True
                _log.exception(
                    "embedding model %s unavailable; falling back to lexical-only",
                    config.EMBED_MODEL,
                )

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray | None:
        """L2-normalised float32 embeddings, or None when unavailable."""
        if not texts or not self.available:
            return None
        payload = (
            [config.EMBED_QUERY_PREFIX + t for t in texts] if is_query else list(texts)
        )
        try:
            vectors = self._model.encode(  # type: ignore[union-attr]
                payload,
                batch_size=config.EMBED_BATCH,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype=np.float32)
        except Exception:  # noqa: BLE001
            _log.exception("encode failed; falling back to lexical-only for this call")
            return None


EMBEDDER = Embedder()
