from __future__ import annotations

from typing import Sequence

import numpy as np
from openai import OpenAI

from .config import Settings


class EmbeddingClient:
    """OpenAI-compatible embedding client (Add / Search share the same model)."""

    # SiliconFlow BAAI/bge-m3 rejects oversized inputs with the same opaque
    # 400/20015 used for unsupported params. Empirically ~8000 CJK chars OK,
    # 8192 fails — keep a safety margin.
    MAX_INPUT_CHARS = 7500

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_api_base,
        )
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions

    def _supports_dimensions(self) -> bool:
        model = (self.model or "").lower()
        base = (self.settings.embedding_api_base or "").lower()
        # bge-m3 fixed-size; SiliconFlow rejects OpenAI-style dimensions.
        if "bge-m3" in model or "bge_m3" in model:
            return False
        if "siliconflow" in base:
            return False
        return True

    @classmethod
    def _clip(cls, text: str) -> str:
        if len(text) <= cls.MAX_INPUT_CHARS:
            return text
        return text[: cls.MAX_INPUT_CHARS]

    def embed(self, text: str) -> np.ndarray:
        vectors = self.embed_batch([text])
        return vectors[0]

    def embed_batch(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        all_vecs: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = [self._clip(t) for t in texts[start : start + batch_size]]
            kwargs = {
                "model": self.model,
                "input": chunk,
            }
            # Only send dimensions when the provider/model actually supports it.
            # SiliconFlow BAAI/bge-m3 rejects dimensions (HTTP 400 code 20015).
            if self.dimensions and self._supports_dimensions():
                kwargs["dimensions"] = self.dimensions
            resp = self._client.embeddings.create(**kwargs)

            ordered = sorted(resp.data, key=lambda x: x.index)
            all_vecs.extend([item.embedding for item in ordered])

        arr = np.asarray(all_vecs, dtype=np.float32)
        # L2-normalize so cosine = dot product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return arr / norms
