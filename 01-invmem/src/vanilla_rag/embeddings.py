from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local Hugging Face embedding backend for server-side benchmark runs."""

    def __init__(self, model_name: str, device: str | None = None, query_prefix: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        if query_prefix is None and "bge" in model_name.lower() and "-en-" in model_name.lower():
            query_prefix = "Represent this sentence for searching relevant passages: "
        self.query_prefix = query_prefix or ""
        self.identity = f"{model_name}|query_prefix={self.query_prefix}"
        self.model = SentenceTransformer(model_name, device=device)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + text for text in texts])


def build_embedder(model_name: str, device: str | None = None, query_prefix: str | None = None) -> Embedder:
    return SentenceTransformerEmbedder(model_name=model_name, device=device, query_prefix=query_prefix)
