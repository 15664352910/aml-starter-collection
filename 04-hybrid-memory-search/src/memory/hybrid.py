from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25_index import BucketBM25
from .embedding import EmbeddingClient
from .store import MemoryRecord


@dataclass
class SearchHit:
    id: str
    text: str
    score: float
    bm25: float
    bm25_norm: float
    dense: float
    dense_norm: float
    metadata: dict


def min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1]. Flat range → all zeros (or [1.0] for n=1)."""
    n = len(scores)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    arr = np.asarray(scores, dtype=np.float64)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-12:
        return [0.0] * n
    return [float((x - mn) / (mx - mn)) for x in arr]


def hybrid_search(
    records: list[MemoryRecord],
    embeddings: dict[str, np.ndarray],
    query: str,
    embedder: EmbeddingClient,
    top_k: int = 5,
    alpha: float = 0.5,
) -> list[SearchHit]:
    """Same-bucket hybrid: α * bm25_norm + (1-α) * dense_norm, return top-k texts."""
    if not records:
        return []

    ids = [r.id for r in records]
    texts = [r.text for r in records]
    id_to_rec = {r.id: r for r in records}

    # Sparse
    bm25 = BucketBM25(ids, texts)
    sparse = bm25.score(query)
    bm25_map = {i: s for i, s in zip(sparse.ids, sparse.scores)}

    # Dense — only memories that already have cached embeddings
    usable_ids = [mid for mid in ids if mid in embeddings]
    dense_map: dict[str, float] = {mid: 0.0 for mid in ids}
    if usable_ids:
        q_vec = embedder.embed(query)
        mat = np.stack([embeddings[mid] for mid in usable_ids], axis=0)
        sims = mat @ q_vec  # both L2-normalized → cosine
        for mid, sim in zip(usable_ids, sims.tolist()):
            dense_map[mid] = float(sim)

    bm25_raw = [bm25_map.get(mid, 0.0) for mid in ids]
    dense_raw = [dense_map.get(mid, 0.0) for mid in ids]
    bm25_norm = min_max_normalize(bm25_raw)
    dense_norm = min_max_normalize(dense_raw)

    hits: list[SearchHit] = []
    for i, mid in enumerate(ids):
        score = alpha * bm25_norm[i] + (1.0 - alpha) * dense_norm[i]
        rec = id_to_rec[mid]
        hits.append(
            SearchHit(
                id=mid,
                text=rec.text,
                score=float(score),
                bm25=float(bm25_raw[i]),
                bm25_norm=float(bm25_norm[i]),
                dense=float(dense_raw[i]),
                dense_norm=float(dense_norm[i]),
                metadata=dict(rec.metadata),
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    k = max(0, int(top_k))
    return hits[:k] if k else hits
