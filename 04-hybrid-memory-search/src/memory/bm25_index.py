from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .tokenizer import tokenize


@dataclass
class BM25Scores:
    ids: list[str]
    scores: list[float]


class BucketBM25:
    """In-memory BM25 over one bucket's memory texts."""

    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = ids
        self.corpus_tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def score(self, query: str) -> BM25Scores:
        if not self._bm25 or not self.ids:
            return BM25Scores(ids=[], scores=[])
        q_tokens = tokenize(query)
        if not q_tokens:
            return BM25Scores(ids=list(self.ids), scores=[0.0] * len(self.ids))
        scores = self._bm25.get_scores(q_tokens)
        return BM25Scores(ids=list(self.ids), scores=[float(s) for s in scores])
