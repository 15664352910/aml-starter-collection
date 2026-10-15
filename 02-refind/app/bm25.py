from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone

from .store import MemoryChunk
from .tokenizer import Tokenizer


@dataclass(frozen=True)
class SearchHit:
    anchor: MemoryChunk
    context: tuple[MemoryChunk, ...]
    score: float


class BM25Index:
    """Conversation-chunk BM25 with conversation/session rank fusion."""

    def __init__(self, chunks: list[MemoryChunk], k1: float = 1.2, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokenizer = Tokenizer()
        self.term_frequencies: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.document_frequency: dict[str, int] = defaultdict(int)
        self._id_to_index = {chunk.id: index for index, chunk in enumerate(chunks)}
        self._session_positions: dict[str, list[int]] = defaultdict(list)

        for index, chunk in enumerate(chunks):
            terms = self.tokenizer.tokenize(chunk.content)
            counts = Counter(terms)
            self.term_frequencies.append(counts)
            self.doc_lengths.append(len(terms))
            for term in counts:
                self.document_frequency[term] += 1
            self._session_positions[chunk.session_id].append(index)
        self.average_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )

    def _score(self, query_terms: list[str], index: int) -> float:
        if not self.average_length:
            return 0.0
        length = self.doc_lengths[index]
        frequencies = self.term_frequencies[index]
        total = 0.0
        count = len(self.chunks)
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            documents = self.document_frequency[term]
            inverse = math.log((count - documents + 0.5) / (documents + 0.5) + 1.0)
            numerator = frequency * (self.k1 + 1.0)
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * length / self.average_length
            )
            total += inverse * numerator / denominator
        return total

    def search(
        self,
        keywords: list[str],
        top_k: int,
        context_window: int,
        excluded_ids: set[str] | None = None,
        session_rrf: bool = True,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SearchHit]:
        if not self.chunks:
            return []
        query_terms = self.tokenizer.tokenize(" ".join(keywords))
        if not query_terms:
            return []
        excluded_ids = excluded_ids or set()
        scores = {
            index: self._score(query_terms, index) for index in range(len(self.chunks))
        }
        scores = {index: score for index, score in scores.items() if score > 0}

        if session_rrf:
            scores = self._rrf_with_session_ranking(scores)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        hits: list[SearchHit] = []
        for index, score in ranked:
            chunk = self.chunks[index]
            if chunk.id in excluded_ids or not _within_dates(chunk, date_from, date_to):
                continue
            context = self._context(index, context_window)
            hits.append(SearchHit(anchor=chunk, context=tuple(context), score=score))
            if len(hits) >= top_k:
                break
        return hits

    def _rrf_with_session_ranking(self, scores: dict[int, float]) -> dict[int, float]:
        """Fuse conversation BM25 rank with aggregate session BM25 rank."""
        if not scores:
            return {}
        smoothing = 60
        conversation_order = sorted(scores, key=lambda item: (-scores[item], item))
        conversation_rank = {
            index: rank for rank, index in enumerate(conversation_order, 1)
        }

        session_scores: dict[str, float] = defaultdict(float)
        for index, score in scores.items():
            session_scores[self.chunks[index].session_id] += score
        session_order = sorted(
            session_scores, key=lambda item: (-session_scores[item], item)
        )
        session_rank = {
            session_id: rank for rank, session_id in enumerate(session_order, 1)
        }
        return {
            index: 1.0 / (smoothing + conversation_rank[index])
            + 1.0 / (smoothing + session_rank[self.chunks[index].session_id])
            for index in scores
        }

    def _context(self, index: int, window: int) -> list[MemoryChunk]:
        chunk = self.chunks[index]
        positions = self._session_positions[chunk.session_id]
        position = positions.index(index)
        start = max(0, position - window)
        end = min(len(positions), position + window + 1)
        return [self.chunks[item] for item in positions[start:end]]


def _within_dates(
    chunk: MemoryChunk, date_from: str | None, date_to: str | None
) -> bool:
    if chunk.timestamp_ms is None or (not date_from and not date_to):
        return True
    source_date = datetime.fromtimestamp(
        chunk.timestamp_ms / 1000, tz=timezone.utc
    ).date()
    try:
        lower = date.fromisoformat(date_from.replace("/", "-")) if date_from else None
        upper = date.fromisoformat(date_to.replace("/", "-")) if date_to else None
    except ValueError:
        return True
    return (lower is None or source_date >= lower) and (
        upper is None or source_date <= upper
    )
