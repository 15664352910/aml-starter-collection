from __future__ import annotations

import math
from datetime import datetime, timezone

from .config import Settings
from .db import MemoryDatabase, MemoryRow
from .llm import MemoryLLM
from .schemas import AddRequest, SearchItem, SearchRequest
from .text import (
    contains_update_marker,
    coverage_score,
    exact_anchor_score,
    fts_query,
    informative_terms,
    normalize_text,
    retrieval_terms,
    semantic_terms,
    temporal_intent,
)


class MemoryService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.database = MemoryDatabase(settings.database_path)
        self.llm = MemoryLLM(settings)

    def initialize(self) -> None:
        self.database.initialize()

    def add(self, request: AddRequest) -> None:
        annotations = self.llm.annotate_messages(request.messages)
        self.database.add(request, annotations)

    def search(self, request: SearchRequest) -> list[SearchItem]:
        analysis = self.llm.analyze_query(request.query, request.options)
        combined = request.query
        if request.options:
            combined += " " + " ".join(request.options)
        if analysis.terms:
            combined += " " + " ".join(analysis.terms)
        if analysis.facets:
            combined += " " + " ".join(analysis.facets)
        combined += " " + " ".join(semantic_terms(request.query, limit=96))
        terms = retrieval_terms(combined, limit=160)
        query = fts_query(terms)

        candidates: list[MemoryRow] = []
        neighbor_boost: dict[str, float] = {}
        if query:
            try:
                candidates = self.database.search(
                    request.user_id,
                    query,
                    self.settings.candidate_limit,
                )
            except Exception:
                candidates = []
        # Conversation questions frequently match the setup turn while the
        # answer lives in the immediately adjacent reply. Expand around the
        # strongest lexical seeds before adding broad fallback candidates.
        if candidates:
            seed_ids = [row.id for row in candidates[:32]]
            seen = {row.id for row in candidates}
            for row, distance, seed_rank in self.database.neighbors(
                request.user_id,
                seed_ids,
                radius=2,
            ):
                if distance:
                    proximity = 0.20 / distance
                    rank_discount = 1.0 / (1.0 + 0.08 * seed_rank)
                    neighbor_boost[row.id] = max(
                        neighbor_boost.get(row.id, 0.0),
                        proximity * rank_discount,
                    )
                if row.id not in seen:
                    candidates.append(row)
                    seen.add(row.id)
        query_terms = set(informative_terms(request.query, limit=48))
        semantic_expansion = semantic_terms(request.query, limit=96)
        expansion_terms = set(
            informative_terms(
                " ".join(analysis.terms + analysis.facets + semantic_expansion),
                limit=96,
            )
        ) - query_terms
        option_text = " ".join(request.options or [])
        option_terms = set(informative_terms(option_text, limit=48))
        intent = analysis.temporal_intent
        if intent == "none":
            intent = temporal_intent(request.query)
        timestamps = [row.occurred_at for row in candidates if row.occurred_at is not None]
        minimum = min(timestamps) if timestamps else 0
        maximum = max(timestamps) if timestamps else 0

        scored: list[tuple[float, MemoryRow]] = []
        row_terms: dict[str, set[str]] = {}
        normalized_query = normalize_text(request.query)
        for rank, row in enumerate(candidates):
            rank_score = 1.0 / (1.0 + rank)
            searchable = f"{row.content} {row.search_text}"
            row_terms[row.id] = set(informative_terms(searchable, limit=512))
            coverage = coverage_score(query_terms, searchable)
            expansion_coverage = coverage_score(expansion_terms, searchable)
            option_coverage = coverage_score(option_terms, searchable)
            anchor = exact_anchor_score(f"{request.query} {option_text}", row.content)
            phrase = 0.10 if normalized_query and normalized_query in normalize_text(row.content) else 0.0
            time_score = 0.0
            relative_time = 0.5
            if row.occurred_at is not None and maximum > minimum:
                relative_time = (row.occurred_at - minimum) / (maximum - minimum)
            if intent == "latest":
                time_score = 0.10 * relative_time
            elif intent == "earliest":
                time_score = 0.10 * (1.0 - relative_time)
            elif intent in {"point", "sequence"} and row.occurred_at is not None:
                time_score = 0.025

            correction_score = 0.04 if intent == "latest" and contains_update_marker(row.content) else 0.0
            score = (
                0.42 * coverage
                + 0.22 * rank_score
                + 0.10 * expansion_coverage
                + 0.06 * option_coverage
                + 0.12 * anchor
                + phrase
                + time_score
                + correction_score
                + neighbor_boost.get(row.id, 0.0)
            )
            if not math.isfinite(score):
                score = 0.0
            scored.append((score, row))

        scored.sort(key=lambda item: (item[0], item[1].occurred_at or 0), reverse=True)
        # Greedy evidence-set reranking rewards complementary facts for multi-hop
        # questions and suppresses near-duplicate user/assistant echoes.
        selected: list[tuple[float, MemoryRow]] = []
        covered: set[str] = set()
        remaining = list(scored)
        max_redundancy = {row.id: 0.0 for _, row in remaining}
        while remaining and len(selected) < request.top_k:
            best_index = 0
            best_utility = -1.0
            for index, (base_score, row) in enumerate(remaining):
                terms_for_row = row_terms[row.id]
                relevant = (query_terms | expansion_terms | option_terms) & terms_for_row
                new_gain = len(relevant - covered) / max(1, len(query_terms | expansion_terms | option_terms))
                utility = base_score + 0.14 * new_gain - 0.12 * max_redundancy[row.id]
                if utility > best_utility:
                    best_index = index
                    best_utility = utility
            _, best_row = remaining.pop(best_index)
            selected.append((best_utility, best_row))
            covered.update((query_terms | expansion_terms | option_terms) & row_terms[best_row.id])
            best_terms = row_terms[best_row.id]
            for _, row in remaining:
                terms_for_row = row_terms[row.id]
                union = terms_for_row | best_terms
                if union:
                    similarity = len(terms_for_row & best_terms) / len(union)
                    max_redundancy[row.id] = max(max_redundancy[row.id], similarity)

        # Return compact verbatim conversation windows. Every line remains a
        # traceable source record; no summary or answer is synthesized.
        window_rows: dict[int, list[MemoryRow]] = {}
        if selected:
            for context_row, _distance, seed_rank in self.database.neighbors(
                request.user_id,
                [row.id for _, row in selected],
                radius=2,
            ):
                window_rows.setdefault(seed_rank, []).append(context_row)

        output: list[SearchItem] = []
        output_characters = 0
        for index, (score, row) in enumerate(selected):
            context = window_rows.get(index, [row])
            context.sort(key=MemoryRow.source_order)
            evidence_lines: list[str] = []
            for source in context:
                if source.occurred_at is not None:
                    source_created_at = datetime.fromtimestamp(
                        source.occurred_at / 1000,
                        tz=timezone.utc,
                    )
                    source_time = source_created_at.isoformat().replace("+00:00", "Z")
                    evidence_lines.append(
                        f"[{source.id} | {source.role} | {source_time}] {source.content}"
                    )
                else:
                    evidence_lines.append(f"[{source.id} | {source.role}] {source.content}")
            evidence = "\n".join(evidence_lines)
            if output_characters + len(evidence) > self.settings.max_output_chars:
                anchor_lines = [line for line in evidence_lines if line.startswith(f"[{row.id} |")]
                evidence = anchor_lines[0] if anchor_lines else evidence_lines[0]
            if output and output_characters + len(evidence) > self.settings.max_output_chars:
                break
            output_characters += len(evidence)
            if row.occurred_at is not None:
                created_at = datetime.fromtimestamp(row.occurred_at / 1000, tz=timezone.utc)
            else:
                created_at = datetime.fromisoformat(row.created_at)
            output.append(
                SearchItem(
                    id=row.id,
                    content=evidence,
                    score=round(max(0.0, min(1.0, score)), 6),
                    created_at=created_at,
                )
            )
        return output
