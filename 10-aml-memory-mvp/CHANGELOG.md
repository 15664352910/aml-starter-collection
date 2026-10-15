# Changelog

## 0.3.0 — final competition candidate

- Added Porter-stemmed FTS, bounded typo fragments, phrase tokens, and general concept expansion.
- Added source-order-safe adjacent-turn expansion across concurrently ingested chunks.
- Added bounded verbatim conversation windows with traceable source identifiers.
- Removed unrelated recent-memory backfill to improve privacy and abstention behavior.
- Added a 240,000-character Answer-context safety budget.
- Accepted protocol-valid blank source messages found in public benchmark replay.
- Limited concurrent optional model calls and strengthened non-answering retrieval prompts.
- Added public LoCoMo and LongMemEval-S retrieval-recall development harnesses.
- Expanded regression coverage for privacy-safe empty results and session-bounded windows.

## 0.2.0 — competition candidate

- Added per-message gpt-4o-mini indexing cues and multi-facet query analysis.
- Added latest, earliest, point-in-time, and sequence intent scoring.
- Added correction/update, option, exact date/number, and informative-term signals.
- Added incremental maximal-marginal-relevance evidence-set reranking.
- Added source timestamps to returned evidence.
- Expanded regression coverage for temporal updates, multi-hop retrieval, choices, and cue isolation.

## 0.1.0 — initial MVP

- Added synchronous Add/Search contracts, SQLite FTS5, CJK n-grams, Docker, CI, and smoke tests.
