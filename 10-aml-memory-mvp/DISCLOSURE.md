# Method and originality disclosure

## Submission identity

- System: AML Memory MVP
- Version: 0.3.0
- Track: Textual Memory
- Division: Academic Methods
- Route: Public source code, organizer Docker deployment

## Original work

The API service, database schema, request isolation, retrieval feature construction, ranking, tests, Docker packaging, and documentation in this repository were created specifically for this submission.

No existing agent-memory repository or paper implementation was copied. The implementation combines well-known, independently implemented techniques:

1. SQLite FTS5 and its built-in BM25 ranking.
2. Unicode NFKC normalization.
3. Porter stemming, bounded English character fragments for typo tolerance, and character bigram/trigram indexing for CJK text.
4. General-purpose concept groups for common preference, career, family, health, travel, time, and other memory vocabulary; these contain no benchmark questions or stored answers.
5. Informative-term coverage, candidate rank, phrase overlap, answer-option evidence, and exact date/number anchors.
6. Temporal-intent handling for latest, earliest, point-in-time, and sequence questions, including a correction/update signal.
7. Source-order-safe adjacent-turn expansion and bounded verbatim conversation windows.
8. Incremental maximal-marginal-relevance selection that rewards complementary evidence and suppresses near-duplicate user/assistant echoes.
9. Timestamp rendering inside returned source evidence so the shared Answer model can reason about chronology without relying on hidden metadata.
10. Optional `gpt-4o-mini` per-message index cues and Search-time retrieval terms, evidence facets, and temporal intent.

## Evaluation integrity

- Search returns original source messages, not generated final answers.
- The system contains no benchmark answers, dataset-specific hard-coding, prompt injection, or human-in-the-loop answering path.
- Public benchmark replay scripts measure generic evidence recall during development; benchmark records and answers are not packaged in the image or repository.
- All storage and retrieval is scoped to the exact `user_id` supplied by the evaluator.
- `request_id` provides idempotent ingestion.
- Model and runtime configuration are documented and versioned.
- Optional-model Search analysis is capped at 256 output tokens and Add indexing at 1024; challenge credentials must use a separate API project with a USD 50 budget ceiling.

## AI-assisted development

OpenAI Codex was used as a software-engineering assistant to research the public interface contract, implement the repository, write tests, and prepare reproducibility documentation. Codex is not invoked as a hidden human-in-the-loop participant during evaluation. Runtime model use, when enabled, is explicit in environment configuration and code.
