# Prepared evaluation submission

Copy the following technical information into the Agent Memory Leaderboard application. Personal contact fields must be completed by the participant.

## Recommended selections

- Evaluation type: Textual Memory
- Participant division: Academic Methods
- Submission route: Code submission / organizer deployment
- System name: AML Memory MVP
- Version: 0.3.0
- Repository: `https://github.com/0xboyu/aml-memory-mvp`
- License: MIT

## Short method description

AML Memory MVP is an evidence-only long-term memory retriever built for reproducibility and strict evaluation isolation. It persists each source message in SQLite, creates Porter-stemmed English, typo-fragment, phrase, and CJK lexical features, retrieves candidates with FTS5 BM25, expands around adjacent source turns, and reranks with general semantic concepts, informative-term coverage, exact anchors, answer-option evidence, correction-aware temporal intent, and incremental maximal-marginal relevance. Search returns bounded, timestamped, verbatim conversation windows with stable source identifiers. Optional gpt-4o-mini calls add per-message retrieval cues during Add and query facets during Search; they never generate final answers.

## Docker build and run

```bash
docker build -t aml-memory-mvp:0.3.0 .
docker run --rm -p 8000:8000 \
  -e AML_DATABASE_PATH=/data/memory.db \
  -e AML_LLM_MODE=best_effort \
  -e OPENAI_MODEL=gpt-4o-mini \
  -e OPENAI_API_KEY \
  -v aml-memory-data:/data \
  aml-memory-mvp:0.3.0
```

## API entrypoints

- Health: `GET /health`
- Add: `POST /add`
- Search: `POST /search`
- Authentication: none for organizer-deployed code; optional `AML_API_KEY` is supported

## Runtime notes

- Python 3.12 Docker base image.
- Persistent path: `/data/memory.db`.
- One Uvicorn worker; SQLite WAL and busy timeout handle concurrent evaluator requests.
- Add is synchronous and returns only when evidence is searchable.
- Search accepts and returns at most Top K 100.
- No external database is required.
- Search evidence is capped at 240,000 characters to protect the shared Answer context.
- gpt-4o-mini JSON output is capped at 256 tokens for Search and 1024 tokens for larger Add batches.
- If the platform supports runtime secret injection, configure a dedicated OpenAI API project with a USD 50 budget ceiling and supply its credential only through an organizer-approved secret channel. Without a credential, `best_effort` mode automatically uses the deterministic retriever and keeps Add/Search available.

## Source and changes disclosure

Original implementation; no existing memory-system codebase was reused. Standard SQLite FTS5/BM25, Unicode normalization, stemming, typo fragments, general concept expansion, adjacent source windows, temporal/anchor scoring, maximal-marginal-relevance selection, and optional model-generated retrieval cues are disclosed in `DISCLOSURE.md`.

## Information the participant must add

- Full name
- Contact email
- Organization/team, or `Independent`
- GitHub repository URL after publication
- Public display consent choices
- Any organizer-requested model credential delivery method; never put a secret in this file or an email body
