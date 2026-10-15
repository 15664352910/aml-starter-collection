# Agent Memory Challenge submission sheet

Use these values when creating the evaluation request. Keep all issued keys out of this repository and out of free-text fields unless a field explicitly requests one.

| Field | Declared value |
|---|---|
| System name | ReFind |
| Version | v0.1.0 |
| Evaluation type | Textual Memory |
| Division | Academic Methods |
| Participation route | Code — platform deployment |
| Public repository | `https://github.com/imlrz/ReFind` |
| License | MIT |
| API authentication | None inside the isolated deployment |
| Add endpoint | `POST /v1/memories/add` |
| Search endpoint | `POST /v1/memories/search` |
| Health endpoint | `GET /health` |
| API entrypoint | `app.main:app` |
| Container port | 8000 |
| Planner model | `openai/gpt-4o-mini` through OpenRouter |
| Suggested Add concurrency | 16 |
| Suggested Search concurrency | 16 |
| Supported platform top_k | 1–1000; 100 is recommended for the official run |

The contact email and non-public credentials should be entered only in the organizer's application form, not committed here.

## Platform build and startup

```bash
docker build -t refind:v0.1.0 .
docker run --rm \
  -p 8000:8000 \
  -e RETRIEVAL_MODE=agent \
  -e LLM_API_KEY="$OPENROUTER_API_KEY" \
  -e LLM_MODEL=openai/gpt-4o-mini \
  -v refind-data:/data \
  refind:v0.1.0
```

No evaluation or memory-system key is embedded in the image. The maintainer must inject an OpenRouter-compatible GPT-4o-mini key at runtime because the declared retrieval method uses that model during Search.

## Pre-submission evidence

- Unit and contract suite: `python -m pytest`
- Local contract smoke: `RETRIEVAL_MODE=bm25 python scripts/smoke_test.py http://127.0.0.1:8000`
- Container health: Docker `HEALTHCHECK` calls `/health`
- Method and adaptation disclosure: [METHOD_CARD.md](METHOD_CARD.md)
- API contract and configuration: [../README.md](../README.md)

Before starting Full, record the immutable Git commit and confirm that it matches the `v0.1.0` tag. Full evaluation freezes this version under the challenge rules.
