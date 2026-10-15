# ReFind method and competition adaptation

## Original work

ReFind is the competition service for the method described in the technical report:

> Ruizhe Li, Benfeng Xu, Mingxuan Du, Licheng Zhang, and Zhendong Mao. **When Your Agent Opens the Chat App: A Human-Inspired Search Interface for Conversational Memory.** Technical report/manuscript, 2026.

- Manuscript source: <https://github.com/imlrz/wyaoca-paper>
- Method implementation and competition wrapper: <https://github.com/imlrz/ReFind>
- Corresponding author: Zhendong Mao
- Affiliations: Ruizhe Li, Mingxuan Du, Licheng Zhang, and Zhendong Mao — University of Science and Technology of China; Benfeng Xu — Metastone Technology.

This repository is the authors' own competition adaptation. It does not claim third-party memory-system code as original work.

## Preserved method components

- Raw conversation text is retained rather than rewritten into generated summaries.
- A ReAct-style planner iteratively chooses keyword searches, reviews results, records useful evidence, and may narrow by date.
- Retrieval operates at conversation-chunk granularity with BM25 (`k1=1.2`, `b=0.75`) and Porter-style stemming.
- Conversation-level and aggregate session-level BM25 rankings are fused with Reciprocal Rank Fusion (`k=60`).
- Retrieved chunks are expanded with a local context window of two neighboring chunks on each side.
- The default planner is GPT-4o-mini, the internal search depth is 5, and the planner budget is 4 actions.

## Changes required by the Agent Memory Challenge

The leaderboard contract differs from the paper's offline LongMemEval runner. This repository makes the following explicit adaptations:

1. **Dynamic Add storage.** Offline per-question JSON indexes are replaced by synchronous, idempotent SQLite ingestion. Each platform Add chunk is stored raw. At Search time, chunks sharing the exact `user_id` and `session_id` are source-ordered and adjacent turns are paired into the conversation-level records used by the paper.
2. **Strict isolation.** All storage and retrieval are keyed by the exact platform `user_id`. No state or index is shared across users.
3. **Seen-chunk filtering.** The paper's seen-session filter is applied at chunk level here. The platform may assign one `session_id` to all chunks for a sample; excluding that whole session after the first search would incorrectly hide the remaining memory.
4. **No answer stage.** The paper's second LLM stage is intentionally omitted. Search returns the selected raw evidence plus local context, and the competition's shared Answer pipeline generates the final answer.
5. **Competition `top_k`.** The platform value is treated as a hard response cap. ReFind's internal planner still sees at most 5 results per search, matching the reported retrieval setup.
6. **Timestamp conversion and ordering.** Unix-millisecond message timestamps are used for optional date filtering and rendered as UTC in returned evidence. Within each session, timestamps are the primary ordering key, the ordinal embedded in platform `request_id` is the stable tie-breaker, and database insertion order is only the final fallback.
7. **Evaluation-data governance.** Raw content is never written to application logs. SQLite records are automatically subject to an at-most-30-day retention window.

No model is used during Add; ingestion is deterministic. The only model used by the memory method is GPT-4o-mini during Search. Selecting a different `LLM_MODEL` is supported for development but would be a different competition configuration and must not be used for the declared submission.

## Declared submission configuration

```text
RETRIEVAL_MODE=agent
LLM_MODEL=openai/gpt-4o-mini
AGENT_MAX_ITERATIONS=4
SEARCH_TOP_K=5
CONTEXT_WINDOW=2
SESSION_RRF=true
RETENTION_DAYS=30
```
