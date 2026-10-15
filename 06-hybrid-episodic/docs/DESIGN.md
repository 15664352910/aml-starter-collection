# Design rationale

Every decision below traces to a specific fact in the official `Add` / `Search`
contract or in the platform's published evaluation code
(`AML-memory/agent-memory-leaderboard`, pinned at commit `34405461`). Anything
without such a basis did not get built.

## Why retrieval runs over raw turns, not extracted facts

The obvious alternative was to build on an extraction-based memory system — mem0
and its peers distil conversations into condensed fact statements, and their
published numbers are strong. That path is implemented here
(`src/mem0_backend.py`) but **disabled in this submission**, and the reason it is
not the primary design is that extraction is lossy in three ways the platform
happens to score.

From the official answer prompt (`locomo-refined/pipeline.py`):

> Your memories are **episodic raw observations**. Reason about what they imply.

> **Preserve specific names, titles, places, and labels** from the memories.
> Use "Rob" not "a colleague", "Sweden" not "home country".

> Convert relative times like "yesterday", "last month", and "last year" into
> dates, months, or years **when the memory timestamp makes it clear**.

An extracted fact is a conclusion, not an observation; it paraphrases surface
form; and it usually no longer carries the timestamp of the turn it came from.
Each of those maps onto a scored competency — explicit fact recall, and time and
event sequences.

So this system indexes the original turns, each rendered with its own clock and
speaker intact, and retrieves over those. The two fused channels are lexical
(BM25) and dense (a frozen encoder) over that same raw index.

The extraction channel remains in the tree behind `MEM0_ENABLED` because the two
approaches are complementary rather than exclusive — semantic consolidation plus
verbatim evidence — and the fusion layer already accepts it as a third channel.
It ships off because it has not been validated against a live endpoint, and
submitting an unverified path would risk the contract-validation stage. It also
makes the system non-deterministic, which matters under a rule that re-runs
submissions and invalidates results that drift.

## Why timestamp and role are baked into `content`

The platform hands `content` verbatim to the answer model. `created_at` is an
*optional* field in the contract, and nothing indicates the answer model reads it
— what it reads is `content`. A unit whose text does not carry its own timestamp
cannot support the relative-date resolution the prompt asks for.

The same prompt attributes memories per speaker (`{{speaker_1_memories}}` /
`{{speaker_2_memories}}`), so role goes in as well.

Format: `[YYYY-MM-DD HH:MM] role: text`, in `chunker.render()`.

## Why BM25 alongside dense retrieval

The prompt's instruction to preserve specific names, titles and places is a
lexical requirement. Dense retrieval is unreliable on low-frequency proper nouns —
"Freya", "Vasa Museum", "Uppsala" — and BM25 is what pins them. Paraphrased
questions run the other way: "Who is training for a marathon?" shares almost no
surface form with "my sister Freya is training for the Berlin marathon", and only
the dense channel bridges that. Both are needed.

The encoder is `bge-small-en-v1.5`: 33M parameters, CPU-viable at the platform's
64-way write concurrency, and frozen — no fine-tuning, and no training on
evaluation data. `EMBED_ENABLED=false` drops to BM25 alone, which puts zero models
in the Add/Search path.

## Why RRF rather than weighted score blending

BM25 scores are unbounded and drift with corpus statistics; cosine similarity sits
in a bounded range. Blending them linearly requires score normalisation, and the
normalisation constants depend on the corpus — which here is *per `user_id`*, with
a different distribution for every conversation. One set of constants cannot fit
them all.

RRF consumes only ranks, so it needs no per-corpus calibration. That property is
also what lets the optional extraction channel drop in later without re-tuning
anything. Damping `k=60` is the value from the original paper.

## Why options get their own channel

Multiple-choice options carry the discriminating content, so ignoring them wastes
signal. But appending them to the query dilutes the question: a query is on the
order of ten words and the options can be a hundred, so the term statistics end up
dominated by option text. Results then drift toward "lexically similar to some
option" rather than "answers this question".

A separate channel at weight `0.6`, fused alongside the primary query, keeps the
question dominant while still exploiting the options.
Pinned by `test_options_are_used_for_retrieval`.

## Why recency is only a small nudge

The answer prompt again:

> If memories conflict, **prefer the most recent** supported memory.

Memory governance — updates, conflict resolution — is a scored competency, so
aligning retrieval order with how conflicts are actually resolved downstream is
right. But the weight has to stay small: many explicit-fact questions are answered
by early turns, and a heavy recency term evicts them from the candidate set
outright. `RECENCY_WEIGHT=0.15`, scaled by the current top fused score, so it only
breaks near-ties. Set to `0` to disable.

## Why near-duplicates are collapsed across channels

Two channels over the same index will often surface overlapping text, and an
extracted fact — when that channel is enabled — frequently restates a raw turn
almost verbatim. Returning both wastes a `top_k` slot and, worse, reads to the
answer model as two independent sources agreeing on the same claim. Dedup is on
punctuation- and case-normalised text, keeping whichever copy fused higher.

## Why short messages are merged forward

Conversations are full of "yes", "ok", "haha". Indexed alone they crowd out
informative units, are worthless as `content` for the answer model, and depress
the mean document length in a way that distorts BM25's length normalisation.

Anything under `MIN_UNIT_CHARS` merges into the following message, and consecutive
same-role messages merge too, since those are usually one utterance split across
sends. The merged unit keeps the **earliest** timestamp — that is what any relative
time expression inside it is anchored to.

## Why indexing runs inline on the write path

The contract:

> A success response may only be returned once the write has completed and the
> memory is **immediately retrievable**. Do not return HTTP 202, a task ID, or a
> status URL.

A background queue with an immediate 200 is a direct violation, so embedding and
indexing both happen inside the request. The platform drives `Add` with 64
workers, which is why the encoder is a 33M-parameter CPU model rather than
something larger, and why the index is append-only under a per-user lock with the
BM25 statistics updated in the same critical section — there is no deferred build
step that a following `Search` could race.

## Why a failed write returns 503, not 200

This matters only when the optional extraction channel is enabled, but the
principle is the reason the code is shaped this way. If a write fails partway —
an upstream rate limit, say — returning 200 would report success for a memory that
was never stored, and the platform would never retry it. Those facts would be
absent from the run for good.

503 is inside the platform's `Add` retry set (408/409/425/429/5xx), so failing
loudly hands the write back for backoff. The request is also **not** marked as
applied, so the retry actually redoes the work instead of being skipped as a
duplicate. Both are pinned by tests in `test_fusion.py`.

`MEM0_FAIL_ADD_ON_ERROR=false` inverts this for cases where degraded coverage
beats a stalled run.

## Why a failed read degrades instead

The mirror image, and the asymmetry is deliberate:

> Even with HTTP status 200, if `Search` does not return a `data` array … the
> current stage **fails immediately**.

On the read path there is no retry that helps and a bad response aborts the whole
stage. So `Search` catches extraction failures and answers from the raw channel
alone. A lower score is strictly better than a failed evaluation — especially
given `full` runs once per three months.

## Why `request_id` deduplication

`Add` is retried on 408/409/425/429/5xx and on transport errors. Without dedup, a
single timeout-then-retry indexes the same content twice: duplicate memories
consume `top_k` slots and present the answer model with false corroboration.

## Why a single process

The raw index lives in process memory. Multiple uvicorn workers would scatter one
`user_id`'s memories across processes, and each `Search` would see only the shard
it happened to land on — a silent recall loss that raises no error and shows up
only as a mediocre score. Do not raise `WEB_CONCURRENCY` without first moving the
index out of process.

## Known trade-offs, left for the next cycle

1. **No LLM fact extraction in the submitted configuration.** This is the largest
   known gap against extraction-based systems, and the code for it already exists
   behind `MEM0_ENABLED`. It ships off because it is unvalidated end-to-end and
   because it makes the system non-deterministic under a rule that re-runs
   submissions. Enabling and validating it is the first priority for the next
   cycle.
2. **No entity or relation graph.** Multi-hop composition is the weakest covered
   competency; single-shot retrieval over turns only goes so far.
3. **No explicit conflict resolution.** Memory governance is approximated by the
   recency nudge; nothing models "this fact supersedes that one".
4. **`MAX_RETURN` is untuned.** The judge penalises over-answering on list
   questions while the large-context datasets reward recall, and the trade-off
   cannot be resolved without private-set feedback.
5. **No persistence across restarts.** Fine within a single evaluation run, but a
   restart mid-run loses the index.
6. **The optional extraction channel takes one timestamp per `Add` call**, not per
   message, since that is mem0's calling convention. Per-message granularity
   survives only in the raw index.
