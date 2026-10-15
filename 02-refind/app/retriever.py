from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

from .bm25 import BM25Index, SearchHit
from .config import Settings
from .llm import LLMReply, OpenAICompatibleLLM
from .models import MemoryResult
from .store import MemoryStore

SYSTEM_PROMPT = """You are a research assistant collecting evidence from a user's conversation history. You are NOT answering the question; only gather the most useful memory.

Actions:
search_chatrecord — search by concise keywords. Previously returned chunks are automatically excluded.
Action Input: {"keywords":["keyword1","keyword2"],"date_from":"YYYY-MM-DD","date_to":"YYYY-MM-DD"}
The date fields are optional; omit them unless an absolute date range is useful.

take_note — save relevant results from the LAST search by their 1-based result number.
Action Input: {"indices":[1,3]}

finish_search — finish collecting evidence.
Action Input: {}

Workflow: search -> take_note -> search again -> take_note -> finish_search.
Review every result. Call take_note before another search because the prior result list is replaced. Use diverse keywords on later searches. Never answer the question.

Output exactly:
Thought: one short sentence
Action: one action name
Action Input: valid JSON object
"""


class Planner(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> LLMReply: ...


@dataclass(frozen=True)
class RetrievalStats:
    iterations: int
    searches: int
    notes: int
    prompt_tokens: int
    completion_tokens: int


def _parse_action(text: str) -> tuple[str | None, dict | None]:
    action_match = re.search(r"Action:\s*([\w_]+)", text)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
    if not action_match:
        return None, None
    action = action_match.group(1)
    if action == "finish_search" and not input_match:
        return action, {}
    if not input_match:
        return action, None
    raw = input_match.group(1)
    for candidate in (raw, raw.replace("'", '"')):
        try:
            value = json.loads(candidate)
            return action, value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            continue
    return action, None


class ReFindRetriever:
    def __init__(
        self,
        store: MemoryStore,
        settings: Settings,
        planner: Planner | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.planner = planner
        if (
            self.planner is None
            and settings.retrieval_mode == "agent"
            and settings.llm_api_key
        ):
            self.planner = OpenAICompatibleLLM(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                timeout_seconds=settings.llm_timeout_seconds,
                max_tokens=settings.llm_max_tokens,
            )

    def search(
        self,
        query: str,
        options: list[str] | None,
        user_id: str,
        top_k: int,
    ) -> tuple[list[MemoryResult], RetrievalStats]:
        chunks = self.store.list_chunks(user_id)
        if not chunks:
            return [], RetrievalStats(0, 0, 0, 0, 0)
        index = BM25Index(chunks)
        if self.settings.retrieval_mode == "agent" and self.planner is None:
            raise RuntimeError(
                "agent retrieval is enabled but no LLM_API_KEY was configured"
            )
        if self.settings.retrieval_mode == "bm25":
            return self._direct_search(index, query, options, top_k)
        return self._agent_search(index, query, options, top_k)

    def _direct_search(
        self,
        index: BM25Index,
        query: str,
        options: list[str] | None,
        top_k: int,
    ) -> tuple[list[MemoryResult], RetrievalStats]:
        keywords = [query, *(options or [])]
        hits = index.search(
            keywords=keywords,
            top_k=top_k,
            context_window=self.settings.context_window,
            session_rrf=self.settings.session_rrf,
        )
        results = [self._to_result(hit, rank) for rank, hit in enumerate(hits)]
        return results, RetrievalStats(0, 1, len(results), 0, 0)

    def _agent_search(
        self,
        index: BM25Index,
        query: str,
        options: list[str] | None,
        top_k: int,
    ) -> tuple[list[MemoryResult], RetrievalStats]:
        option_text = "\nOptions:\n" + "\n".join(options) if options else ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question to research: {query}{option_text}\n"
                    "Search the conversation history and collect all relevant evidence. "
                    "Start with a broad keyword search."
                ),
            },
        ]
        excluded_ids: set[str] = set()
        notes: OrderedDict[str, SearchHit] = OrderedDict()
        last_hits: list[SearchHit] = []
        searches = 0
        prompt_tokens = 0
        completion_tokens = 0
        iterations = 0

        for iteration in range(self.settings.agent_max_iterations):
            iterations = iteration + 1
            reply = self.planner.complete(messages)  # type: ignore[union-attr]
            prompt_tokens += reply.prompt_tokens
            completion_tokens += reply.completion_tokens
            action, action_input = _parse_action(reply.text)
            messages.append({"role": "assistant", "content": reply.text})

            if action == "finish_search":
                break
            if action == "search_chatrecord" and isinstance(action_input, dict):
                keywords = action_input.get("keywords")
                if not isinstance(keywords, list) or not all(
                    isinstance(item, str) for item in keywords
                ):
                    observation = (
                        "Error: keywords must be a non-empty array of strings."
                    )
                else:
                    date_from = action_input.get("date_from")
                    date_to = action_input.get("date_to")
                    last_hits = index.search(
                        keywords=keywords,
                        top_k=self.settings.search_top_k,
                        context_window=self.settings.context_window,
                        excluded_ids=excluded_ids,
                        session_rrf=self.settings.session_rrf,
                        date_from=date_from if isinstance(date_from, str) else None,
                        date_to=date_to if isinstance(date_to, str) else None,
                    )
                    searches += 1
                    for hit in last_hits:
                        excluded_ids.add(hit.anchor.id)
                    observation = self._format_observation(last_hits)
            elif action == "take_note" and isinstance(action_input, dict):
                indices = action_input.get("indices", [])
                saved = 0
                if isinstance(indices, list):
                    for item in indices:
                        if isinstance(item, int) and 1 <= item <= len(last_hits):
                            hit = last_hits[item - 1]
                            if hit.anchor.id not in notes:
                                notes[hit.anchor.id] = hit
                                saved += 1
                observation = f"Saved {saved} new result(s). Total notes: {len(notes)}."
            else:
                observation = "Use exactly one valid action: search_chatrecord, take_note, or finish_search."

            messages.append(
                {
                    "role": "user",
                    "content": f"Observation: {observation}\nRespond with the next action only.",
                }
            )

        # Preserve useful retrieval even if the planner used its last action to search
        # or failed to issue take_note within the four-step budget.
        if not notes:
            for hit in last_hits:
                notes[hit.anchor.id] = hit
        if not notes:
            fallback, stats = self._direct_search(index, query, options, top_k)
            return fallback, RetrievalStats(
                iterations,
                searches + stats.searches,
                len(fallback),
                prompt_tokens,
                completion_tokens,
            )

        selected = list(notes.values())[:top_k]
        results = [self._to_result(hit, rank) for rank, hit in enumerate(selected)]
        return results, RetrievalStats(
            iterations, searches, len(results), prompt_tokens, completion_tokens
        )

    @staticmethod
    def _format_observation(hits: list[SearchHit]) -> str:
        if not hits:
            return "No matching memory found."
        blocks: list[str] = []
        for rank, hit in enumerate(hits, 1):
            lines = [
                f"Result {rank} | score={hit.score:.4f} | session={hit.anchor.session_id}"
            ]
            for chunk in hit.context:
                marker = "[MATCH] " if chunk.id == hit.anchor.id else ""
                lines.append(f"{marker}{chunk.content}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @staticmethod
    def _to_result(hit: SearchHit, rank: int) -> MemoryResult:
        # Context is returned as one evidence block, which mirrors ReFind's phase-2
        # input while leaving final answer generation to the competition pipeline.
        sections: list[str] = []
        seen: set[str] = set()
        for chunk in hit.context:
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            sections.append(chunk.content)
        score = float(hit.score / (1.0 + 0.03 * rank))
        return MemoryResult(
            id=hit.anchor.id,
            content="\n\n".join(sections),
            score=score,
            created_at=hit.anchor.created_at,
        )
