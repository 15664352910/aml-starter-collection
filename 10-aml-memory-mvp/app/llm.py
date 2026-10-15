from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import Settings
from .schemas import MemoryMessage


LOGGER = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class QueryAnalysis:
    terms: list[str]
    facets: list[str]
    temporal_intent: str


class MemoryLLM:
    """Uses gpt-4o-mini only for indexing cues, never for final answers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._slots = threading.BoundedSemaphore(value=8)

    @property
    def ready(self) -> bool:
        return bool(self.settings.openai_api_key)

    def _enabled(self) -> bool:
        return self.settings.llm_mode != "off"

    def _complete_json(self, system: str, user: str, max_tokens: int = 256) -> dict:
        if not self._enabled():
            return {}
        if not self.settings.openai_api_key:
            if self.settings.llm_mode == "required":
                raise LLMUnavailable("OPENAI_API_KEY is required when AML_LLM_MODE=required")
            return {}

        payload = {
            "model": self.settings.openai_model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            f"{self.settings.openai_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._slots:
                with urllib.request.urlopen(
                    request,
                    timeout=self.settings.openai_timeout_seconds,
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return json.loads(content)
        except (
            urllib.error.URLError,
            TimeoutError,
            TypeError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            if self.settings.llm_mode == "required":
                raise LLMUnavailable(f"gpt-4o-mini request failed: {exc}") from exc
            LOGGER.warning("Optional gpt-4o-mini request failed: %s", exc)
            return {}

    def annotate_messages(self, messages: list[MemoryMessage]) -> list[str]:
        transcript = [
            {
                "index": index,
                "role": message.role,
                "timestamp": message.timestamp,
                "content": message.content,
            }
            for index, message in enumerate(messages)
        ]
        result = self._complete_json(
            system=(
                "You create retrieval index cues for an agent memory system. "
                "Return JSON with an 'items' array. Each item must contain the supplied message "
                "'index' and a 'terms' string array. Extract only names, entities, aliases, dates, "
                "quantities, preferences, decisions, corrections, relationships, rules, and "
                "distinctive phrases that appear in that specific message. Add concise synonyms, "
                "hypernyms, and spelling variants only when they are directly entailed and useful "
                "for later retrieval. Do not move a cue from "
                "one message to another, answer any question, infer new facts, or write a narrative. "
                "Preserve original language where useful."
            ),
            user=json.dumps({"messages": transcript}, ensure_ascii=False),
            max_tokens=min(1024, max(256, len(messages) * 64)),
        )
        annotations = [""] * len(messages)
        items = result.get("items", []) if isinstance(result, dict) else []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                    continue
                index = item["index"]
                if not 0 <= index < len(annotations):
                    continue
                terms = item.get("terms", [])
                if isinstance(terms, list):
                    annotations[index] = " ".join(
                        str(term) for term in terms if isinstance(term, (str, int, float))
                    )

        # Backward-compatible fallback for a valid but coarser JSON response.
        if not any(annotations) and isinstance(result, dict):
            terms = result.get("terms", [])
            if isinstance(terms, list):
                shared = " ".join(
                    str(term) for term in terms if isinstance(term, (str, int, float))
                )
                annotations = [shared] * len(messages)
        return annotations

    def analyze_query(self, query: str, options: list[str] | None) -> QueryAnalysis:
        result = self._complete_json(
            system=(
                "You analyze a memory-retrieval query without answering it. Return JSON with: "
                "(1) a 'terms' string array containing entities, dates, aliases, likely paraphrases, "
                "spelling variants, relations, generic semantic categories, and distinctive option "
                "phrases that could occur in supporting memories; (2) a 'facets' string array containing "
                "the separate evidence needs for a multi-hop question; and (3) 'temporal_intent' as "
                "one of none, point, earliest, latest, or sequence. Never answer the query, select "
                "an option, infer a missing fact, or write a narrative."
            ),
            user=json.dumps({"query": query, "options": options}, ensure_ascii=False),
        )
        terms = result.get("terms", []) if isinstance(result, dict) else []
        facets = result.get("facets", []) if isinstance(result, dict) else []
        intent = result.get("temporal_intent", "none") if isinstance(result, dict) else "none"
        if intent not in {"none", "point", "earliest", "latest", "sequence"}:
            intent = "none"
        return QueryAnalysis(
            terms=[str(term) for term in terms if isinstance(term, (str, int, float))][:48],
            facets=[str(facet) for facet in facets if isinstance(facet, (str, int, float))][:12],
            temporal_intent=str(intent),
        )
