#!/usr/bin/env python3
"""Measure evidence recall on the public LoCoMo conversations.

This is a retrieval-only development check. It never reads answer labels while
ranking and does not change the service configuration used by the evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.schemas import AddRequest, MemoryMessage, SearchRequest
from app.service import MemoryService


def memory_id(request_id: str, ordinal: int) -> str:
    digest = hashlib.sha256(f"{request_id}:{ordinal}".encode()).hexdigest()[:24]
    return f"mem_{digest}"


def session_number(name: str) -> int:
    return int(name.removeprefix("session_"))


def evaluate(
    dataset_path: Path,
    candidate_limit: int,
    max_conversations: int | None = None,
    max_questions: int | None = None,
    show_failures: int = 0,
) -> dict:
    dataset = json.loads(dataset_path.read_text())
    if max_conversations is not None:
        dataset = dataset[:max_conversations]
    cutoffs = (1, 5, 10, 20, 50, 100)
    hits = Counter()
    totals = Counter()
    failures: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="aml-locomo-") as temp_dir:
        settings = Settings(
            database_path=Path(temp_dir) / "memory.db",
            api_key=None,
            llm_mode="off",
            openai_api_key=None,
            openai_base_url="https://api.openai.com/v1",
            openai_model="gpt-4o-mini",
            openai_timeout_seconds=1,
            candidate_limit=candidate_limit,
        )
        service = MemoryService(settings)
        service.initialize()

        for conversation_index, sample in enumerate(dataset):
            conversation = sample["conversation"]
            user_id = f"locomo-{conversation_index}"
            id_to_dialogue: dict[str, str] = {}
            session_names = sorted(
                (
                    key
                    for key, value in conversation.items()
                    if key.startswith("session_")
                    and not key.endswith("_date_time")
                    and isinstance(value, list)
                ),
                key=session_number,
            )
            for session_name in session_names:
                session_index = session_number(session_name)
                turns = conversation[session_name]
                for chunk_index in range(0, len(turns), 20):
                    chunk = turns[chunk_index : chunk_index + 20]
                    request_id = f"{user_id}:{session_name}:chunk-{chunk_index // 20}"
                    messages = []
                    for ordinal, turn in enumerate(chunk):
                        role = "user" if turn.get("speaker") == conversation.get("speaker_a") else "assistant"
                        content = str(turn.get("text") or "")
                        caption = str(turn.get("blip_caption") or "").strip()
                        if caption:
                            content = f"{content}\n[Image: {caption}]"
                        messages.append(
                            MemoryMessage(
                                role=role,
                                content=content,
                                timestamp=1_700_000_000_000 + session_index * 86_400_000 + ordinal,
                            )
                        )
                        id_to_dialogue[memory_id(request_id, ordinal)] = str(turn["dia_id"])
                    service.add(
                        AddRequest(
                            request_id=request_id,
                            messages=messages,
                            user_id=user_id,
                            session_id=f"{user_id}:{session_name}",
                        )
                    )

            for item in sample["qa"]:
                evidence = {str(value) for value in item.get("evidence") or []}
                category = int(item.get("category", 0))
                if not evidence or category not in {1, 2, 3}:
                    continue
                if max_questions is not None and totals["questions"] >= max_questions:
                    break
                results = service.search(
                    SearchRequest(
                        query=str(item["question"]),
                        user_id=user_id,
                        top_k=100,
                    )
                )
                ranked_groups: list[list[str]] = []
                for result in results:
                    source_ids = re.findall(r"\[(mem_[0-9a-f]{24})\s*\|", result.content)
                    if not source_ids:
                        source_ids = [result.id]
                    ranked_groups.append(
                        [id_to_dialogue[source_id] for source_id in source_ids if source_id in id_to_dialogue]
                    )
                ranked = [dialogue_id for group in ranked_groups for dialogue_id in group]
                totals["questions"] += 1
                totals[f"category_{category}"] += 1
                for cutoff in cutoffs:
                    retrieved = {
                        dialogue_id
                        for group in ranked_groups[:cutoff]
                        for dialogue_id in group
                    }
                    if evidence <= retrieved:
                        hits[f"all@{cutoff}"] += 1
                    hits[f"micro@{cutoff}"] += len(evidence & retrieved)
                    totals[f"micro@{cutoff}"] += len(evidence)
                    if evidence & retrieved:
                        hits[f"any@{cutoff}"] += 1
                    if evidence <= retrieved:
                        hits[f"category_{category}_all@{cutoff}"] += 1
                top_ten = {
                    dialogue_id for group in ranked_groups[:10] for dialogue_id in group
                }
                if evidence.isdisjoint(top_ten) and len(failures) < show_failures:
                    failures.append(
                        {
                            "question": item["question"],
                            "category": category,
                            "expected": sorted(evidence),
                            "retrieved": ranked[:10],
                            "contents": [result.content for result in results[:10]],
                        }
                    )

    metrics: dict[str, float | int] = {"questions": totals["questions"]}
    for cutoff in cutoffs:
        metrics[f"all_recall@{cutoff}"] = hits[f"all@{cutoff}"] / totals["questions"]
        metrics[f"any_recall@{cutoff}"] = hits[f"any@{cutoff}"] / totals["questions"]
        metrics[f"evidence_recall@{cutoff}"] = hits[f"micro@{cutoff}"] / totals[f"micro@{cutoff}"]
        for category in (1, 2, 3):
            metrics[f"category_{category}_all@{cutoff}"] = (
                hits[f"category_{category}_all@{cutoff}"] / totals[f"category_{category}"]
            )
    if failures:
        metrics["failures"] = failures
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--candidate-limit", type=int, default=400)
    parser.add_argument("--max-conversations", type=int)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--show-failures", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.dataset,
                args.candidate_limit,
                args.max_conversations,
                args.max_questions,
                args.show_failures,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
