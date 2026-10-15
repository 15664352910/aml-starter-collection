#!/usr/bin/env python3
"""Measure answer-session recall on the public LongMemEval-S dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.schemas import AddRequest, MemoryMessage, SearchRequest
from app.service import MemoryService


def memory_id(request_id: str, ordinal: int) -> str:
    digest = hashlib.sha256(f"{request_id}:{ordinal}".encode()).hexdigest()[:24]
    return f"mem_{digest}"


def timestamp_ms(value: str) -> int:
    parsed = datetime.strptime(value, "%Y/%m/%d (%a) %H:%M").replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--per-type", type=int)
    parser.add_argument("--candidate-limit", type=int, default=400)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text())[args.offset :]
    if args.per_type is not None:
        selected = []
        selected_counts = Counter()
        for item in dataset:
            question_type = str(item.get("question_type", "unknown"))
            if selected_counts[question_type] >= args.per_type:
                continue
            selected.append(item)
            selected_counts[question_type] += 1
        dataset = selected
    if args.limit is not None:
        dataset = dataset[: args.limit]

    cutoffs = (1, 5, 10, 20, 50, 100)
    totals = Counter()
    hits = Counter()
    with tempfile.TemporaryDirectory(prefix="aml-longmemeval-") as temp_dir:
        service = MemoryService(
            Settings(
                database_path=Path(temp_dir) / "memory.db",
                api_key=None,
                llm_mode="off",
                openai_api_key=None,
                openai_base_url="https://api.openai.com/v1",
                openai_model="gpt-4o-mini",
                openai_timeout_seconds=1,
                candidate_limit=args.candidate_limit,
            )
        )
        service.initialize()
        for item_index, item in enumerate(dataset):
            answer_sessions = {str(value) for value in item.get("answer_session_ids") or []}
            if not answer_sessions:
                continue
            user_id = f"longmemeval-{item_index}"
            id_to_session: dict[str, str] = {}
            for session_index, (session_id, date, messages) in enumerate(
                zip(
                    item["haystack_session_ids"],
                    item["haystack_dates"],
                    item["haystack_sessions"],
                )
            ):
                for chunk_start in range(0, len(messages), 20):
                    chunk = messages[chunk_start : chunk_start + 20]
                    request_id = f"{user_id}:{session_id}:chunk-{chunk_start // 20}"
                    service.add(
                        AddRequest(
                            request_id=request_id,
                            messages=[
                                MemoryMessage(
                                    role=message["role"],
                                    content=message["content"],
                                    timestamp=timestamp_ms(date) + ordinal,
                                )
                                for ordinal, message in enumerate(chunk)
                            ],
                            user_id=user_id,
                            session_id=str(session_id),
                        )
                    )
                    for ordinal in range(len(chunk)):
                        id_to_session[memory_id(request_id, ordinal)] = str(session_id)

            results = service.search(
                SearchRequest(query=item["question"], user_id=user_id, top_k=100)
            )
            ranked_groups: list[set[str]] = []
            for result in results:
                source_ids = re.findall(r"\[(mem_[0-9a-f]{24})\s*\|", result.content)
                if not source_ids:
                    source_ids = [result.id]
                ranked_groups.append(
                    {id_to_session[source_id] for source_id in source_ids if source_id in id_to_session}
                )
            question_type = str(item.get("question_type", "unknown"))
            totals["questions"] += 1
            totals[question_type] += 1
            for cutoff in cutoffs:
                visible = set().union(*ranked_groups[:cutoff]) if ranked_groups[:cutoff] else set()
                if visible & answer_sessions:
                    hits[f"recall@{cutoff}"] += 1
                    hits[f"{question_type}@{cutoff}"] += 1

    output: dict[str, int | float | dict[str, float]] = {"questions": totals["questions"]}
    for cutoff in cutoffs:
        output[f"recall@{cutoff}"] = hits[f"recall@{cutoff}"] / totals["questions"]
    output["by_type@100"] = {
        question_type: hits[f"{question_type}@100"] / count
        for question_type, count in totals.items()
        if question_type != "questions"
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
