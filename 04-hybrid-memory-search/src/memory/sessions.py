from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


@dataclass
class SessionUnit:
    """One memory unit = one full session with header metadata."""

    session_index: int
    date_time: str
    text: str
    turns: list[dict[str, Any]]


@dataclass
class ConversationSample:
    bucket_id: str
    sample_id: str
    sessions: list[SessionUnit]
    source_path: str | None = None


def _turn_line(turn: dict[str, Any]) -> str:
    speaker = turn.get("speaker") or turn.get("role") or "Unknown"
    text = turn.get("text") or turn.get("content") or ""
    if turn.get("type") == "narration" and not turn.get("speaker"):
        return str(text)
    return f"{speaker}: {text}"


def _format_session_text(session_index: int, date_time: str, turns: list[dict[str, Any]]) -> str:
    header = f"[Session {session_index} | {date_time}]"
    body = "\n".join(_turn_line(t) for t in turns if (t.get("text") or t.get("content")))
    return f"{header}\n{body}".strip()


def extract_sessions_from_conversation(conversation: dict[str, Any]) -> list[SessionUnit]:
    """Parse LoCoMo / LongMemEval style conversation dict into session units."""
    # Nested format_example (placeholder / stripped releases)
    if "format_example" in conversation and isinstance(conversation["format_example"], dict):
        conversation = conversation["format_example"]

    # Public LoCoMo export: sessions: [{session_index, date_time, messages}]
    if isinstance(conversation.get("sessions"), list):
        units: list[SessionUnit] = []
        for sess in conversation["sessions"]:
            idx = int(sess.get("session_index") or sess.get("session_id") or (len(units) + 1))
            date_time = str(sess.get("date_time") or sess.get("session_date_time") or "Unknown")
            turns = sess.get("messages") or sess.get("dialogues") or sess.get("turns") or []
            text = _format_session_text(idx, date_time, turns)
            units.append(SessionUnit(session_index=idx, date_time=date_time, text=text, turns=list(turns)))
        units.sort(key=lambda s: s.session_index)
        return units

    # Classic session_N / session_N_date_time keys
    indices: list[int] = []
    for key in conversation.keys():
        m = _SESSION_KEY_RE.match(key)
        if m and isinstance(conversation[key], list):
            indices.append(int(m.group(1)))
    indices.sort()

    units = []
    for idx in indices:
        turns = conversation.get(f"session_{idx}") or []
        date_time = str(conversation.get(f"session_{idx}_date_time") or "Unknown")
        text = _format_session_text(idx, date_time, turns)
        units.append(SessionUnit(session_index=idx, date_time=date_time, text=text, turns=list(turns)))
    return units


def _infer_bucket_id(obj: dict[str, Any], fallback: str) -> str:
    for key in ("conversation_id", "sample_id", "bucket_id", "user_id", "dialogue_id"):
        if obj.get(key):
            return str(obj[key])
    if obj.get("source") and obj.get("sample_id"):
        return f"{obj['source']}:{obj['sample_id']}"
    return fallback


def _sample_from_obj(obj: dict[str, Any], fallback_bucket: str, source_path: str | None) -> ConversationSample | None:
    conversation = obj.get("conversation")
    if conversation is None and ("session_1" in obj or "sessions" in obj):
        conversation = obj
    if not isinstance(conversation, dict):
        return None

    sessions = extract_sessions_from_conversation(conversation)
    if not sessions:
        return None

    bucket_id = _infer_bucket_id(obj, fallback_bucket)
    sample_id = str(obj.get("sample_id") or bucket_id)
    return ConversationSample(
        bucket_id=bucket_id,
        sample_id=sample_id,
        sessions=sessions,
        source_path=source_path,
    )


def load_samples_from_path(data_path: str | Path) -> list[ConversationSample]:
    """Load conversation samples from a file or directory (path is never hardcoded)."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"data path not found: {path}")

    files: list[Path]
    if path.is_dir():
        files = sorted(
            [p for p in path.rglob("*") if p.suffix.lower() in {".json", ".jsonl"}]
        )
    else:
        files = [path]

    samples: list[ConversationSample] = []
    for fp in files:
        # skip obvious non-conversation artifacts
        name = fp.name.lower()
        if name in {"manifest.json", "submission_template.json", "submission_template.jsonl"}:
            continue
        if "questions" in name and "conversation" not in name:
            continue

        if fp.suffix.lower() == ".jsonl":
            samples.extend(_load_jsonl(fp))
        else:
            samples.extend(_load_json(fp))
    return samples


def _load_json(fp: Path) -> list[ConversationSample]:
    data = json.loads(fp.read_text(encoding="utf-8"))
    return list(_iter_samples(data, fallback=fp.stem, source_path=str(fp)))


def _load_jsonl(fp: Path) -> list[ConversationSample]:
    out: list[ConversationSample] = []
    with fp.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sample = _sample_from_obj(obj, fallback_bucket=f"{fp.stem}:{i}", source_path=str(fp))
            if sample:
                out.append(sample)
    return out


def _iter_samples(data: Any, fallback: str, source_path: str | None) -> Iterable[ConversationSample]:
    if isinstance(data, list):
        for i, obj in enumerate(data):
            if isinstance(obj, dict):
                # Single-sample files → use file stem as bucket; multi → stem:index
                fb = fallback if len(data) == 1 else f"{fallback}:{i}"
                sample = _sample_from_obj(obj, fallback_bucket=fb, source_path=source_path)
                if sample:
                    yield sample
        return

    if isinstance(data, dict):
        # mapping of id -> sample
        if all(isinstance(v, dict) for v in data.values()) and any(
            ("conversation" in v or "session_1" in v or "sessions" in v) for v in data.values()
        ):
            for key, obj in data.items():
                sample = _sample_from_obj(obj, fallback_bucket=str(key), source_path=source_path)
                if sample:
                    yield sample
            return

        sample = _sample_from_obj(data, fallback_bucket=fallback, source_path=source_path)
        if sample:
            yield sample


def memory_id_for(bucket_id: str, session_index: int) -> str:
    """Stable idempotent id: {bucket}:session_{i}."""
    return f"{bucket_id}:session_{session_index}"
