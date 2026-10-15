"""Turn an Add payload's messages into indexable memory units.

Two things drive the design here, both from the platform side:

1. ``content`` is handed verbatim to the platform's answer model, and that
   model's prompt resolves relative times ("yesterday", "last month") *using the
   memory timestamp*. A unit whose text does not carry its own timestamp is
   useless for the time/event-sequence dimension, so the timestamp is baked into
   the text rather than left only in the ``created_at`` field.
2. The same prompt attributes memories per speaker, so the role is baked in too.
"""

from __future__ import annotations

import datetime as _dt
import re

from . import config
from .models import Message

_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


def format_timestamp(ms: int | None) -> str | None:
    """Unix milliseconds -> ``YYYY-MM-DD HH:MM`` (UTC), or None."""
    if ms is None:
        return None
    try:
        # Tolerate seconds-precision inputs that skipped the ms convention.
        seconds = ms / 1000.0 if abs(ms) > 1e11 else float(ms)
        return _dt.datetime.fromtimestamp(seconds, _dt.timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
    except (OverflowError, OSError, ValueError):
        return None


def iso_timestamp(ms: int | None) -> str | None:
    """Unix milliseconds -> ISO 8601 UTC, for the optional ``created_at`` field."""
    if ms is None:
        return None
    try:
        seconds = ms / 1000.0 if abs(ms) > 1e11 else float(ms)
        return (
            _dt.datetime.fromtimestamp(seconds, _dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (OverflowError, OSError, ValueError):
        return None


def _split_long(text: str, limit: int) -> list[str]:
    """Split over-long text on sentence boundaries, falling back to hard cuts."""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_END.split(text):
        if not sentence:
            continue
        if buffer and len(buffer) + len(sentence) + 1 > limit:
            pieces.append(buffer)
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer:
        pieces.append(buffer)
    # A single sentence can still exceed the limit.
    out: list[str] = []
    for piece in pieces:
        while len(piece) > limit:
            out.append(piece[:limit])
            piece = piece[limit:]
        if piece:
            out.append(piece)
    return out or [text[:limit]]


def render(role: str, timestamp_ms: int | None, text: str) -> str:
    """Compose the string the answer model will actually read."""
    stamp = format_timestamp(timestamp_ms)
    prefix = f"[{stamp}] {role}:" if stamp else f"{role}:"
    return f"{prefix} {text}".strip()


def build_units(messages: list[Message]) -> list[tuple[str, int | None, str]]:
    """Messages -> ``(role, timestamp_ms, text)`` units.

    Consecutive same-role messages are merged, and anything shorter than
    ``MIN_UNIT_CHARS`` is merged forward, so no unit is a context-free fragment
    like "yes" or "ok". The timestamp kept is the *first* in each merged group,
    which is the one that anchors any relative time expression inside it.
    """
    cleaned = [
        (m.role or "user", m.timestamp, m.content.strip())
        for m in messages
        if m.content and m.content.strip()
    ]
    if not cleaned:
        return []

    merged: list[tuple[str, int | None, str]] = []
    for role, timestamp, text in cleaned:
        if merged:
            prev_role, prev_ts, prev_text = merged[-1]
            same_role = config.MERGE_SAME_ROLE and prev_role == role
            too_short = len(prev_text) < config.MIN_UNIT_CHARS
            if (same_role or too_short) and len(prev_text) + len(text) + 1 <= config.MAX_UNIT_CHARS:
                merged[-1] = (
                    prev_role if too_short and not same_role else role,
                    prev_ts if prev_ts is not None else timestamp,
                    f"{prev_text} {text}".strip(),
                )
                continue
        merged.append((role, timestamp, text))

    units: list[tuple[str, int | None, str]] = []
    for role, timestamp, text in merged:
        for piece in _split_long(text, config.MAX_UNIT_CHARS):
            units.append((role, timestamp, piece))
    return units
