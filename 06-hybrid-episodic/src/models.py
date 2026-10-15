"""Contract models — mirror the official Add/Search spec field-for-field.

Reference: docs/API_CONTRACT.md sections 05.
Extra fields are tolerated (pydantic ignores unknown keys by default) so a
platform-side additive change cannot 422 us mid-run.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    """One source message. ``timestamp`` is Unix milliseconds when present."""

    model_config = ConfigDict(extra="ignore")

    role: str
    content: str
    timestamp: int | None = None


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str
    messages: list[Message]
    user_id: str
    session_id: str


class AddResponse(BaseModel):
    """``success`` must be boolean true and all three IDs must echo the request.

    The platform fails the stage immediately on any deviation, even with HTTP 200.
    """

    success: bool
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str
    user_id: str
    top_k: int
    # Present for multiple-choice items, absent for open-ended ones.
    options: list[str] | None = None


class MemoryItem(BaseModel):
    """``content`` is handed straight to the platform's answer model."""

    id: str
    content: str
    score: float | None = None
    created_at: str | None = None


class SearchResponse(BaseModel):
    """No ``items`` wrapper, and never a bare top-level array."""

    data: list[MemoryItem]
