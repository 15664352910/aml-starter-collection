from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings, intentionally sourced only from environment variables."""

    database_path: Path
    retrieval_mode: str
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_timeout_seconds: float
    llm_max_tokens: int
    agent_max_iterations: int
    search_top_k: int
    context_window: int
    session_rrf: bool
    retention_days: int

    @classmethod
    def from_env(cls) -> Settings:
        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        mode = os.getenv("RETRIEVAL_MODE", "agent").strip().lower()
        if mode not in {"agent", "bm25"}:
            raise ValueError("RETRIEVAL_MODE must be 'agent' or 'bm25'")

        settings = cls(
            database_path=Path(os.getenv("DATABASE_PATH", "./data/refind.sqlite3")),
            retrieval_mode=mode,
            llm_api_key=api_key,
            llm_base_url=os.getenv(
                "LLM_BASE_URL", "https://openrouter.ai/api/v1"
            ).rstrip("/"),
            llm_model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "900")),
            agent_max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "4")),
            search_top_k=int(os.getenv("SEARCH_TOP_K", "5")),
            context_window=int(os.getenv("CONTEXT_WINDOW", "2")),
            session_rrf=_env_bool("SESSION_RRF", True),
            retention_days=int(os.getenv("RETENTION_DAYS", "30")),
        )
        if settings.agent_max_iterations < 1:
            raise ValueError("AGENT_MAX_ITERATIONS must be at least 1")
        if settings.search_top_k < 1:
            raise ValueError("SEARCH_TOP_K must be at least 1")
        if settings.context_window < 0:
            raise ValueError("CONTEXT_WINDOW cannot be negative")
        if settings.retention_days < 1 or settings.retention_days > 30:
            raise ValueError("RETENTION_DAYS must be between 1 and 30")
        return settings
