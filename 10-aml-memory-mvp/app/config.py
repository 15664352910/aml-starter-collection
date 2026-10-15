from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_path: Path
    api_key: str | None
    llm_mode: str
    openai_api_key: str | None
    openai_base_url: str
    openai_model: str
    openai_timeout_seconds: float
    candidate_limit: int
    max_output_chars: int = 240_000

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.getenv("AML_LLM_MODE", "off").strip().lower()
        if mode not in {"off", "best_effort", "required"}:
            raise ValueError("AML_LLM_MODE must be off, best_effort, or required")
        return cls(
            database_path=Path(os.getenv("AML_DATABASE_PATH", "data/memory.db")),
            api_key=os.getenv("AML_API_KEY") or None,
            llm_mode=mode,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90")),
            candidate_limit=max(100, int(os.getenv("AML_CANDIDATE_LIMIT", "400"))),
            max_output_chars=max(32_000, int(os.getenv("AML_MAX_OUTPUT_CHARS", "240000"))),
        )
