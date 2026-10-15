from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    embedding_api_base: str
    embedding_api_key: str
    embedding_model: str
    embedding_dimensions: int
    hybrid_alpha: float
    store_dir: Path
    openai_api_base: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None


def load_settings(env_path: str | Path | None = None) -> Settings:
    """Load API / hybrid settings from .env (or process env)."""
    if env_path is None:
        # project root: .../hybird_search/.env
        env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path, override=False)

    store = os.getenv("MEMORY_STORE_DIR", "store")
    return Settings(
        embedding_api_base=os.environ["OPENAI_EMBEDDING_API_BASE"].rstrip("/"),
        embedding_api_key=os.environ["OPENAI_EMBEDDING_API_KEY"],
        embedding_model=os.environ["OPENAI_EMBEDDING_MODEL"],
        embedding_dimensions=int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1024")),
        hybrid_alpha=float(os.getenv("HYBRID_ALPHA", "0.5")),
        store_dir=Path(store),
        openai_api_base=(os.getenv("OPENAI_API_BASE") or None),
        openai_api_key=(os.getenv("OPENAI_API_KEY") or None),
        openai_model=(os.getenv("OPENAI_MODEL") or None),
    )
