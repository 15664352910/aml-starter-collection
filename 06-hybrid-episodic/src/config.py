"""Environment-driven configuration.

Every knob is an env var so the platform can run the published image unmodified
and we can still tune without a rebuild.
"""

from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ── Auth ────────────────────────────────────────────────────────────────────
# AUTH_MODE:
#   auto        (default) — if API_KEY is set, accept the secret via any of
#                 "Authorization: Bearer", "Authorization: Token" or "X-Api-Key".
#                 If API_KEY is unset, allow unauthenticated access.
#   bearer / token / x-api-key — require that exact scheme.
#   none        — no authentication at all (the spec permits this for public smoke
#                 only, so it is not the default).
#
# 'auto' exists because on the code-submission route the platform controls the
# deployment and picks the scheme from its own dropdown. Requiring an exact match
# to a mode we guessed in advance would turn a scheme mismatch into a 401 during
# contract validation — and 401 is not retried.
AUTH_MODE = os.environ.get("AUTH_MODE", "auto").strip().lower()
API_KEY = os.environ.get("API_KEY", "")

# ── Embedding backend ───────────────────────────────────────────────────────
# On by default: the local encoder is the dense half of the submitted retrieval
# core. Set false to run purely lexical (BM25 only), which needs neither torch nor
# a model download and puts zero models in the Add/Search path.
EMBED_ENABLED = _bool("EMBED_ENABLED", True)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_BATCH = _int("EMBED_BATCH", 64)
# bge models expect this prefix on the *query* side only.
EMBED_QUERY_PREFIX = os.environ.get(
    "EMBED_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
)

# ── Indexing ────────────────────────────────────────────────────────────────
# How long a duplicate Add waits for the in-flight original to land before giving
# up and asking the platform to retry. Comfortably under the platform's 1,200s Add
# timeout so our answer arrives before it stops listening.
DUPLICATE_WAIT_SECONDS = _float("DUPLICATE_WAIT_SECONDS", 300.0)
# Messages shorter than this get merged forward so no unit is context-free.
MIN_UNIT_CHARS = _int("MIN_UNIT_CHARS", 24)
# Cap on a single indexed unit; longer messages are split on sentence bounds.
MAX_UNIT_CHARS = _int("MAX_UNIT_CHARS", 1200)
# Merge consecutive messages from the same role into one unit.
MERGE_SAME_ROLE = _bool("MERGE_SAME_ROLE", True)

# ── Retrieval ───────────────────────────────────────────────────────────────
# Primary tuning knob. top_k from the platform is a ceiling (100 in full runs),
# not a requirement — this caps what we actually return.
MAX_RETURN = _int("MAX_RETURN", 100)
# Reciprocal-rank-fusion damping.
RRF_K = _float("RRF_K", 60.0)
# Per-channel candidate depth before fusion.
CANDIDATE_DEPTH = _int("CANDIDATE_DEPTH", 400)
# Weight on the option-augmented query channel (multiple-choice items only).
OPTION_CHANNEL_WEIGHT = _float("OPTION_CHANNEL_WEIGHT", 0.6)
# Mild recency preference, applied as a rank-space nudge. 0 disables.
RECENCY_WEIGHT = _float("RECENCY_WEIGHT", 0.15)

# ── BM25 ────────────────────────────────────────────────────────────────────
BM25_K1 = _float("BM25_K1", 1.5)
BM25_B = _float("BM25_B", 0.75)

# ── Mem0 extraction channel — NOT ENABLED IN THIS SUBMISSION ────────────────
# Implemented and unit-tested (tests/test_fusion.py, mem0 stubbed) but never
# validated against a live gpt-4o-mini endpoint, so it ships off. Turning it on
# requires OPENAI_API_KEY and changes the declared model set — see README.
# Third-party, Apache-2.0: https://github.com/mem0ai/mem0
MEM0_ENABLED = _bool("MEM0_ENABLED", False)
# Pinned by the Academic division rule: the model used inside Add/Search must be
# gpt-4o-mini. Overridable only so the Commercial board or local mocks can differ.
MEM0_LLM_MODEL = os.environ.get("MEM0_LLM_MODEL", "gpt-4o-mini")
MEM0_EMBEDDER_PROVIDER = os.environ.get("MEM0_EMBEDDER_PROVIDER", "openai")
MEM0_EMBEDDER_MODEL = os.environ.get("MEM0_EMBEDDER_MODEL", "text-embedding-3-small")
# faiss / chroma / qdrant all run embedded — no sidecar container.
MEM0_VECTOR_STORE = os.environ.get("MEM0_VECTOR_STORE", "faiss")
MEM0_VECTOR_PATH = os.environ.get("MEM0_VECTOR_PATH", "/tmp/mem0_vectors")
MEM0_COLLECTION = os.environ.get("MEM0_COLLECTION", "amc_memories")
MEM0_HISTORY_DB = os.environ.get("MEM0_HISTORY_DB", "/tmp/mem0_history.db")
# mem0 defaults to 0.1, which drops weak-but-correct evidence before we can fuse.
MEM0_SEARCH_THRESHOLD = _float("MEM0_SEARCH_THRESHOLD", 0.0)
# Fusion weight of the extracted-fact channel relative to the raw-turn channels.
MEM0_CHANNEL_WEIGHT = _float("MEM0_CHANNEL_WEIGHT", 1.0)
# Return 503 from /add when extraction fails, so the platform's Add retry applies
# instead of us reporting success for a memory that was never written.
MEM0_FAIL_ADD_ON_ERROR = _bool("MEM0_FAIL_ADD_ON_ERROR", True)
