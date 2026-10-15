from __future__ import annotations

from fastapi.testclient import TestClient

from app.bm25 import BM25Index
from app.config import Settings
from app.llm import LLMReply
from app.main import create_app
from app.retriever import ReFindRetriever
from app.store import MemoryChunk, MemoryStore


def settings(tmp_path, mode: str = "bm25") -> Settings:
    return Settings(
        database_path=tmp_path / "refind.sqlite3",
        retrieval_mode=mode,
        llm_api_key="",
        llm_base_url="https://example.invalid/v1",
        llm_model="openai/gpt-4o-mini",
        llm_timeout_seconds=1,
        llm_max_tokens=200,
        agent_max_iterations=4,
        search_top_k=5,
        context_window=1,
        session_rrf=True,
        retention_days=30,
    )


def add_payload(
    request_id: str = "run:chunk-0",
    user_id: str = "run:user-0",
    session_id: str = "run:session-0",
    content: str = "I adopted a cat named Luna.",
    role: str = "user",
    timestamp: int | None = 1704067200000,
) -> dict:
    return {
        "request_id": request_id,
        "messages": [{"role": role, "timestamp": timestamp, "content": content}],
        "user_id": user_id,
        "session_id": session_id,
    }


def test_add_is_synchronous_idempotent_and_searchable(tmp_path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    payload = add_payload()

    first = client.post("/v1/memories/add", json=payload)
    retry = client.post("/v1/memories/add", json=payload)
    search = client.post(
        "/v1/memories/search",
        json={
            "query": "What is the cat's name?",
            "user_id": payload["user_id"],
            "top_k": 5,
        },
    )

    expected = {
        "success": True,
        "request_id": payload["request_id"],
        "user_id": payload["user_id"],
        "session_id": payload["session_id"],
    }
    assert first.status_code == 200 and first.json() == expected
    assert retry.status_code == 200 and retry.json() == expected
    assert search.status_code == 200
    assert len(search.json()["data"]) == 1
    assert "Luna" in search.json()["data"][0]["content"]


def test_conflicting_request_id_is_rejected(tmp_path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    assert client.post("/v1/memories/add", json=add_payload()).status_code == 200
    conflict = client.post(
        "/v1/memories/add", json=add_payload(content="This is a different payload.")
    )
    assert conflict.status_code == 409


def test_search_isolated_by_user_and_respects_top_k(tmp_path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    client.post("/v1/memories/add", json=add_payload(content="Luna likes salmon."))
    client.post(
        "/v1/memories/add",
        json=add_payload("run:chunk-1", content="Luna sleeps by the window."),
    )
    client.post(
        "/v1/memories/add",
        json=add_payload("other:chunk-0", "other:user", content="Secret cat Orion."),
    )

    response = client.post(
        "/v1/memories/search",
        json={"query": "cat Luna", "user_id": "run:user-0", "top_k": 1},
    )
    body = response.json()
    assert response.status_code == 200
    assert len(body["data"]) == 1
    assert "Orion" not in body["data"][0]["content"]


class ScriptedPlanner:
    def __init__(self) -> None:
        self.replies = iter(
            [
                'Thought: Search for the pet name.\nAction: search_chatrecord\nAction Input: {"keywords":["cat","name"]}',
                'Thought: The first result is relevant.\nAction: take_note\nAction Input: {"indices":[1]}',
                "Thought: Evidence is sufficient.\nAction: finish_search\nAction Input: {}",
            ]
        )

    def complete(self, messages: list[dict[str, str]]) -> LLMReply:
        return LLMReply(next(self.replies), prompt_tokens=10, completion_tokens=5)


def test_agent_mode_returns_selected_context_without_answer_generation(
    tmp_path,
) -> None:
    configured = settings(tmp_path, mode="agent")
    store = MemoryStore(configured.database_path)
    retriever = ReFindRetriever(store, configured, planner=ScriptedPlanner())
    client = TestClient(create_app(configured, store=store, retriever=retriever))
    client.post("/v1/memories/add", json=add_payload())
    response = client.post(
        "/v1/memories/search",
        json={"query": "What is my cat's name?", "user_id": "run:user-0", "top_k": 100},
    )
    assert response.status_code == 200
    assert response.json()["data"][0]["content"].endswith("Luna.")


def test_health_and_schema_validation(tmp_path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    health = client.get("/health")
    invalid = client.post(
        "/v1/memories/add", json={**add_payload(), "unexpected": "field"}
    )
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert invalid.status_code == 422


def test_session_rrf_preserves_conversation_rank_inside_one_session() -> None:
    chunks = [
        MemoryChunk(
            id="mem-0",
            user_id="user",
            session_id="one-session",
            request_id="request-0",
            sequence=1,
            messages=({"role": "user", "content": "alpha", "timestamp": None},),
            timestamp_ms=None,
            created_at="2026-08-07T00:00:00Z",
        ),
        MemoryChunk(
            id="mem-1",
            user_id="user",
            session_id="one-session",
            request_id="request-1",
            sequence=2,
            messages=(
                {"role": "user", "content": "alpha alpha alpha", "timestamp": None},
            ),
            timestamp_ms=None,
            created_at="2026-08-07T00:00:01Z",
        ),
    ]
    hits = BM25Index(chunks).search(
        keywords=["alpha"], top_k=2, context_window=0, session_rrf=True
    )
    assert [hit.anchor.id for hit in hits] == ["mem-1", "mem-0"]


def test_turn_adds_are_time_ordered_and_rebuilt_as_conversations(tmp_path) -> None:
    configured = settings(tmp_path)
    store = MemoryStore(configured.database_path)
    client = TestClient(create_app(configured, store=store))

    # Deliberately arrive in reverse order. Timestamp order must win over the
    # SQLite insertion sequence before adjacent turns are paired.
    client.post(
        "/v1/memories/add",
        json=add_payload(
            "run:chunk-1",
            content="The assistant confirms the name Luna.",
            role="assistant",
            timestamp=1704067201000,
        ),
    )
    client.post(
        "/v1/memories/add",
        json=add_payload(
            "run:chunk-0",
            content="What is my cat's name?",
            role="user",
            timestamp=1704067200000,
        ),
    )

    chunks = store.list_chunks("run:user-0")
    assert len(chunks) == 1
    assert [message["role"] for message in chunks[0].messages] == [
        "user",
        "assistant",
    ]
    assert chunks[0].messages[0]["content"] == "What is my cat's name?"
    assert chunks[0].messages[1]["content"].endswith("Luna.")


def test_equal_timestamp_turns_use_chunk_ordinal_and_keep_sessions_separate(
    tmp_path,
) -> None:
    configured = settings(tmp_path)
    store = MemoryStore(configured.database_path)
    client = TestClient(create_app(configured, store=store))
    timestamp = 1704067200000

    for ordinal, role, content in [
        (3, "assistant", "assistant three"),
        (1, "assistant", "assistant one"),
        (2, "user", "user two"),
        (0, "user", "user zero"),
    ]:
        client.post(
            "/v1/memories/add",
            json=add_payload(
                f"run:chunk-{ordinal}",
                content=content,
                role=role,
                timestamp=timestamp,
            ),
        )
    client.post(
        "/v1/memories/add",
        json=add_payload(
            "run:other:chunk-0",
            session_id="run:session-1",
            content="separate session",
            timestamp=timestamp,
        ),
    )

    chunks = store.list_chunks("run:user-0")
    assert len(chunks) == 3
    assert [message["content"] for message in chunks[0].messages] == [
        "user zero",
        "assistant one",
    ]
    assert [message["content"] for message in chunks[1].messages] == [
        "user two",
        "assistant three",
    ]
    assert chunks[2].session_id == "run:session-1"
