from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def settings(tmp_path: Path, api_key: str | None = None) -> Settings:
    return Settings(
        database_path=tmp_path / "memory.db",
        api_key=api_key,
        llm_mode="off",
        openai_api_key=None,
        openai_base_url="https://api.openai.com/v1",
        openai_model="gpt-4o-mini",
        openai_timeout_seconds=1,
        candidate_limit=100,
    )


def required_llm_settings(tmp_path: Path) -> Settings:
    base = settings(tmp_path)
    return Settings(
        database_path=base.database_path,
        api_key=None,
        llm_mode="required",
        openai_api_key=None,
        openai_base_url=base.openai_base_url,
        openai_model=base.openai_model,
        openai_timeout_seconds=base.openai_timeout_seconds,
        candidate_limit=base.candidate_limit,
    )


def add_payload(user_id: str = "user-a") -> dict:
    return {
        "request_id": f"req-{user_id}",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "My favorite drink is jasmine tea.",
            },
            {
                "role": "assistant",
                "timestamp": 1704067260000,
                "content": "I will remember that you prefer jasmine tea.",
            },
        ],
        "user_id": user_id,
        "session_id": "session-1",
    }


def test_health_add_and_search_contract(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["version"] == "0.3.0"

    payload = add_payload()
    added = client.post("/add", json=payload)
    assert added.status_code == 200
    assert added.json() == {
        "success": True,
        "request_id": payload["request_id"],
        "user_id": payload["user_id"],
        "session_id": payload["session_id"],
    }

    searched = client.post(
        "/search",
        json={"query": "What drink do I prefer?", "user_id": "user-a", "top_k": 5},
    )
    assert searched.status_code == 200
    result = searched.json()
    assert set(result) == {"data"}
    assert result["data"]
    assert "jasmine tea" in result["data"][0]["content"]
    assert set(result["data"][0]) <= {"id", "content", "score", "created_at"}


def test_idempotent_add(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    payload = add_payload()
    assert client.post("/add", json=payload).status_code == 200
    assert client.post("/add", json=payload).status_code == 200

    result = client.post(
        "/search",
        json={"query": "jasmine tea", "user_id": "user-a", "top_k": 100},
    ).json()
    assert len(result["data"]) == 2


def test_user_scope_isolation(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    assert client.post("/add", json=add_payload("user-a")).status_code == 200
    assert client.post("/add", json=add_payload("user-b")).status_code == 200

    result = client.post(
        "/search",
        json={"query": "jasmine tea", "user_id": "missing-user", "top_k": 10},
    ).json()
    assert result == {"data": []}


def test_optional_authentication(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path, api_key="secret")))
    assert client.post("/add", json=add_payload()).status_code == 401
    assert client.post("/add", json=add_payload(), headers={"X-Api-Key": "secret"}).status_code == 200


def test_rejects_extra_fields_and_top_k_over_100(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    payload = add_payload()
    payload["metadata"] = {"not": "supported"}
    assert client.post("/add", json=payload).status_code == 422

    response = client.post(
        "/search",
        json={"query": "anything", "user_id": "user-a", "top_k": 101},
    )
    assert response.status_code == 422


def test_add_accepts_protocol_valid_blank_source_message(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    payload = add_payload()
    payload["request_id"] = "blank-source"
    payload["messages"] = [{"role": "assistant", "content": "", "timestamp": None}]
    assert client.post("/add", json=payload).status_code == 200


def test_required_llm_mode_fails_closed_without_credential(tmp_path: Path) -> None:
    client = TestClient(create_app(required_llm_settings(tmp_path)))
    health = client.get("/health").json()
    assert health["llm_mode"] == "required"
    assert health["llm_ready"] is False
    response = client.post("/add", json=add_payload())
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def add_memories(client: TestClient, messages: list[dict], user_id: str = "ranking-user") -> None:
    for index, message in enumerate(messages):
        response = client.post(
            "/add",
            json={
                "request_id": f"ranking-{index}",
                "messages": [message],
                "user_id": user_id,
                "session_id": f"session-{index}",
            },
        )
        assert response.status_code == 200


def test_latest_and_earliest_preferences_use_source_time(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    add_memories(client, [
        {"role": "user", "timestamp": 1704067200000, "content": "My favorite drink is coffee."},
        {"role": "user", "timestamp": 1735689600000, "content": "Actually, I changed my favorite drink to jasmine tea."},
    ])

    latest = client.post(
        "/search",
        json={"query": "What is my latest favorite drink?", "user_id": "ranking-user", "top_k": 1},
    ).json()["data"][0]
    earliest = client.post(
        "/search",
        json={"query": "What was my first favorite drink?", "user_id": "ranking-user", "top_k": 1},
    ).json()["data"][0]

    assert "jasmine tea" in latest["content"]
    assert "2025-01-01" in latest["content"]
    assert "coffee" in earliest["content"]
    assert "2024-01-01" in earliest["content"]


def test_multi_hop_retrieval_returns_complementary_evidence(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    add_memories(client, [
        {"role": "user", "timestamp": 1704067200000, "content": "Alice's mentor is Bob."},
        {"role": "assistant", "timestamp": 1704153600000, "content": "Bob lives in Kyoto."},
        {"role": "user", "timestamp": 1704240000000, "content": "Alice likes jazz music."},
    ])

    result = client.post(
        "/search",
        json={"query": "Where does Alice's mentor live?", "user_id": "ranking-user", "top_k": 2},
    ).json()["data"]
    evidence = " ".join(item["content"] for item in result)
    assert "mentor is Bob" in evidence
    assert "Bob lives in Kyoto" in evidence
    assert [item["score"] for item in result] == sorted(
        [item["score"] for item in result], reverse=True
    )


def test_choice_options_help_retrieve_exact_quantity(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    add_memories(client, [
        {"role": "assistant", "timestamp": 1704067200000, "content": "The access token expires after 24 hours."},
        {"role": "assistant", "timestamp": 1704153600000, "content": "The backup job runs every 12 hours."},
    ])
    result = client.post(
        "/search",
        json={
            "query": "How long does the access token last?",
            "options": ["12 hours", "24 hours", "48 hours"],
            "user_id": "ranking-user",
            "top_k": 1,
        },
    ).json()["data"][0]
    assert "24 hours" in result["content"]


def test_unrelated_query_does_not_backfill_private_recent_memory(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    assert client.post("/add", json=add_payload()).status_code == 200
    result = client.post(
        "/search",
        json={
            "query": "What is the launch authorization code?",
            "user_id": "user-a",
            "top_k": 100,
        },
    ).json()
    assert result == {"data": []}


def test_conversation_window_includes_reply_without_crossing_session(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    first = {
        "request_id": "window-first",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "Which city did the conference select?",
            },
            {
                "role": "assistant",
                "timestamp": 1704067200001,
                "content": "The conference selected Reykjavík.",
            },
        ],
        "user_id": "window-user",
        "session_id": "session-a",
    }
    second = {
        "request_id": "window-second",
        "messages": [
            {
                "role": "assistant",
                "timestamp": 1704067200002,
                "content": "A private unrelated note from another session.",
            }
        ],
        "user_id": "window-user",
        "session_id": "session-b",
    }
    assert client.post("/add", json=first).status_code == 200
    assert client.post("/add", json=second).status_code == 200
    result = client.post(
        "/search",
        json={"query": "Where was the conference?", "user_id": "window-user", "top_k": 1},
    ).json()["data"][0]["content"]
    assert "Reykjavík" in result
    assert "private unrelated note" not in result


def test_window_repairs_out_of_order_concurrent_chunks(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    common = {"user_id": "chunk-user", "session_id": "shared-session"}
    chunk_one = {
        **common,
        "request_id": "eval:sample:chunk-1",
        "messages": [{"role": "assistant", "content": "The answer is zebra."}],
    }
    chunk_zero = {
        **common,
        "request_id": "eval:sample:chunk-0",
        "messages": [{"role": "user", "content": "The checkpoint question is ready."}],
    }
    # Simulate concurrent ingestion completing in the opposite source order.
    assert client.post("/add", json=chunk_one).status_code == 200
    assert client.post("/add", json=chunk_zero).status_code == 200
    content = client.post(
        "/search",
        json={
            "query": "What answer follows the checkpoint question?",
            "user_id": "chunk-user",
            "top_k": 1,
        },
    ).json()["data"][0]["content"]
    assert content.index("checkpoint question") < content.index("answer is zebra")
