"""离线契约测试：使用 hash 嵌入后端，无需下载模型。

运行：pytest -v
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def client(tmp_path):
    settings = Settings(
        embedding_backend="hash",
        chroma_path=str(tmp_path / "chroma"),
    )
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def _add(client, request_id, user_id, contents):
    ts = int(time.time() * 1000)
    return client.post("/aml/add", json={
        "request_id": request_id,
        "messages": [
            {"role": "user", "content": c, "timestamp": ts + i * 1000}
            for i, c in enumerate(contents)
        ],
        "user_id": user_id,
        "session_id": "s-1",
    })


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["auth_enabled"] is False


def test_add_echoes_contract_fields(client):
    r = _add(client, "req-1", "u1", ["I love hiking in the mountains."])
    assert r.status_code == 200
    body = r.json()
    assert body == {"success": True, "request_id": "req-1", "user_id": "u1", "session_id": "s-1"}


def test_search_returns_scored_items(client):
    _add(client, "req-2", "u1", ["I love hiking in the mountains.", "I work as a teacher."])
    r = client.post("/aml/search", json={"query": "hiking mountains", "user_id": "u1", "top_k": 10})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data, "应至少命中一条"
    assert set(data[0]) == {"id", "content", "score", "created_at"}
    assert data[0]["score"] >= data[-1]["score"]
    assert "hiking" in data[0]["content"]


def test_user_isolation(client):
    _add(client, "req-3", "u-secret", ["My passport number is XK123456."])
    _add(client, "req-4", "u-other", ["I like pizza."])
    r = client.post("/aml/search", json={"query": "passport number", "user_id": "u-other", "top_k": 10})
    assert all("XK123456" not in d["content"] for d in r.json()["data"])
    r = client.post("/aml/search", json={"query": "passport number", "user_id": "u-secret", "top_k": 10})
    assert any("XK123456" in d["content"] for d in r.json()["data"])


def test_idempotent_retry(client):
    payload = {
        "request_id": "req-dup",
        "messages": [{"role": "user", "content": "I moved to Porto.", "timestamp": 1726000000000}],
        "user_id": "u1",
        "session_id": "s-1",
    }
    assert client.post("/aml/add", json=payload).status_code == 200
    assert client.post("/aml/add", json=payload).status_code == 200  # 重试
    r = client.post("/aml/search", json={"query": "moved to Porto", "user_id": "u1", "top_k": 100})
    hits = [d for d in r.json()["data"] if "moved to Porto" in d["content"]]
    assert len(hits) == 1


def test_auth_when_key_configured(tmp_path):
    settings = Settings(
        embedding_backend="hash",
        chroma_path=str(tmp_path / "chroma"),
        memory_system_key="secret-key",
    )
    app = create_app(settings)
    with TestClient(app) as c:
        assert c.post("/aml/search", json={"query": "x", "user_id": "u", "top_k": 1}).status_code == 401
        r = c.post(
            "/aml/search",
            json={"query": "x", "user_id": "u", "top_k": 1},
            headers={"Authorization": "Bearer secret-key"},
        )
        assert r.status_code == 200
