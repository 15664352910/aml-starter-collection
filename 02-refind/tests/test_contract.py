"""ReFind (#2) 契约测试：bm25 模式，无需 LLM。运行：pytest -v（在本目录下）"""

from __future__ import annotations

import os
import time

_DB_DIR = os.path.join(os.path.dirname(__file__), ".test-data")
os.makedirs(_DB_DIR, exist_ok=True)
if os.path.exists(os.path.join(_DB_DIR, "refind.sqlite3")):
    os.remove(os.path.join(_DB_DIR, "refind.sqlite3"))  # 清理上次运行残留
os.environ["DATABASE_PATH"] = os.path.join(_DB_DIR, "refind.sqlite3")
os.environ["RETRIEVAL_MODE"] = "bm25"  # 免 LLM 的纯检索模式

import pytest
from fastapi.testclient import TestClient

from aml_entry import app


@pytest.fixture()
def client():
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
    assert r.json()["status"] == "ok"


def test_add_echoes_contract_fields(client):
    r = _add(client, "req-1", "u1", ["I love hiking in the mountains."])
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["request_id"] == "req-1"
    assert body["user_id"] == "u1"
    assert body["session_id"] == "s-1"


def test_search_and_isolation(client):
    _add(client, "req-2", "u-secret", ["My passport number is XK123456."])
    _add(client, "req-3", "u-other", ["I like pizza very much."])
    r = client.post("/aml/search", json={"query": "passport number", "user_id": "u-other", "top_k": 10})
    assert all("XK123456" not in d["content"] for d in r.json()["data"])
    r = client.post("/aml/search", json={"query": "passport", "user_id": "u-secret", "top_k": 10})
    assert any("XK123456" in d["content"] for d in r.json()["data"])


def test_idempotent_retry(client):
    payload = {
        "request_id": "req-dup",
        "messages": [{"role": "user", "content": "I moved to Porto.", "timestamp": 1726000000000}],
        "user_id": "u1",
        "session_id": "s-1",
    }
    assert client.post("/aml/add", json=payload).status_code == 200
    assert client.post("/aml/add", json=payload).status_code == 200
    r = client.post("/aml/search", json={"query": "moved to Porto", "user_id": "u1", "top_k": 100})
    hits = [d for d in r.json()["data"] if "moved to Porto" in d["content"]]
    assert len(hits) == 1
