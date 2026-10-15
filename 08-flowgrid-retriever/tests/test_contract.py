"""FlowGrid AML Retriever (#8) 契约测试：路径走配置，无需改码。运行：pytest -v"""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from aml_retriever.config import RetrieverConfig
from aml_retriever.server import RetrieverServer


@pytest.fixture()
def server(tmp_path):
    config = RetrieverConfig(
        db_path=str(tmp_path / "flowgrid.db"),
        add_path="/aml/add",
        search_path="/aml/search",
        auth_mode="none",
        host="127.0.0.1",
        port=0,
    )
    httpd = RetrieverServer(config, quiet=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    thread.join(timeout=5)


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


def test_health(server):
    r = httpx.get(f"{server}/health", timeout=10)
    assert r.status_code == 200


def test_add_echoes_contract_fields(server):
    r = _add(httpx.Client(base_url=server, timeout=10), "req-1", "u1",
             ["I love hiking in the mountains."])
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["request_id"] == "req-1"
    assert body["user_id"] == "u1"
    assert body["session_id"] == "s-1"


def test_search_and_isolation(server):
    with httpx.Client(base_url=server, timeout=10) as c:
        _add(c, "req-2", "u-secret", ["My passport number is XK123456."])
        _add(c, "req-3", "u-other", ["I like pizza very much."])
        r = c.post("/aml/search", json={"query": "passport number", "user_id": "u-other", "top_k": 10})
        assert all("XK123456" not in d["content"] for d in r.json()["data"])
        r = c.post("/aml/search", json={"query": "passport", "user_id": "u-secret", "top_k": 10})
        assert any("XK123456" in d["content"] for d in r.json()["data"])


def test_idempotent_retry(server):
    with httpx.Client(base_url=server, timeout=10) as c:
        payload = {
            "request_id": "req-dup",
            "messages": [{"role": "user", "content": "I moved to Porto.", "timestamp": 1726000000000}],
            "user_id": "u1",
            "session_id": "s-1",
        }
        assert c.post("/aml/add", json=payload).status_code == 200
        assert c.post("/aml/add", json=payload).status_code == 200
        r = c.post("/aml/search", json={"query": "moved to Porto", "user_id": "u1", "top_k": 100})
        # FlowGrid 会同时返回原始消息与聚合视图（视图带 "user: " 前缀），只数原始消息
        hits = [d for d in r.json()["data"] if d["content"] == "I moved to Porto."]
        assert len(hits) == 1
