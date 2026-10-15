"""ActiveMemoryIndex (#3) 契约测试。

无 LLM（AMI_EXTRACT=0 + 不设 key），嵌入用确定性哈希向量打桩。
运行：pytest -v
"""

from __future__ import annotations

import hashlib
import os
import time

_DB_DIR = os.path.join(os.path.dirname(__file__), ".test-data")
os.makedirs(_DB_DIR, exist_ok=True)
if os.path.exists(os.path.join(_DB_DIR, "ami.db")):
    os.remove(os.path.join(_DB_DIR, "ami.db"))  # 清理上次运行残留
os.environ["AMI_DB_PATH"] = os.path.join(_DB_DIR, "ami.db")
os.environ["AMI_EXTRACT"] = "0"          # Add 不做 LLM 事实抽取
os.environ.pop("AMI_LLM_API_KEY", None)  # LLM 功能全部走 raw-text 回退
os.environ["AMI_REQUEST_LOG"] = ""       # 不写请求日志
os.environ["AMI_AUTH_SCHEME"] = "none"   # 本地测试关闭认证（部署时应设 bearer + AMI_AUTH_TOKEN）

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import embed  # noqa: E402

DIMS = 1024


def _hash_encode(texts, *, is_query=False):
    vec = np.zeros((len(texts), DIMS), dtype=np.float32)
    for row, text in enumerate(texts):
        norm = " ".join(text.lower().split())
        for i in range(max(len(norm) - 2, 0) + (1 if norm else 0)):
            h = int.from_bytes(hashlib.md5(norm[i : i + 3].encode("utf-8")).digest()[:4], "big")
            vec[row, h % DIMS] += 1.0
    norms = np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)
    return vec / norms


embed.encode = _hash_encode  # 打桩：测试无需真实嵌入模型
embed.dim = lambda: DIMS
embed.embedding_identity = lambda: {"backend": "hash", "model": "hash-stub", "dimensions": DIMS}

from aml_entry import app  # noqa: E402


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
