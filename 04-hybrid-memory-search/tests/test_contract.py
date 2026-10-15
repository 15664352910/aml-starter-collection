"""Hybrid Memory Search (#4) 契约测试。

嵌入 API 用确定性哈希向量打桩（无需真实 API key）；BM25 通道为原始实现。
运行：pytest -v
"""

from __future__ import annotations

import hashlib
import math
import os
import time

_STORE_DIR = os.path.join(os.path.dirname(__file__), ".test-data", "store")
import shutil
if os.path.exists(_STORE_DIR):
    shutil.rmtree(_STORE_DIR)  # 清理上次运行残留
os.makedirs(_STORE_DIR, exist_ok=True)
os.environ["MEMORY_STORE_DIR"] = _STORE_DIR
os.environ["OPENAI_EMBEDDING_API_BASE"] = "http://127.0.0.1:9/v1"  # 不会真正请求
os.environ["OPENAI_EMBEDDING_API_KEY"] = "test-key"
os.environ["OPENAI_EMBEDDING_MODEL"] = "hash-stub"
os.environ["OPENAI_EMBEDDING_DIMENSIONS"] = "256"
os.environ.pop("API_TOKEN", None)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.memory.embedding import EmbeddingClient  # noqa: E402


def _hash_embed(texts, dims=256):  # 与 aml-memory-starter 的 HashEmbedder 同思路
    vec = np.zeros((len(texts), dims), dtype=np.float32)
    for row, text in enumerate(texts):
        norm = " ".join(text.lower().split())
        for i in range(max(len(norm) - 2, 0) + (1 if norm else 0)):
            gram = norm[i : i + 3]
            h = int.from_bytes(hashlib.md5(gram.encode("utf-8")).digest()[:4], "big")
            vec[row, h % dims] += 1.0
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vec / norms


# 打桩嵌入：测试不需要真实嵌入服务
EmbeddingClient.embed_batch = lambda self, texts, batch_size=16: _hash_embed(texts)

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
    assert client.get("/health").status_code == 200


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
