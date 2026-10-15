"""InvMem (#1) 契约测试。

嵌入用确定性哈希向量打桩（无需下载模型），检索为原始 hybrid（BM25+dense/RRF）实现。
运行：pytest -v
"""

from __future__ import annotations

import hashlib
import os
import time

_DB_DIR = os.path.join(os.path.dirname(__file__), ".test-data")
os.makedirs(_DB_DIR, exist_ok=True)
if os.path.exists(os.path.join(_DB_DIR, "memory.db")):
    os.remove(os.path.join(_DB_DIR, "memory.db"))  # 清理上次运行残留
os.environ["MEMORY_DB_PATH"] = os.path.join(_DB_DIR, "memory.db")
os.environ["ALLOW_UNAUTHENTICATED"] = "true"   # 本地测试免认证
os.environ["SKIP_MODEL_STARTUP"] = "true"      # 不打不存在的模型

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import src.vanilla_rag.api as vanilla_api  # noqa: E402

DIMS = 384


class HashEmbedder:
    """实现 Embedder 协议（encode_documents / encode_queries）。"""

    identity = "hash-stub"

    @staticmethod
    def _encode(texts):
        vec = np.zeros((len(texts), DIMS), dtype=np.float32)
        for row, text in enumerate(texts):
            norm = " ".join(text.lower().split())
            for i in range(max(len(norm) - 2, 0) + (1 if norm else 0)):
                h = int.from_bytes(hashlib.md5(norm[i : i + 3].encode("utf-8")).digest()[:4], "big")
                vec[row, h % DIMS] += 1.0
        norms = np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)
        return vec / norms

    def encode_documents(self, texts):
        return self._encode(texts)

    def encode_queries(self, texts):
        return self._encode(texts)


# 打桩嵌入：在 get_store 首次调用前替换
vanilla_api.build_embedder = lambda *args, **kwargs: HashEmbedder()

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
