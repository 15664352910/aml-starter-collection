"""向量嵌入后端。

两种实现：
- StEmbedder：sentence-transformers 本地模型（默认，推荐参赛用）
- HashEmbedder：字符 n-gram 哈希向量，零依赖、确定性，适合快速演示和测试
"""

from __future__ import annotations

import hashlib
import math
from typing import List


class StEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


class HashEmbedder:
    """把文本映射到 512 维 n-gram 哈希向量并归一化。词面重叠多则相似度高。"""

    DIMS = 512

    def __init__(self) -> None:
        self.model_name = "hash-ngram"

    @staticmethod
    def _ngrams(text: str, n: int = 3) -> List[str]:
        text = " ".join(text.lower().split())
        if len(text) <= n:
            return [text] if text else []
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.DIMS
        for gram in self._ngrams(text):
            h = int.from_bytes(
                hashlib.md5(gram.encode("utf-8")).digest()[:4], "big"
            )
            vec[h % self.DIMS] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def build_embedder(backend: str, model_name: str):
    if backend == "hash":
        return HashEmbedder()
    return StEmbedder(model_name)
