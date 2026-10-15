"""运行时配置：全部通过环境变量覆盖，见 .env.example。"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # 认证：设置后所有 /aml/* 请求必须携带 Authorization: Bearer <key>
    memory_system_key: str = ""
    # 向量后端：sentence-transformers（默认，效果好）或 hash（纯本地零依赖，演示/测试用）
    embedding_backend: str = "st"
    embedding_model: str = "all-MiniLM-L6-v2"
    # 存储路径
    chroma_path: str = "./chroma_data"
    # 写入时语义去重阈值（余弦相似度），1.0 表示关闭去重
    dedup_threshold: float = 0.95
    # 检索弃权阈值：得分低于该值的结果被丢弃；返回空数组 = 平台视为弃权。
    # 默认 0.0 表示不弃权（上榜系统多为此行为，简单可靠）。
    abstention_threshold: float = 0.0


def load_settings() -> Settings:
    return Settings(
        memory_system_key=os.getenv("AML_MEMORY_SYSTEM_KEY", ""),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "st"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        chroma_path=os.getenv("MEMORY_DB_PATH", "./chroma_data"),
        dedup_threshold=float(os.getenv("DEDUP_THRESHOLD", "0.95")),
        abstention_threshold=float(os.getenv("ABSTENTION_THRESHOLD", "0.0")),
    )
