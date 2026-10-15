"""AML Add/Search 契约入口。

- POST /aml/add    — 同步摄入 messages，request_id 幂等
- POST /aml/search — 向量检索，返回证据数组；空数组 = 弃权
- GET  /health     — 存活探针 + 简单统计

认证：设置环境变量 AML_MEMORY_SYSTEM_KEY 后，所有 /aml/* 请求必须携带
Authorization: Bearer <key>（兼容任意 scheme，只比对密钥本身）。
"""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request

from .config import Settings, load_settings
from .embedder import build_embedder
from .schemas import (
    AddRequest,
    AddResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from .store import MemoryStore, RequestLedger


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or load_settings()
    os.makedirs(settings.chroma_path, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        embedder = build_embedder(settings.embedding_backend, settings.embedding_model)
        ledger = RequestLedger(os.path.join(settings.chroma_path, "ledger.db"))
        app.state.store = MemoryStore(
            persist_path=settings.chroma_path,
            embedder=embedder,
            ledger=ledger,
            dedup_threshold=settings.dedup_threshold,
            abstention_threshold=settings.abstention_threshold,
        )
        yield

    app = FastAPI(
        title="AML Memory Starter",
        description="最小可参赛的 AML Add/Search 记忆服务（改编自榜单 #9 MemoryAgent）",
        version="1.0.0",
        lifespan=lifespan,
    )

    def authorize(request: Request) -> None:
        if not settings.memory_system_key:
            return  # 本地/测试模式：不启用认证
        header = request.headers.get("Authorization", "")
        parts = header.split(None, 1)
        token = parts[1].strip() if len(parts) == 2 else ""
        if not token or not secrets.compare_digest(token, settings.memory_system_key):
            raise HTTPException(status_code=401, detail="invalid memory system key")

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> dict:
        store: MemoryStore = request.app.state.store
        return {
            "status": "ok",
            **store.stats(),
            "embedding_backend": settings.embedding_backend,
            "auth_enabled": bool(settings.memory_system_key),
        }

    @app.post("/aml/add", response_model=AddResponse)
    def aml_add(payload: AddRequest, request: Request) -> AddResponse:
        authorize(request)
        store: MemoryStore = request.app.state.store
        store.add_messages(
            request_id=payload.request_id,
            user_id=payload.user_id,
            messages=[m.model_dump() for m in payload.messages],
        )
        return AddResponse(
            success=True,
            request_id=payload.request_id,
            user_id=payload.user_id,
            session_id=payload.session_id,
        )

    @app.post("/aml/search", response_model=SearchResponse)
    def aml_search(payload: SearchRequest, request: Request) -> SearchResponse:
        authorize(request)
        store: MemoryStore = request.app.state.store
        items = store.search(
            user_id=payload.user_id,
            query=payload.query,
            top_k=payload.top_k,
        )
        return SearchResponse(data=[SearchResultItem(**item) for item in items])

    return app


app = create_app()
