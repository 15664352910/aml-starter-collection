"""[starter 适配] AML 官方契约入口：/aml/add、/aml/search、/health。

原项目（04-Hybrid Memory Search，榜单 #4）已实现完整契约语义，仅路径不同（/v1/*）。
本文件在其 FastAPI app 上注册官方路径别名。启动：uvicorn aml_entry:app --port 8000

依赖 OpenAI 兼容的嵌入 API（环境变量 OPENAI_EMBEDDING_API_BASE / _API_KEY / _MODEL），
写入时同步嵌入，检索为 BM25+dense 加权融合，per-user 隔离。
"""

from __future__ import annotations

from fastapi import FastAPI

from src.server import app as theirs

_BY_PATH = {getattr(r, "path", None): r for r in theirs.router.routes}

app = FastAPI(
    title="Hybrid Memory Search (starter)",
    description="榜单 #4 开源实现的 AML 契约封装；原始路径 /v1/* 仍可用",
    version="1.0.0",
)

_ALIAS = {
    "/v1/add": "/aml/add",
    "/v1/search": "/aml/search",
    "/health": "/health",
}

for original, alias in _ALIAS.items():
    route = _BY_PATH.get(original)
    if route is None:
        continue
    app.add_api_route(
        alias,
        route.endpoint,
        methods=[m.lower() for m in route.methods],
        dependencies=list(route.dependencies),
    )
