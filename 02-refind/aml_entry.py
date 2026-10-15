"""[starter 适配] AML 官方契约入口：/aml/add、/aml/search、/health。

原项目（02-ReFind，榜单 #2）已实现完整契约语义，仅路径不同（/v1/memories/*、/add、/search）。
本文件在其 FastAPI app 上注册官方路径别名。启动：uvicorn aml_entry:app --port 8000

免 LLM 运行：环境变量 RETRIEVAL_MODE=bm25（纯 BM25 检索；agent 模式需配置 LLM_API_KEY）。
"""

from __future__ import annotations

from fastapi import FastAPI

from app.main import create_app
from app.models import AddResponse, HealthResponse, SearchResponse

theirs = create_app()

_BY_PATH = {getattr(r, "path", None): r for r in theirs.router.routes}

app = FastAPI(
    title="ReFind (starter)",
    description="榜单 #2 开源实现的 AML 契约封装；原始路径仍可用",
    version="1.0.0",
)

_ALIAS = {
    "/add": ("/aml/add", AddResponse),
    "/search": ("/aml/search", SearchResponse),
    "/health": ("/health", HealthResponse),
}

for original, (alias, model) in _ALIAS.items():
    route = _BY_PATH.get(original)
    if route is None:
        continue
    app.add_api_route(
        alias,
        route.endpoint,
        methods=[m.lower() for m in route.methods],
        response_model=model,
        dependencies=list(route.dependencies),
    )
