"""[starter 适配] AML 官方契约入口：/aml/add、/aml/search、/health。

原项目（05-ChronoHybridMem，榜单 #5）已实现完整契约语义，仅路径不同。
本文件在其 FastAPI app 上注册官方路径别名。启动：uvicorn aml_entry:app --port 8000
默认纯 SQLite FTS5 检索，无需任何外部服务；可选 LLM 增强均通过环境变量开启。
"""

from __future__ import annotations

from fastapi import FastAPI

from app.main import create_app
from app.schemas import AddResponse, SearchResponse

theirs = create_app()

_BY_PATH = {getattr(r, "path", None): r for r in theirs.router.routes}

app = FastAPI(
    title="ChronoHybridMem (starter)",
    description="榜单 #5 开源实现的 AML 契约封装；原始路径 /add /search 仍可用",
    version="1.0.0",
)

_ALIAS = {
    "/add": ("/aml/add", AddResponse),
    "/search": ("/aml/search", SearchResponse),
    "/health": ("/health", None),
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
