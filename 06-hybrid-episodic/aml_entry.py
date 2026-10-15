"""[starter 适配] AML 官方契约入口：/aml/add、/aml/search、/health。

原项目（06-Hybrid Episodic Memory，榜单 #6）已实现完整契约语义，仅路径不同。
本文件在其 FastAPI app 上注册官方路径别名。启动：uvicorn aml_entry:app --port 8000

免模型运行：EMBED_ENABLED=false 时为纯 BM25 + RRF 确定性检索（榜单提交配置即此）。
"""

from __future__ import annotations

from fastapi import FastAPI

from src.app import app as theirs
from src.models import AddResponse, SearchResponse

_BY_PATH = {getattr(r, "path", None): r for r in theirs.router.routes}

app = FastAPI(
    title="Hybrid Episodic Memory (starter)",
    description="榜单 #6 开源实现的 AML 契约封装；原始路径 /add /search 仍可用",
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
