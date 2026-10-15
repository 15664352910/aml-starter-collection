"""[starter 适配] AML 官方契约入口：/aml/add、/aml/search、/health。

原项目（01-InvMem，榜单 #1）已实现完整契约语义，仅路径不同。
在其 FastAPI app 上直接注册官方路径别名（保留 lifespan 与校验错误处理）。
启动：uvicorn aml_entry:app --port 8000

本地免认证：ALLOW_UNAUTHENTICATED=true（否则必须设置 MEMORY_API_KEY，缺省返回 503）。
免模型预热：SKIP_MODEL_STARTUP=true。
"""

from __future__ import annotations

from src.vanilla_rag.api import app

_BY_PATH = {getattr(r, "path", None): r for r in app.router.routes}

for original, alias in (("/add", "/aml/add"), ("/search", "/aml/search")):
    route = _BY_PATH.get(original)
    if route is None:
        continue
    app.add_api_route(
        alias,
        route.endpoint,
        methods=[m.lower() for m in route.methods],
        include_in_schema=False,
    )
