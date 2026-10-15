"""[starter 适配] AML 官方契约入口：/aml/add、/aml/search、/health。

原项目（03-ActiveMemoryIndex，榜单 #3）已实现完整契约语义，仅路径不同。
在其 FastAPI app 上直接注册官方路径别名（保留原中间件：准入阀、死线等）。
启动：uvicorn aml_entry:app --port 8000

免 LLM 运行：不设 AMI_LLM_API_KEY 时所有 LLM 功能自动降级为 raw-text 回退；
AMI_EXTRACT=0 时 Add 不做事实抽取（仅存原文 turn）。
嵌入可选：AMI_EMBED_BACKEND=bge（本地模型，默认）或 openai 兼容 API。
"""

from __future__ import annotations

from app.main import app

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
