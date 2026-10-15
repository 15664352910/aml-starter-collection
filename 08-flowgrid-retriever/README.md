# 8. FlowGrid AML Retriever — AML 契约 starter

> AML 2026-08 学术文本榜 **#8** 开源实现 · 原仓库：[dlxeva/flowgrid-aml-retriever](https://github.com/dlxeva/flowgrid-aml-retriever) · 许可证：MIT

**算法**：确定性 evidence-first：存原始消息 + 聚合视图（window/session 段），加权 RRF 融合与软重排

## 运行

```bash
pip install -r requirements.txt   # 测试仅需 fastapi uvicorn httpx pytest（见下）
python -m aml_retriever.cli serve
```

## 关键环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `AML_ADD_PATH` | `/aml/add` | 官方契约路径（本项目开箱即用） |
| `AML_SEARCH_PATH` | `/aml/search` |  |
| `AML_AUTH_MODE` | `bearer` | none | bearer | token | x-api-key |
| `AML_API_KEY` | `` | 认证密钥 |
| `AML_DB_PATH` | `` | SQLite 路径（默认内存） |

路径完全由配置控制，契约适配零改码。


## 契约适配说明

原项目已实现完整 AML Add/Search 语义（request_id 幂等、user_id 隔离、契约字段回显），
差异仅在 URL 路径。本 starter 的适配层：

- FastAPI 项目：`aml_entry.py` 在原有 app 上注册 `/aml/add`、`/aml/search` 别名（原始路径保留）；
- stdlib 项目：`app.py` 内 3 行路径归一化（已标注 `[starter 适配]`）。

## 验证

```bash
pip install fastapi uvicorn httpx pytest
pytest -v                                    # 契约测试：契约回显/隔离/幂等
python scripts/smoke_check.py http://127.0.0.1:8000   # 端到端冒烟（服务先跑起来）
```

> 归因与许可：本目录代码版权归原仓库作者所有（许可证见上）。改编仅限契约入口与测试，
> 检索核心保持原样。原仓库未附许可证的项目（InvMem/Hybrid Memory Search/Chronicle Memory/
> MemoryAgent）仅供学习参考。
