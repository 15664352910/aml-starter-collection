# 5. ChronoHybridMem — AML 契约 starter

> AML 2026-08 学术文本榜 **#5** 开源实现 · 原仓库：[Tin11Mn/chrono-hybrid-mem](https://github.com/Tin11Mn/chrono-hybrid-mem) · 许可证：MIT

**算法**：SQLite FTS5 evidence-only 多路召回 + 可选 LLM 事实抽取/查询规划（默认全关）

## 运行

```bash
pip install -r requirements.txt   # 测试仅需 fastapi uvicorn httpx pytest（见下）
uvicorn aml_entry:app --host 0.0.0.0 --port 8000
```

## 关键环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MEMORY_DB_PATH` | `./data/chrono.db` | SQLite 存储路径 |

默认零外部依赖；增强功能（结构化查询计划、证据图等）见 app/main.py 顶部环境变量。


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
