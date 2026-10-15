# 2. ReFind — AML 契约 starter

> AML 2026-08 学术文本榜 **#2** 开源实现 · 原仓库：[imlrz/ReFind](https://github.com/imlrz/ReFind) · 许可证：MIT

**算法**：对话级 BM25 索引 + 迭代检索规划（agent 模式用 LLM 规划多轮搜索，bm25 模式纯检索）

## 运行

```bash
pip install -r requirements.txt   # 测试仅需 fastapi uvicorn httpx pytest（见下）
uvicorn aml_entry:app --host 0.0.0.0 --port 8000
```

## 关键环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RETRIEVAL_MODE` | `bm25` | bm25（免 LLM）| agent（需 LLM_API_KEY） |
| `DATABASE_PATH` | `./data/refind.sqlite3` | SQLite 存储路径 |
| `LLM_API_KEY` | `` | agent 模式的 OpenAI 兼容 key（可选） |


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
