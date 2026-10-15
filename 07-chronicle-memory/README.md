# 7. Chronicle Memory — AML 契约 starter

> AML 2026-08 学术文本榜 **#7** 开源实现 · 原仓库：[simple-boy/Chronicle-Memory](https://github.com/simple-boy/Chronicle-Memory) · 许可证：原仓库未附许可证

**算法**：SQLite 混合词法打分 + 短语/时间/会话多样性加分（单文件 app.py，零第三方依赖）

## 运行

```bash
pip install -r requirements.txt   # 测试仅需 fastapi uvicorn httpx pytest（见下）
python app.py
```

## 关键环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `HOST` | `0.0.0.0` |  |
| `PORT` | `8000` |  |
| `MEMORY_DB_PATH` | `./data/memories.sqlite3` | SQLite 存储路径 |
| `MEMORY_API_KEY` | `` | 认证密钥（可选） |

标准库 http.server，无需安装任何依赖即可运行。


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
