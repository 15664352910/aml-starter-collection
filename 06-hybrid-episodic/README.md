# 6. Hybrid Episodic Memory — AML 契约 starter

> AML 2026-08 学术文本榜 **#6** 开源实现 · 原仓库：[tlysanhuo/agent-memory-challenge](https://github.com/tlysanhuo/agent-memory-challenge) · 许可证：MIT

**算法**：确定性 BM25+dense 加权 RRF 融合，检索路径无 LLM（榜单提交配置：纯 BM25）

## 运行

```bash
pip install -r requirements.txt   # 测试仅需 fastapi uvicorn httpx pytest（见下）
uvicorn aml_entry:app --host 0.0.0.0 --port 8000
```

## 关键环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EMBED_ENABLED` | `false` | false=纯 BM25（免模型，榜单配置）；true=启用本地嵌入 |
| `AUTH_MODE` | `auto` | auto | bearer | token | x-api-key | none |
| `API_KEY` | `` | 认证密钥 |


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
