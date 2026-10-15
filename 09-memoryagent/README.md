# AML Memory Starter

面向 [Agent Memory Leaderboard（AML）](https://agentmemoryleaderboard.ai/) **文本记忆赛道 · 开源方法组** 新生的最小可参赛模板。

改编自 2026-08 学术文本榜 **第 9 名**开源实现 [MemoryAgent](https://github.com/llLAlisa/memory-agent-submission)（273 行 Python 上榜），按官方 Add/Search 契约改造为可直接部署的完整项目：**约 500 行代码 + 测试 + 冒烟脚本 + Docker**，从零到通过 Smoke 测试最快半天。

> 设计取向：**简单、确定、契约正确**。上榜事实证明 AML 不挑复杂度——把 Add/Search 语义对接正确即可得分，先把最简单的跑通，再谈改进。

## 功能特性

| 特性 | 说明 |
|---|---|
| Add/Search 契约 | `POST /aml/add`、`POST /aml/search`，字段与官方接口指南一致 |
| user_id 隔离 | 每个用户独立 Chroma 集合（哈希命名），检索不串数据 |
| 幂等摄入 | `request_id` 账本（SQLite），平台重试不产生重复数据 |
| 写入去重 | 写入时向量相似度去重（阈值可调） |
| 可选弃权 | 得分低于 `ABSTENTION_THRESHOLD` 返回空数组（默认关闭） |
| 认证 | 设置 `AML_MEMORY_SYSTEM_KEY` 后强制 Bearer 认证 |
| 双嵌入后端 | sentence-transformers（默认）/ 纯本地 hash 后端（免下载模型，演示测试用） |

## 快速开始

### 方式一：本地运行（5 分钟，免模型下载）

```bash
cd aml-memory-starter
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install fastapi uvicorn pydantic chromadb httpx

# 用 hash 后端起服务（零模型下载，适合先跑通流程）
EMBEDDING_BACKEND=hash uvicorn app.main:app --port 8000
```

另开一个终端：

```bash
pip install httpx
python scripts/smoke_check.py http://127.0.0.1:8000
# 看到 "全部冒烟检查通过" 即契约验证成功
```

### 方式二：Docker（推荐参赛部署）

```bash
cp .env.example .env            # 编辑 .env，设置 AML_MEMORY_SYSTEM_KEY
docker compose up -d --build
curl http://127.0.0.1:8000/health
python scripts/smoke_check.py http://127.0.0.1:8000 <你的key>
```

国内构建拉不动 HuggingFace 模型时：

```bash
docker compose build --build-arg HF_ENDPOINT=https://hf-mirror.com
```

## 测试

```bash
pip install pytest
pytest -v        # 全部使用 hash 后端，离线可跑
```

## 参赛流程（照做即可）

1. **读规则**：[赛事规则](https://agentmemoryleaderboard.ai/rules) + [接口指南](https://agentmemoryleaderboard.ai/api-guide)
2. **部署到公网**：任意 VPS + Docker，套一层 HTTPS 反代（如 Caddy，自动签发证书）
3. **生成密钥**：`python -c "import secrets; print(secrets.token_hex(32))"` 写入 `.env` 的 `AML_MEMORY_SYSTEM_KEY`
4. **提交评测申请**：[evaluation 页面](https://agentmemoryleaderboard.ai/evaluation)，填线上 endpoint 与联系信息
5. **Smoke 测试**：用平台审核后给的 key 跑 `python scripts/smoke_check.py https://<你的域名> <key>`
6. **发起完整评测** → 个人页看私有结果 → 提交复核 → 上榜单

## 目录结构

```text
aml-memory-starter/
├── app/
│   ├── main.py        # FastAPI 入口：/aml/add /aml/search /health + 认证
│   ├── config.py      # 环境变量配置
│   ├── schemas.py     # 契约请求/响应模型
│   ├── embedder.py    # 双嵌入后端（st / hash）
│   └── store.py       # ChromaDB 存储 + 去重 + request_id 幂等账本
├── scripts/
│   └── smoke_check.py # 端到端契约冒烟检查
├── tests/
│   └── test_contract.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── LICENSE            # MIT，含对 MemoryAgent 的署名致谢
```

## 契约速查

**Add** `POST /aml/add`

```json
{
  "request_id": "conv-0:chunk-0",
  "messages": [{"role": "user", "content": "...", "timestamp": 1726000000000}],
  "user_id": "user-a",
  "session_id": "session-0"
}
→ {"success": true, "request_id": "...", "user_id": "...", "session_id": "..."}
```

**Search** `POST /aml/search`

```json
{"query": "...", "user_id": "user-a", "top_k": 100, "options": null}
→ {"data": [{"id": "...", "content": "...", "score": 0.92, "created_at": "2026-..."}]}
```

- 空数组 `{"data": []}` 表示弃权（证据不足），平台允许
- `timestamp` 为 Unix 毫秒；`created_at` 返回 ISO 8601

## 改进方向（跑通后再做）

1. 混合检索：向量 + BM25（SQLite FTS5）加权融合，参考榜单 #5/#6/#7
2. 会话邻域扩展：把同一 session 的相邻消息一并返回
3. 短语/时间感知打分：参考榜单 #7 Chronicle Memory
4. 换上更强的嵌入模型（如 bge-m3，中英都强）

## 许可证

MIT。改编自 MemoryAgent（榜单 #9，原仓库未附许可证），致谢原作者。

---

> 本目录是 [aml-starter-collection](../README.md) 的一部分：AML 2026-08 学术文本榜 #9
> MemoryAgent（[llLAlisa/memory-agent-submission](https://github.com/llLAlisa/memory-agent-submission)，
> 原仓库未附许可证）的契约改编 starter，也是合集中唯一深度改编版（其余 9 个仅加契约入口）。
