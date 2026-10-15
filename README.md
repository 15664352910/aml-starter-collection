# AML Starter Collection

[Agent Memory Leaderboard（AML）](https://agentmemoryleaderboard.ai/) 2026-08 学术文本榜 **前 10 名开源实现的契约合规 starter 合集**，面向刚参加比赛的新生。

每个子目录是一个**自包含、可运行、已通过契约测试**的参赛参考实现：官方 Add/Search 契约（`/aml/add`、`/aml/search`）+ 各自的原算法 + 冒烟脚本 + 契约测试。

> 核心事实：榜单前 10 全部存在开源实现，且**没有一个需要重型框架**——最简的 273 行 Python（#9）也能上榜。先把最简单的跑通，再谈改进。

## 榜单与选择指南

| 目录 | 名次 | 项目 | 算法一句话 | 外部依赖 | 许可证 |
|---|---|---|---|---|---|
| [01-invmem](01-invmem/) | #1 | InvMem | 分块 + 向量(FAISS) + BM25 混合 RRF | 重（torch/st/faiss） | 未附 |
| [02-refind](02-refind/) | #2 | ReFind | 对话级 BM25 + 迭代检索规划（可选 LLM） | 轻（可纯 BM25） | MIT |
| [03-active-memory-index](03-active-memory-index/) | #3 | ActiveMemoryIndex | 原文 turn + 原子事实双存储，全部可降级 | 中（嵌入必需） | MIT |
| [04-hybrid-memory-search](04-hybrid-memory-search/) | #4 | Hybrid Memory Search | per-user 桶 + 同步嵌入 + BM25/dense 融合 | 嵌入 API 必需 | 未附 |
| [05-chronohybridmem](05-chronohybridmem/) | #5 | ChronoHybridMem | FTS5 多路召回 + 可选 LLM 增强（默认关） | 轻（默认零依赖） | MIT |
| [06-hybrid-episodic](06-hybrid-episodic/) | #6 | Hybrid Episodic Memory | 确定性 BM25+dense RRF，无 LLM | 轻（榜单配置纯 BM25） | MIT |
| [07-chronicle-memory](07-chronicle-memory/) | #7 | Chronicle Memory | SQLite 混合词法打分 + 多样性加分 | **零依赖**（单文件） | 未附 |
| [08-flowgrid-retriever](08-flowgrid-retriever/) | #8 | FlowGrid AML Retriever | evidence-first：原始消息 + 聚合视图 RRF | **零依赖** | MIT |
| [09-memoryagent](09-memoryagent/) | #9 | MemoryAgent（改编） | ChromaDB 向量检索 + 写入去重 | 中（可 hash 后端） | 未附（改编 MIT） |
| [10-aml-memory-mvp](10-aml-memory-mvp/) | #10 | AML Memory MVP | FTS5 + n-gram typo 容忍 + 邻域扩展 | 轻（默认零依赖） | MIT |

**怎么选**：

- 想最快跑通、一行依赖都不想装 → **07**（单文件、标准库）
- 想要"零依赖 + 有竞争力的完整设计" → **08** 或 **10**
- 想研究混合检索公式 → **06**（确定性 RRF，公式最干净）
- 想直接部署参赛（接受嵌入 API）→ **04**
- 想看 LLM 增强如何接入（且可全关）→ **03** 或 **05**
- 想要最小可改骨架 → **09**（唯一深度改编版，约 500 行）

## 通用使用方式

每个子目录：

```bash
cd <子目录>
pip install -r requirements.txt        # 或按各 README 的轻量依赖
<按 README 启动服务>
pytest -v                              # 契约测试（契约回显/用户隔离/幂等重试）
python scripts/smoke_check.py http://127.0.0.1:8000   # 端到端冒烟
```

参赛流程（详见各 README 与 [09 的完整参赛清单](09-memoryagent/README.md)）：部署到公网 →
生成密钥 → [提交评测申请](https://agentmemoryleaderboard.ai/evaluation) → Smoke → Full → 复核上榜。

## 契约适配原理

9 个原项目（01–08、10）**本来就实现了完整的 AML Add/Search 语义**（request_id 幂等、
user_id 隔离、契约字段回显）——它们就是这样上的榜，差异仅在 URL 路径。因此适配层极薄：

- **FastAPI 项目**（01/02/03/04/05/06/10）：`aml_entry.py` 在原有 app 上注册
  `/aml/add`、`/aml/search` 别名路由，指向原端点函数，原始路径全部保留；
- **stdlib 项目**（07）：`app.py` 内 3 行路径归一化（已标注 `[starter 适配]`）；
- **配置型项目**（08）：路径本来就由环境变量控制，零改码。

改编没有触碰任何检索核心代码。测试中的嵌入/LLM 打桩只在测试文件里，不影响原始实现。

## 验证状态

每个子目录的 `tests/test_contract.py` 均已通过（契约字段回显、user_id 隔离、
request_id 幂等重试、检索命中），其中 02/10 另经真实 uvicorn 进程 + HTTP 客户端
端到端复核（`scripts/smoke_check.py` 10/10 项）。

## 许可与归因

各子目录代码版权归原仓库作者所有：02/03/05/06/08/10 为 MIT（LICENSE 文件随原仓库保留），
01/04/07 原仓库未附许可证、09 改编版以 MIT 重发布——**未附许可证的四个仅供学习参考**，
如需再发布请自行联系原作者。合集本身的组织与适配层代码以 MIT 发布。

官方仓库：<https://github.com/AML-memory/agent-memory-leaderboard> ·
收录列表：<https://github.com/TeleAI-UAGI/Awesome-Agent-Memory>
