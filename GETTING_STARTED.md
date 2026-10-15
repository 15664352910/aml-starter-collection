# 新生上手指南：从克隆仓库到通过 AML Smoke 测试

> 目标读者：刚参加 Agent Memory Leaderboard（AML）文本记忆赛道的同学。
> 预期耗时：本地跑通 30 分钟；部署上线 + 提交申请约半天。
> 本指南命令以 Windows PowerShell 为主，Linux/macOS 差异处会注明。

## 0. 这份合集是什么

[AML](https://agentmemoryleaderboard.ai/) 2026-08 学术文本榜前 10 名全部存在开源实现，本合集把它们整理成**可直接运行的契约合规 starter**：每个目录都是完整服务，暴露官方 Add/Search 接口，自带契约测试和冒烟脚本。

| 目录 | 名次 | 特点 | 依赖 |
|---|---|---|---|
| `07-chronicle-memory` | #7 | 单文件、纯标准库，**零安装** | 无 |
| `08-flowgrid-retriever` | #10→#8 | 零依赖、证据溯源设计完整 | 无 |
| `10-aml-memory-mvp` | #10 | FTS5 + typo 容忍，默认零外部依赖 | 无 |
| `09-memoryagent` | #9 | 唯一深度改编版，约 500 行，最适合当自己的骨架 | 可用 hash 后端免模型 |
| `06-hybrid-episodic` | #6 | 确定性 BM25+dense RRF，公式最干净 | 可纯 BM25 |
| `02-refind` | #2 | 对话级 BM25 + 可选 LLM 迭代规划 | 轻 |
| `05-chronohybridmem` | #5 | FTS5 多路召回 + 可选 LLM 增强 | 轻 |
| `03-active-memory-index` | #3 | 原文+原子事实双存储，LLM/嵌入均可降级 | 中 |
| `04-hybrid-memory-search` | #4 | per-user 隔离 + 同步嵌入融合 | 需嵌入 API |
| `01-invmem` | #1 | 最简 RAG（FAISS+BM25） | 重（本地模型） |

## 1. 准备环境（10 分钟）

必需：

```powershell
# Python 3.11 或 3.12
python --version
# Git
git --version
```

克隆本合集（网络不畅见第 8 节）：

```powershell
git clone https://github.com/15664352910/aml-starter-collection.git
cd aml-starter-collection
```

可选（部署阶段需要）：

- Docker Desktop（本机试容器用）
- 一台 VPS（参赛部署用，阿里云/腾讯云轻量应用服务器即可，2C2G 够用）

## 2. 第一步：5 分钟跑通一个服务（零依赖的 07）

```powershell
cd 07-chronicle-memory
python app.py
```

看到 `Chronicle Memory listening on http://0.0.0.0:8000` 即启动成功。新开一个 PowerShell：

```powershell
cd 07-chronicle-memory
pip install httpx                    # 冒烟脚本唯一依赖
python scripts/smoke_check.py http://127.0.0.1:8000
```

输出 `全部冒烟检查通过` 即完成——你刚刚用官方契约完成了 Add 摄入、检索、用户隔离、幂等重试的全流程。

## 3. 第二步：跑契约测试（理解契约的最好方式）

```powershell
pip install fastapi uvicorn pydantic httpx pytest
cd 07-chronicle-memory
pytest -v
```

每个 starter 的 `tests/test_contract.py` 都验证四件事，这四件事就是 AML 契约的硬要求：

1. **契约字段回显**：`POST /aml/add` 必须原样返回 `request_id` / `user_id` / `session_id` 且 `success: true`（平台逐字节比对）；
2. **user_id 隔离**：用户 A 的记忆绝不能出现在用户 B 的搜索结果里（隐私红线）；
3. **request_id 幂等**：同一 `request_id` 重试不能产生重复数据（平台会重试）;
4. **检索语义**：返回 `{"data": [{"id","content","score","created_at"}]}`，空数组表示弃权。

建议：把 `tests/test_contract.py` 逐行读一遍，再对照 `aml_entry.py`（或 07 的 `app.py`），半小时就能理解契约全貌。

## 4. 第三步：跑一个"轻量但有真算法"的项目（02 / 06 / 10）

```powershell
cd 02-refind
pip install fastapi uvicorn pydantic httpx
$env:RETRIEVAL_MODE = "bm25"          # 免 LLM
$env:DATABASE_PATH = "./data/refind.sqlite3"
uvicorn aml_entry:app --port 8000
```

另开终端：`python scripts/smoke_check.py http://127.0.0.1:8000`。

对照阅读顺序：`aml_entry.py`（10 行契约别名）→ `app/main.py`（API 与存储）→ `app/retriever.py`（BM25 索引与检索规划）。想研究混合检索公式，换 `06-hybrid-episodic` 读 `src/index.py` 的 RRF 融合。

## 5. 第四步：完整模式（真实嵌入模型 + Docker）

以 09 为例（hash 后端→真实模型的切换路径在所有 starter 里思路相同）：

```powershell
cd 09-memoryagent
pip install fastapi uvicorn pydantic chromadb sentence-transformers httpx
uvicorn app.main:app --port 8000      # 首次启动自动下载 all-MiniLM-L6-v2（约 90MB）
```

国内下载 HuggingFace 慢/失败时，设镜像（PowerShell）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

Docker 一键（含 CPU 版 torch 和模型预下载）：

```powershell
cd 09-memoryagent
copy .env.example .env                # 按需编辑
docker compose up -d --build
```

## 6. 第五步：部署到公网 + 提交参赛申请

### 6.1 部署（以阿里云轻量应用服务器为例，约 1 小时）

1. 购买：轻量应用服务器，2C2G、Ubuntu 22.04，地域任选（~60 元/月或学生机更便宜）
2. 控制台 → 实例详情 → **防火墙**：放行 80、443 端口
3. SSH 登录后：

```bash
# 装 Docker（官方脚本）
curl -fsSL https://get.docker.com | sh
# 装 compose 插件（新版 docker 已含；若无：apt install docker-compose-plugin）

# 拉代码（git 不通时见第 8 节代理）
cd /srv && git clone https://github.com/15664352910/aml-starter-collection.git
cd aml-starter-collection/10-aml-memory-mvp   # 任选一个零依赖的，最省事
docker build -t aml-svc .
docker run -d --name aml --restart always -p 8000:8000 \
  -e AML_API_KEY=<生成的密钥> aml-svc
```

4. 域名 + HTTPS：买一个便宜域名（如 `.icu`/`.top` 首年几块钱），DNS 加一条 A 记录指向服务器 IP，等解析生效（`nslookup 你的域名` 能解析即可）。然后用 Caddy 自动签发证书：

```bash
cd /srv && echo '你的域名.icu

reverse_proxy 127.0.0.1:8000' > Caddyfile
docker run -d --name caddy --network host --restart always \
  -v /srv/Caddyfile:/etc/caddy/Caddyfile caddy:2
```

5. 本机验证：

```powershell
python scripts/smoke_check.py https://你的域名.icu <你的密钥>
```

### 6.2 申请参赛（照官方流程六步）

1. 读规则：[agentmemoryleaderboard.ai/rules](https://agentmemoryleaderboard.ai/rules) + [接口指南](https://agentmemoryleaderboard.ai/api-guide)
2. 实现/部署 Add 和 Search（本合集已完成）
3. 提交[评测权限申请](https://agentmemoryleaderboard.ai/evaluation)：填线上 endpoint、联系方式、系统信息
4. 拿到 AML Key 后跑 Smoke（不计分兼容性测试）——即用平台给的 key 再跑一遍 `smoke_check.py`
5. 发起完整评测，个人页面看私有结果
6. 提交复核，通过后进入公开榜单（开源方法组需公开代码仓库）

生成密钥：

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

## 7. 如何改成"你自己的"系统（参赛要求开源方法组有原创性）

推荐路径（由浅入深）：

1. **改 09**（约 500 行，结构最简）：替换嵌入模型、调整去重阈值、加自己的打分逻辑；
2. **给 07/10 加一路召回**：在 FTS5 结果上叠一层向量或 BM25，做加权融合（参考 06 的 RRF 公式）；
3. **加治理能力**：冲突检测、时间感知、弃权阈值——这是当前榜单系统普遍薄弱处，也是论文点；
4. **换更强嵌入**：bge-m3（中英兼优）通常比 all-MiniLM 有可见提升；
5. **写清楚方法说明**：README 里写明相对原实现的改动，保留对原项目的署名（许可证要求）。

注意：参赛系统的 Add/Search 路径内使用的模型有约束（学术组固定 gpt-4o-mini 用于 LLM 功能），详见官方规则。

## 8. 常见问题

**git 连不上 GitHub（clone/push 被重置）**

```powershell
git config --global url."https://gh-proxy.com/https://github.com/".insteadOf "https://github.com/"
```

**HuggingFace 模型下载慢/失败**

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"      # PowerShell
export HF_ENDPOINT=https://hf-mirror.com        # bash
```

**`pip install` 慢**

```powershell
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

**Windows 下 `pytest` 找不到模块**：在本子目录下运行（每个 starter 是独立项目，`python -m pytest tests/test_contract.py`）。

**端口被占用**：`netstat -ano | findstr :8000` 找 PID 后 `taskkill /PID <pid> /F`。

**想干净重跑测试**：删除子目录下的 `.test-data/`（测试数据库）。

## 9. 学习路线建议

1. 第 1 天：跑通 07 + 读它的 `app.py`（单文件，~300 行）
2. 第 2 天：跑通 02 或 10，读测试与检索实现
3. 第 3 天：部署一个到 VPS，跑通 HTTPS + 冒烟（第 6 节）
4. 第 4 天：提交评测申请；同时开始改造自己的版本
5. 之后：对照榜单前 10 的差异（本合集一键可比对），选一个改进点做深

有环境问题先看各子目录 `README.md` 的"关键环境变量"表，再到合集仓库提 Issue。

---

**致谢**：10 个 starter 的检索核心均来自对应上榜项目的开源作者（见各目录 README 署名与 LICENSE），改编仅限契约入口与测试。
