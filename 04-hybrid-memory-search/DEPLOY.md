# Hybrid Memory Search — 部署说明

## 交付内容

| 文件 | 作用 |
|------|------|
| `hybrid-memory-1.0.0.tar` | 预构建镜像 |
| `docker-compose.yml` | 启动配置 |
| `.env.example` | 环境变量模板（无真实密钥） |
| `.env` | 实际配置（可由 `.env.example` 复制填写；若已单独提供则可直接用） |
| `DEPLOY.md` | 本文档 |

## 配置 `.env`

`docker compose` 启动时会自动读取**同目录**下的 `.env`。仓库只提供模板 `.env.example`（不含真实密钥），部署时请自行生成 `.env`：

```bash
cp .env.example .env
# 再用编辑器填写下表中的必填项
```

### 必填项（add / search 依赖）

| 变量 | 含义 | 示例 / 说明 |
|------|------|-------------|
| `OPENAI_EMBEDDING_API_KEY` | Embedding 服务 API Key | 必填；填你的真实密钥，勿提交到 Git |
| `OPENAI_EMBEDDING_API_BASE` | Embedding API 根地址 | 默认 `https://api.siliconflow.cn/v1`；换成自有兼容 OpenAI 的 endpoint 亦可 |
| `OPENAI_EMBEDDING_MODEL` | Embedding 模型名 | 默认 `BAAI/bge-m3`；须与 Key 对应服务商支持的模型一致 |
| `OPENAI_EMBEDDING_DIMENSIONS` | 向量维度 | 默认 `1024`；须与所选模型输出维度一致 |

### 常用可选项

| 变量 | 含义 | 默认值 | 说明 |
|------|------|--------|------|
| `PORT` | 主机映射端口 | `8000` | 访问地址为 `http://<主机>:<PORT>` |
| `HYBRID_ALPHA` | BM25 / 向量融合权重 | `0.5` | `score = α * bm25_norm + (1-α) * dense_norm`；`α` 越大越偏关键词 |
| `MEMORY_STORE_DIR` | 记忆落盘目录 | `store` | Compose 启动时会覆盖为容器内 `/app/store`，一般无需改 |

### 可选（当前 add / search 不依赖）

以下仅在后续接「生成答案 / Chat」时需要，模板中默认注释掉：

| 变量 | 含义 |
|------|------|
| `OPENAI_API_BASE` | Chat LLM API 根地址 |
| `OPENAI_API_KEY` | Chat LLM API Key |
| `OPENAI_MODEL` | Chat 模型名 |

### 注意

- **不需要** PostgreSQL / Redis 等外部数据库；`.env` 模板中也无数据库相关项。
- 主机须能访问 `OPENAI_EMBEDDING_API_BASE`（网络 / 代理 / 防火墙放行）。
- **不要**把填好密钥的 `.env` 提交到 Git；已通过 `.gitignore` 忽略。
- 更换 Embedding 模型或维度后，建议清空 `./store` 再重新导入，避免旧向量维度不一致。

## 目录与数据集放置

建议在部署目录保持如下结构：

```text
<部署目录>/
  hybrid-memory-1.0.0.tar
  docker-compose.yml
  .env.example
  .env
  DEPLOY.md
  data/                          ← 本地导入辅助用（主机路径）
  store/                         ← 写入后的记忆（自动生成，勿手改）
```

## 环境要求

- 已安装 [Docker Desktop](https://docs.docker.com/get-docker/)（或 Docker Engine）与 Compose v2
- 主机可访问 `.env` 中配置的 Embedding API
- **不需要** PostgreSQL / Redis 等外部数据库

## 启动步骤

1. 将交付文件放到同一目录；按上文配置好 `.env`
2. 启动 Docker
3. 执行：

```bash
cd <部署目录>
cp .env.example .env          # 若尚无 .env
# 编辑 .env，填入 OPENAI_EMBEDDING_API_KEY 等

docker load -i hybrid-memory-1.0.0.tar
docker compose up -d
docker compose ps
curl http://127.0.0.1:8000/health
# 期望: {"status":"ok"}
```

- API：`http://127.0.0.1:8000` 或 `http://<主机IP>:8000`
- 交互文档：`http://127.0.0.1:8000/docs`

**评测平台注册链接（默认端口 8000）：**

| 接口 | URL |
|------|-----|
| Add | `http://<主机>:<PORT>/v1/add` |
| Search | `http://<主机>:<PORT>/v1/search` |

修改端口：编辑 `.env` 中 `PORT`，再执行 `docker compose up -d`。

```bash
docker compose logs -f      # 日志
docker compose restart      # 重启
docker compose down         # 停止（不删除 ./store）
```

## 接口说明

### 1. 健康检查

```bash
curl http://127.0.0.1:8000/health
```

### 2. Add（评测平台）— `POST /v1/add`

平台为每个记忆分块发送一次请求。服务会在响应前**同步**存储全部消息并完成 embedding；检索隔离以 `user_id` 为准，`session_id` 仅用于分组元数据。

```bash
curl -X POST http://127.0.0.1:8000/v1/add \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "eval:<run_id>:locomo_refined:conv-0:chunk-0",
    "messages": [{
      "role": "user",
      "timestamp": 1704067200000,
      "content": "raw memory text"
    }],
    "user_id": "eval:<run_id>:locomo:conv-0",
    "session_id": "eval:<run_id>:sample:0"
  }'
```

成功响应（HTTP 200，记忆已可被 Search 检索）：

```json
{
  "success": true,
  "request_id": "eval:<run_id>:locomo_refined:conv-0:chunk-0",
  "user_id": "eval:<run_id>:locomo:conv-0",
  "session_id": "eval:<run_id>:sample:0"
}
```

| 字段 | 要求 | 说明 |
|------|------|------|
| `request_id` | 必填 | 本次写入唯一标识；响应原样返回；同时作为记忆 id |
| `messages` | 必填 | 有序消息列表；全部写入后再返回 |
| `role` | 必填 | `user` 或 `assistant` |
| `content` | 必填 | 原始消息文本 |
| `timestamp` | 可选 | Unix 毫秒 |
| `user_id` | 必填 | 检索作用域（内部 bucket） |
| `session_id` | 必填 | 源会话标识（写入 metadata） |

### 3. Search（评测平台）— `POST /v1/search`

所有 Add 完成后，平台按问题检索。在同一 `user_id` 内做 BM25 + dense 混合检索，按相关性降序返回。

```bash
curl -X POST http://127.0.0.1:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which answer best matches the memory?",
    "options": ["A. First answer", "B. Second answer"],
    "user_id": "eval:<run_id>:locomo:conv-0",
    "top_k": 100
  }'
```

响应：

```json
{
  "data": [
    {
      "id": "mem_123",
      "content": "remembered fact text",
      "score": 0.87,
      "created_at": "2026-07-01T12:00:00Z"
    }
  ]
}
```

| 字段 | 要求 | 说明 |
|------|------|------|
| `query` | 必填 | 原始基准问题 |
| `options` | 可选 | 选择题候选项；无选项时省略；会并入检索 query |
| `user_id` | 必填 | 必须与 Add 时一致 |
| `top_k` | 必填 | 最大返回条数 |

无相关记忆时返回 `{ "data": [] }`。

### 4. 列出记忆桶 — `GET /v1/buckets`

```bash
curl http://127.0.0.1:8000/v1/buckets
```

### 5. 本地辅助：单条写入 — `POST /v1/upsert`

非评测平台路径，仅本地调试用。

```bash
curl -X POST http://127.0.0.1:8000/v1/upsert \
  -H "Content-Type: application/json" \
  -d '{
    "id": "demo:session_1",
    "bucket_id": "demo",
    "text": "[Session 1 | Unknown]\nUser: hello\nAssistant: hi",
    "metadata": {"session_index": 1},
    "embed": true,
    "skip_if_exists": true
  }'
```

### 6. 本地辅助：上传导入 — `POST /v1/add/upload`

将 LoCoMo / LongMemEval 等 JSON/JSONL 上传并按 session 切分导入（非评测平台协议）。

```bash
curl -X POST "http://127.0.0.1:8000/v1/add/upload?embed=true" \
  -F "file=@./data/locomo_refined.json"
```

### 7. 本地辅助：挂载导入 — `POST /v1/add/dataset`

读取容器内已挂载的 `data/` 文件。

| 主机路径 | 接口中的 `data_path` |
|----------|----------------------|
| `<部署目录>/data/locomo_refined.json` | `/app/data/locomo_refined.json` |

```bash
curl -X POST http://127.0.0.1:8000/v1/add/dataset \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "/app/data/locomo_refined.json",
    "embed": true
  }'
```

## 数据持久化

记忆保存在主机 `./store`：

- `docker compose down` / 重启后数据仍在
- 删除 `./store` 后才会清空
