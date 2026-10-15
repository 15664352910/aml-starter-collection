"""AML Add/Search 契约模型。

字段名与语义遵循官方接口指南：<https://agentmemoryleaderboard.ai/api-guide>
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AddMessage(BaseModel):
    role: str = Field(..., description="消息角色：user 或 assistant")
    content: str = Field(..., description="原始消息文本")
    timestamp: Optional[int] = Field(default=None, description="Unix 毫秒时间戳")


class AddRequest(BaseModel):
    request_id: str  # 幂等键：同一 request_id 重试不得重复写入
    messages: list[AddMessage]
    user_id: str
    session_id: str


class AddResponse(BaseModel):
    success: bool
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str
    options: Optional[list[str]] = None
    user_id: str
    top_k: int = Field(default=100, ge=0, le=500)


class SearchResultItem(BaseModel):
    id: str
    content: str
    score: Optional[float] = None
    created_at: Optional[str] = None  # ISO 8601


class SearchResponse(BaseModel):
    data: list[SearchResultItem]


class HealthResponse(BaseModel):
    status: str
    users: int
    memories: int
    embedding_backend: str
    auth_enabled: bool
