from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.memory import MemorySystem, load_settings

API_TOKEN = os.getenv("API_TOKEN", "").strip()


def _check_token(authorization: str | None = Header(default=None)) -> None:
    if not API_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


def get_memory() -> MemorySystem:
    settings = load_settings()
    return MemorySystem(settings=settings)


app = FastAPI(
    title="Hybrid Memory Search",
    description="Platform Add/Search + local upsert / dataset ingest",
    version="1.0.0",
)


# ---------- platform eval schemas ----------


class AddMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: int | float | None = None


class PlatformAddRequest(BaseModel):
    request_id: str
    messages: list[AddMessage] = Field(..., min_length=1)
    user_id: str
    session_id: str


class PlatformSearchRequest(BaseModel):
    query: str
    user_id: str
    top_k: int = Field(..., ge=1, le=1000)
    options: list[str] | None = None


# ---------- local / debug schemas ----------


class UpsertRequest(BaseModel):
    id: str = Field(..., description="Stable memory id, e.g. user:session_1")
    text: str
    bucket_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embed: bool = True
    skip_if_exists: bool = True


class AddDatasetRequest(BaseModel):
    data_path: str = Field(..., description="Path inside container, e.g. /app/data/xxx.json")
    bucket_id: str | None = None
    embed: bool = True
    skip_if_exists: bool = True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/buckets", dependencies=[Depends(_check_token)])
def list_buckets(mem: MemorySystem = Depends(get_memory)) -> dict[str, Any]:
    buckets = []
    for b in mem.store.list_buckets():
        buckets.append({"bucket_id": b, "count": len(mem.store.list_bucket(b))})
    return {"buckets": buckets}


@app.post("/v1/add", dependencies=[Depends(_check_token)])
def platform_add(req: PlatformAddRequest, mem: MemorySystem = Depends(get_memory)) -> dict[str, Any]:
    """Eval-platform Add: store full chunk under user_id, sync-embed, then 200."""
    try:
        return mem.add_platform_chunk(
            request_id=req.request_id,
            messages=[m.model_dump() for m in req.messages],
            user_id=req.user_id,
            session_id=req.session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/search", dependencies=[Depends(_check_token)])
def platform_search(
    req: PlatformSearchRequest, mem: MemorySystem = Depends(get_memory)
) -> dict[str, Any]:
    """Eval-platform Search: isolate by user_id, return ranked data[]."""
    data = mem.platform_search(
        user_id=req.user_id,
        query=req.query,
        top_k=req.top_k,
        options=req.options,
    )
    return {"data": data}


@app.post("/v1/upsert", dependencies=[Depends(_check_token)])
def upsert(req: UpsertRequest, mem: MemorySystem = Depends(get_memory)) -> dict[str, Any]:
    result = mem.upsert(
        id=req.id,
        text=req.text,
        metadata=req.metadata,
        bucket_id=req.bucket_id,
        embed=req.embed,
        skip_if_exists=req.skip_if_exists,
    )
    return result.__dict__


@app.post("/v1/add/dataset", dependencies=[Depends(_check_token)])
def add_dataset(req: AddDatasetRequest, mem: MemorySystem = Depends(get_memory)) -> dict[str, Any]:
    path = Path(req.data_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"data_path not found: {req.data_path}")
    report = mem.add_from_data(
        path,
        bucket_id=req.bucket_id,
        embed=req.embed,
        skip_if_exists=req.skip_if_exists,
        show_progress=False,
    )
    return report.__dict__


@app.post("/v1/add/upload", dependencies=[Depends(_check_token)])
async def add_upload(
    file: UploadFile = File(...),
    bucket_id: str | None = None,
    embed: bool = True,
    skip_if_exists: bool = True,
    mem: MemorySystem = Depends(get_memory),
) -> dict[str, Any]:
    """Upload a JSON/JSONL conversation file and ingest sessions (local helper)."""
    suffix = Path(file.filename or "data.json").suffix or ".json"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        report = mem.add_from_data(
            tmp_path,
            bucket_id=bucket_id,
            embed=embed,
            skip_if_exists=skip_if_exists,
            show_progress=False,
        )
        return report.__dict__
    finally:
        tmp_path.unlink(missing_ok=True)
