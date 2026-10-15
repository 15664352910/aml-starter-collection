"""HTTP surface: POST /add, POST /search, GET /health.

Contract notes that shape this file:
  * Add must be synchronous — 200 only after the memory is searchable. No 202,
    no task id, no status endpoint.
  * The three IDs must echo the request exactly, and ``success`` must be the
    boolean ``true``. Any deviation fails the stage even under HTTP 200.
  * Search returns ``{"data": [...]}`` — no ``items`` wrapper, never a bare array.
  * /health takes no auth and any 2xx counts. When no separate health URL is
    configured the platform probes ``/health`` on the Add origin, which is here.
"""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, Header, HTTPException

from . import config
from .mem0_backend import Mem0Unavailable
from .models import AddRequest, AddResponse, SearchRequest, SearchResponse
from .store import STORE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
_log = logging.getLogger("memory-service")

app = FastAPI(title="Agent Memory Challenge — Textual Memory", version="0.1.0")


def _presented_credentials(authorization: str | None, x_api_key: str | None) -> list[str]:
    """Every credential the caller offered, whichever scheme carried it."""
    found: list[str] = []
    if authorization:
        head, _, rest = authorization.partition(" ")
        value = rest.strip()
        if value and head.lower() in {"bearer", "token"}:
            found.append(value)
        elif not rest:
            # A bare "Authorization: <secret>" with no scheme.
            found.append(authorization.strip())
    if x_api_key and x_api_key.strip():
        found.append(x_api_key.strip())
    return found


async def require_auth(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
) -> None:
    """Authenticate per AUTH_MODE.

    In the default 'auto' mode any of the three accepted schemes will do, since on
    the code-submission route the platform chooses the scheme and a mismatch would
    surface as a 401 — which the platform does not retry.
    """
    mode = config.AUTH_MODE

    if mode == "none":
        return
    if mode == "auto" and not config.API_KEY:
        # No secret provisioned; run open. Explicit modes still fail closed below.
        return
    if not config.API_KEY:
        raise HTTPException(status_code=401, detail={"reason": "server key not configured"})

    if mode == "auto":
        if config.API_KEY in _presented_credentials(authorization, x_api_key):
            return
        raise HTTPException(status_code=401, detail={"reason": "invalid credentials"})

    presented: str | None = None
    if mode in {"bearer", "token"}:
        scheme = "Bearer" if mode == "bearer" else "Token"
        if authorization and authorization.lower().startswith(f"{scheme.lower()} "):
            presented = authorization[len(scheme) + 1 :].strip()
    elif mode == "x-api-key":
        presented = (x_api_key or "").strip() or None
    else:
        raise HTTPException(status_code=401, detail={"reason": f"unknown AUTH_MODE {mode!r}"})

    if presented != config.API_KEY:
        raise HTTPException(status_code=401, detail={"reason": "invalid credentials"})


@app.get("/health")
async def health() -> dict[str, object]:
    """Unauthenticated liveness probe."""
    return {"status": "ok", **STORE.stats()}


@app.post("/add", response_model=AddResponse)
async def add(request: AddRequest, _: None = Depends(require_auth)) -> AddResponse:
    try:
        written = STORE.add(request)
    except Mem0Unavailable as error:
        # 503 is in the platform's Add retry set (408/409/425/429/5xx), so this
        # hands the write back for backoff instead of returning success for a
        # memory that was never stored.
        _log.error("add failed, asking for retry: request=%s %s", request.request_id, error)
        raise HTTPException(status_code=503, detail={"reason": str(error)}) from error

    _log.info(
        "add user=%s session=%s request=%s units=%d",
        request.user_id,
        request.session_id,
        request.request_id,
        written,
    )
    # Echo all three IDs verbatim — the platform compares them byte-for-byte.
    return AddResponse(
        success=True,
        request_id=request.request_id,
        user_id=request.user_id,
        session_id=request.session_id,
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, _: None = Depends(require_auth)) -> SearchResponse:
    items = STORE.search(request)
    _log.info(
        "search user=%s top_k=%d options=%d returned=%d",
        request.user_id,
        request.top_k,
        len(request.options or []),
        len(items),
    )
    return SearchResponse(data=items)
