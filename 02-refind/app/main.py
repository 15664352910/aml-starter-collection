from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from . import __version__
from .config import Settings
from .models import (
    AddRequest,
    AddResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from .retriever import ReFindRetriever
from .store import ConflictingRequestError, MemoryStore

LOGGER = logging.getLogger("refind")


def create_app(
    settings: Settings | None = None,
    store: MemoryStore | None = None,
    retriever: ReFindRetriever | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_store = store or MemoryStore(
        resolved_settings.database_path,
        retention_days=resolved_settings.retention_days,
    )
    resolved_retriever = retriever or ReFindRetriever(resolved_store, resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        resolved_store.delete_expired()
        yield

    application = FastAPI(
        title="ReFind Agent Memory API",
        version=__version__,
        description="Agent Memory Leaderboard-compatible Add/Search service.",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.store = resolved_store
    application.state.retriever = resolved_retriever

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            version=__version__,
            retrieval_mode=resolved_settings.retrieval_mode,
            agent_configured=bool(resolved_retriever.planner),
        )

    def add_memory(payload: AddRequest) -> AddResponse:
        try:
            resolved_store.add(payload)
        except ConflictingRequestError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return AddResponse(
            request_id=payload.request_id,
            user_id=payload.user_id,
            session_id=payload.session_id,
        )

    def search_memory(payload: SearchRequest) -> SearchResponse:
        try:
            results, stats = resolved_retriever.search(
                query=payload.query,
                options=payload.options,
                user_id=payload.user_id,
                top_k=payload.top_k,
            )
        except RuntimeError as error:
            LOGGER.exception("Search failed for an evaluation request")
            raise HTTPException(
                status_code=502, detail="retrieval backend failed"
            ) from error
        LOGGER.info(
            "Search completed: results=%d iterations=%d searches=%d tokens=%d",
            len(results),
            stats.iterations,
            stats.searches,
            stats.prompt_tokens + stats.completion_tokens,
        )
        return SearchResponse(data=results)

    application.add_api_route(
        "/v1/memories/add", add_memory, methods=["POST"], response_model=AddResponse
    )
    application.add_api_route(
        "/v1/memories/search",
        search_memory,
        methods=["POST"],
        response_model=SearchResponse,
    )
    # Short aliases make the container compatible with evaluators that request
    # endpoint paths separately rather than assuming the versioned paths.
    application.add_api_route(
        "/add",
        add_memory,
        methods=["POST"],
        response_model=AddResponse,
        include_in_schema=False,
    )
    application.add_api_route(
        "/search",
        search_memory,
        methods=["POST"],
        response_model=SearchResponse,
        include_in_schema=False,
    )
    return application


app = create_app()
