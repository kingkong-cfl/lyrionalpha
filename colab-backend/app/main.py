"""FastAPI application entrypoint for Lyrion AI Music Studio."""

from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.job_queue import job_queue
from app.routers import generation, health, jobs, producer, projects

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def is_request_authorized(
    method: str,
    path: str,
    headers: dict[str, str],
    query_params: dict[str, str],
    expected_token: str | None = None,
) -> bool:
    """Validate Bearer header, X-Lyrion-Token header, or ?token= query parameter."""
    token_required = (
        expected_token if expected_token is not None else os.environ.get("LYRION_API_TOKEN", settings.api_token)
    ).strip()
    if not token_required:
        return True
    if method.upper() == "OPTIONS" or path in ("/", "/docs", "/openapi.json", "/redoc"):
        return True

    candidate = ""
    auth = headers.get("authorization") or headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        candidate = auth[7:].strip()
    if not candidate:
        candidate = (headers.get("x-lyrion-token") or headers.get("X-Lyrion-Token") or "").strip()
    if not candidate:
        candidate = (query_params.get("token") or "").strip()

    return bool(candidate) and secrets.compare_digest(candidate, token_required)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    job_queue.ensure_worker_started()
    try:
        yield
    finally:
        await job_queue.shutdown()


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
)


@app.middleware("http")
async def api_token_auth_middleware(request: Request, call_next):
    if not is_request_authorized(
        request.method,
        request.url.path,
        dict(request.headers),
        dict(request.query_params),
    ):
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized: valid LYRION_API_TOKEN required."},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


# Register CORSMiddleware AFTER auth middleware so CORS wraps 401 responses too
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "Content-Disposition"],
)

app.include_router(health.router)
app.include_router(producer.router)
app.include_router(projects.router)
app.include_router(generation.router)
app.include_router(jobs.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
