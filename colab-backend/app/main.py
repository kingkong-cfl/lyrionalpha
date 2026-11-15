"""FastAPI application entrypoint for Lyrion AI Music Studio."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core.job_queue import job_queue
from app.routers import generation, health, jobs, producer, projects


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
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
