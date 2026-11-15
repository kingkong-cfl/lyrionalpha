"""Lightweight sequential asyncio job queue for GPU pipeline execution."""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from app.core.model_manager import model_manager
from app.schemas import JobStatusResponse, JobStatusType, utc_now_iso

logger = logging.getLogger("lyrion.job_queue")

ProgressCallback = Callable[[float, str], None]
JobCoroutineFactory = Callable[[ProgressCallback], Awaitable[dict[str, Any]]]


@dataclass
class JobRecord:
    job_id: str
    project_id: str
    job_type: str
    coro_factory: JobCoroutineFactory
    status: JobStatusType = "queued"
    progress: float = 0.0
    stage: str = "Queued for GPU worker..."
    vram_allocated_mb: float = 0.0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    def to_response(self) -> JobStatusResponse:
        return JobStatusResponse(
            job_id=self.job_id,
            project_id=self.project_id,
            job_type=self.job_type,
            status=self.status,
            progress=self.progress,
            stage=self.stage,
            vram_allocated_mb=self.vram_allocated_mb,
            created_at=self.created_at,
            updated_at=self.updated_at,
            result=self.result,
            error=self.error,
        )


class JobQueueManager:
    """Runs heavy AI music tasks sequentially so VRAM never fragments or OOMs."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task[None]] = None

    def ensure_worker_started(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop(), name="lyrion-gpu-worker")

    async def shutdown(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    def queue_depth(self) -> int:
        active = sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))
        return active

    def enqueue(self, project_id: str, job_type: str, coro_factory: JobCoroutineFactory) -> JobRecord:
        self.ensure_worker_started()
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        record = JobRecord(
            job_id=job_id,
            project_id=project_id,
            job_type=job_type,
            coro_factory=coro_factory,
        )
        self._jobs[job_id] = record
        self._queue.put_nowait(job_id)
        return record

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            record = self._jobs.get(job_id)
            if record is None:
                self._queue.task_done()
                continue

            def update_progress(progress: float, stage: str) -> None:
                record.progress = max(0.0, min(1.0, float(progress)))
                record.stage = stage
                record.vram_allocated_mb = model_manager.get_telemetry().vram_used_mb
                record.updated_at = utc_now_iso()

            try:
                record.status = "running"
                update_progress(0.05, f"Starting {record.job_type}...")
                result_payload = await record.coro_factory(update_progress)
                record.result = result_payload
                record.status = "completed"
                update_progress(1.0, "Completed and persisted to storage.")
            except Exception as exc:
                logger.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
                record.status = "failed"
                record.error = str(exc)
                record.stage = f"Failed: {exc}"
                record.vram_allocated_mb = model_manager.get_telemetry().vram_used_mb
                record.updated_at = utc_now_iso()
            finally:
                self._queue.task_done()


job_queue = JobQueueManager()
