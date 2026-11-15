"""GPU telemetry, Drive persistence, and queue health check endpoint."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.core.job_queue import job_queue
from app.core.model_manager import model_manager
from app.core.storage import storage
from app.schemas import HealthResponse

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    telemetry = model_manager.get_telemetry()
    drive_writable = await asyncio.to_thread(storage.is_writable)
    return HealthResponse(
        status="ok",
        mock_gpu=telemetry.mock_gpu,
        gpu_name=telemetry.gpu_name,
        vram_used_mb=telemetry.vram_used_mb,
        vram_total_mb=telemetry.vram_total_mb,
        active_model=telemetry.active_model,
        drive_root=str(storage.drive_root),
        drive_writable=drive_writable,
        queue_depth=job_queue.queue_depth(),
    )
