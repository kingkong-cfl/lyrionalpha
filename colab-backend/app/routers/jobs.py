"""Async job status polling endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.job_queue import job_queue
from app.schemas import JobStatusResponse

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    record = job_queue.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return record.to_response()
