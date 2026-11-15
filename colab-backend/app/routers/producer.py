"""Zero-VRAM AI Producer endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas import ProducerEnhanceRequest, ProducerEnhanceResponse
from app.services.llm_producer import enhance_prompt

router = APIRouter(prefix="/api/v1/producer", tags=["producer"])


@router.post("/enhance", response_model=ProducerEnhanceResponse)
async def enhance_producer_spec(req: ProducerEnhanceRequest) -> ProducerEnhanceResponse:
    return await enhance_prompt(
        raw_prompt=req.raw_prompt,
        preferred_duration_sec=req.preferred_duration_sec,
    )
