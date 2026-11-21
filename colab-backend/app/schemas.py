"""Pydantic v2 data contracts mirroring shared/schemas/project.schema.json."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


TrackType = Literal["audio", "stem", "midi", "mic_recording"]
StemRole = Literal["full_mix", "vocals", "drums", "bass", "other", "custom"]
RepaintMode = Literal["repaint_and_restem", "single_track_repaint", "add_instrument_layer", "lego_layer"]
JobStatusType = Literal["queued", "running", "completed", "failed"]
EnhanceMode = Literal["all", "tags", "lyrics"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EqSettings(BaseModel):
    low_shelf_db: float = 0.0
    mid_gain_db: float = 0.0
    mid_freq_hz: float = 1000.0
    high_shelf_db: float = 0.0


class CompressorSettings(BaseModel):
    enabled: bool = False
    threshold_db: float = -18.0
    ratio: float = 2.5
    attack_ms: float = 10.0
    release_ms: float = 100.0


class ReverbSettings(BaseModel):
    enabled: bool = False
    room_size: float = 0.4
    damping: float = 0.5
    wet_level: float = 0.15


class TrackDspChain(BaseModel):
    eq: EqSettings = Field(default_factory=EqSettings)
    compressor: CompressorSettings = Field(default_factory=CompressorSettings)
    reverb: ReverbSettings = Field(default_factory=ReverbSettings)


class TrackSchema(BaseModel):
    track_id: str
    name: str
    track_type: TrackType
    stem_role: StemRole
    parent_track_id: Optional[str] = None
    file_rel_path: Optional[str] = None
    midi_rel_path: Optional[str] = None
    revision: int = 1
    sample_rate: int = 44100
    start_sample: int = 0
    end_sample: int = 0
    muted: bool = False
    soloed: bool = False
    volume_db: float = 0.0
    pan: float = Field(default=0.0, ge=-1.0, le=1.0)
    dsp_chain: TrackDspChain = Field(default_factory=TrackDspChain)
    plugin_metadata: dict[str, Any] = Field(default_factory=dict)


class BeatGridSchema(BaseModel):
    sample_rate: int = 44100
    total_samples: int = 0
    duration_sec: float = 0.0
    bpm: float = 120.0
    time_signature: str = "4/4"
    musical_key: str = "A minor"
    beat_timestamps: list[float] = Field(default_factory=list)
    beat_samples: list[int] = Field(default_factory=list)
    downbeat_beat_indices: list[int] = Field(default_factory=list)


class ProducerSpecSchema(BaseModel):
    raw_prompt: str = ""
    style_tags: list[str] = Field(default_factory=list)
    structured_lyrics: str = ""
    vocal_language: Optional[str] = None
    seed: Optional[int] = None
    guidance_scale: float = 1.0
    inference_steps: int = 8
    shift: float = 3.0


class RepaintHistoryEntry(BaseModel):
    repaint_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    mode: RepaintMode = "repaint_and_restem"
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    crossfade_samples: int
    prompt_override: Optional[str] = None
    lyrics_override: Optional[str] = None
    target_track_id: Optional[str] = None
    previous_revision: int
    new_revision: int
    previous_files: dict[str, str] = Field(default_factory=dict)
    previous_track_revisions: dict[str, int] = Field(default_factory=dict)


class ProjectSchema(BaseModel):
    schema_version: str = "1.0.0"
    project_id: str
    name: str
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    grid: BeatGridSchema = Field(default_factory=BeatGridSchema)
    producer_spec: ProducerSpecSchema = Field(default_factory=ProducerSpecSchema)
    tracks: list[TrackSchema] = Field(default_factory=list)
    repaint_history: list[RepaintHistoryEntry] = Field(default_factory=list)


# --- API Request / Response Contracts ---


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    mock_gpu: bool
    gpu_name: str
    vram_used_mb: float
    vram_total_mb: float
    active_model: str
    drive_root: str
    drive_writable: bool
    queue_depth: int


class ProducerEnhanceRequest(BaseModel):
    raw_prompt: str = Field(..., min_length=1)
    preferred_duration_sec: Optional[float] = Field(default=None, ge=8.0, le=240.0)
    mode: EnhanceMode = "all"
    existing_style_tags: Optional[list[str]] = None
    existing_lyrics: Optional[str] = None


class ProducerEnhanceResponse(BaseModel):
    provider: str
    mode: EnhanceMode = "all"
    bpm: float
    musical_key: str
    time_signature: str = "4/4"
    duration_sec: float
    style_tags: list[str]
    structured_lyrics: str


class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    bpm: float = Field(default=120.0, gt=20.0, le=300.0)
    musical_key: str = "A minor"
    raw_prompt: str = ""


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    producer_spec: Optional[ProducerSpecSchema] = None
    tracks: Optional[list[TrackSchema]] = None


class GenerateSongRequest(BaseModel):
    raw_prompt: str = ""
    style_tags: list[str] = Field(default_factory=list)
    structured_lyrics: str = ""
    vocal_language: Optional[str] = None
    bpm: Optional[float] = Field(default=None, gt=20.0, le=300.0)
    musical_key: Optional[str] = None
    duration_sec: Optional[float] = Field(default=None, ge=8.0, le=240.0)
    seed: Optional[int] = None
    guidance_scale: float = 1.0
    inference_steps: int = 8
    shift: float = 3.0
    separate_stems: bool = True


class RepaintRegionRequest(BaseModel):
    start_sample: int = Field(..., ge=0)
    end_sample: int = Field(..., gt=0)
    prompt_override: Optional[str] = None
    lyrics_override: Optional[str] = None
    style_tags: Optional[list[str]] = None
    seed: Optional[int] = None
    guidance_scale: float = 1.0
    inference_steps: int = 8
    shift: float = 3.0
    restem_window: bool = True


class LegoLayerRequest(BaseModel):
    reference_track_id: str
    target_stem_role: StemRole = "other"
    start_sample: int = Field(..., ge=0)
    end_sample: int = Field(..., gt=0)
    prompt_override: Optional[str] = None
    seed: Optional[int] = None
    guidance_scale: float = 1.0
    inference_steps: int = 8
    shift: float = 3.0


class SingleTrackRepaintRequest(BaseModel):
    target_track_id: str
    start_sample: int = Field(..., ge=0)
    end_sample: int = Field(..., gt=0)
    prompt_override: Optional[str] = None
    lyrics_override: Optional[str] = None
    seed: Optional[int] = None
    guidance_scale: float = 1.0
    inference_steps: int = 8
    shift: float = 3.0


class AddInstrumentLayerRequest(BaseModel):
    track_name: str = Field(..., min_length=1, max_length=80)
    instrument_prompt: str = Field(..., min_length=1)
    start_sample: int = Field(..., ge=0)
    end_sample: int = Field(..., gt=0)
    seed: Optional[int] = None
    guidance_scale: float = 1.0
    inference_steps: int = 8
    shift: float = 3.0


class ExportProjectRequest(BaseModel):
    format: Literal["wav", "stems_zip", "single_track_wav"] = "wav"
    track_id: Optional[str] = None
    start_sample: Optional[int] = Field(default=None, ge=0)
    end_sample: Optional[int] = Field(default=None, gt=0)
    include_muted: bool = False
    normalize_lufs: bool = True


class JobCreatedResponse(BaseModel):
    job_id: str
    project_id: str
    job_type: str
    status: JobStatusType = "queued"


class JobStatusResponse(BaseModel):
    job_id: str
    project_id: str
    job_type: str
    status: JobStatusType
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    stage: str
    vram_allocated_mb: float = 0.0
    created_at: str
    updated_at: str
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
