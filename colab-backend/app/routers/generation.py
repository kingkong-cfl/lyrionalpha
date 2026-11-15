"""Generation, sample-accurate Repaint, Lego, Stem Separation, Undo, and Export routes."""

from __future__ import annotations

import asyncio
import re
import secrets
import uuid
from typing import Any, Callable

import numpy as np
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.job_queue import ProgressCallback, job_queue
from app.core.storage import storage
from app.schemas import (
    AddInstrumentLayerRequest,
    ExportProjectRequest,
    GenerateSongRequest,
    JobCreatedResponse,
    LegoLayerRequest,
    ProducerSpecSchema,
    ProjectSchema,
    RepaintHistoryEntry,
    RepaintRegionRequest,
    SingleTrackRepaintRequest,
    TrackSchema,
)
from app.services.acestep_service import generate_full_song, repaint_region
from app.services.beat_service import analyze_beat_grid
from app.services.demucs_service import separate_stems
from app.services.dsp_service import export_project_audio
from app.services.splice_service import (
    apply_region_fade_edges,
    compute_context_window,
    equal_power_crossfade_splice,
)

router = APIRouter(prefix="/api/v1/projects/{project_id}", tags=["generation"])


async def _load_project_or_404(project_id: str) -> ProjectSchema:
    try:
        return await asyncio.to_thread(storage.load_project, project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _enforce_reference_mix_muted(project: ProjectSchema) -> None:
    """Keep Full Mix (Reference) muted when stem tracks exist so muting stems yields silence."""
    has_stems = any(t.stem_role != "full_mix" for t in project.tracks)
    if has_stems:
        for t in project.tracks:
            if t.stem_role == "full_mix":
                t.muted = True


def _resolve_job_seed(explicit_seed: int | None) -> int:
    """Return explicit seed override or a fresh random seed for each repaint/layer job."""
    if explicit_seed is not None:
        return int(explicit_seed)
    return secrets.randbelow(2_000_000_000) + 1


def _unique_wav_rel(folder: str, stem_slug: str, rev: int) -> str:
    """Generate a unique immutable asset path so browser cache never collides after undo."""
    return f"{folder}/{stem_slug}_rev{rev}_{uuid.uuid4().hex[:6]}.wav"


def sum_raw_stems(
    project_id: str,
    tracks: list[TrackSchema],
    total_samples: int,
    sr: int,
) -> np.ndarray | None:
    """Sum all raw stem and custom-layer WAV files at unity gain (ignoring mute/solo/volume/pan)."""
    stem_tracks = [t for t in tracks if t.stem_role != "full_mix" and t.file_rel_path]
    if not stem_tracks:
        return None
    mix = np.zeros((max(1, total_samples), 2), dtype=np.float32)
    for st in stem_tracks:
        buf, _ = storage.read_wav(project_id, st.file_rel_path or "", sr)
        usable = min(mix.shape[0], buf.shape[0])
        mix[:usable] += buf[:usable]
    peak = float(np.max(np.abs(mix))) if mix.size else 1.0
    if peak > 0.98:
        mix = mix * (0.98 / peak)
    return np.ascontiguousarray(mix, dtype=np.float32)


async def _rebuild_and_write_master_from_stems(
    project_id: str,
    tracks: list[TrackSchema],
    total_samples: int,
    sr: int,
    master_track: TrackSchema,
) -> tuple[str, int] | None:
    """Re-render raw unity-gain sum(stems) to a new unique master WAV asset."""
    raw_sum = await asyncio.to_thread(sum_raw_stems, project_id, tracks, total_samples, sr)
    if raw_sum is None:
        return None
    next_rev = master_track.revision + 1
    master_rel = _unique_wav_rel("raw", "master_mix", next_rev)
    await asyncio.to_thread(storage.write_wav, project_id, master_rel, raw_sum, sr)
    return master_rel, next_rev


async def _commit_project_with_lock(
    project_id: str,
    mutate_fn: Callable[[ProjectSchema], None],
) -> ProjectSchema:
    """Hold per-project lock only for millisecond read-merge-write so mixer saves preserve changes."""
    async with storage.get_project_lock(project_id):
        latest = await asyncio.to_thread(storage.load_project, project_id)
        mutate_fn(latest)
        _enforce_reference_mix_muted(latest)
        return await asyncio.to_thread(storage.save_project, latest)


@router.post("/generate", response_model=JobCreatedResponse, status_code=202)
async def trigger_generate_song(project_id: str, req: GenerateSongRequest) -> JobCreatedResponse:
    await _load_project_or_404(project_id)

    async def _run(on_progress: ProgressCallback) -> dict[str, Any]:
        on_progress(0.10, "Loading ACE-Step 1.5 (bfloat16) into VRAM...")
        master_audio, sr = await generate_full_song(
            bpm=req.bpm,
            musical_key=req.musical_key,
            duration_sec=req.duration_sec,
            style_tags=req.style_tags,
            structured_lyrics=req.structured_lyrics,
            seed=req.seed,
            guidance_scale=req.guidance_scale,
            inference_steps=req.inference_steps,
        )

        total_samples = int(master_audio.shape[0])
        master_rel = _unique_wav_rel("raw", "master_mix", 1)
        on_progress(0.45, "Writing canonical 44.1kHz master mix to Drive...")
        await asyncio.to_thread(storage.write_wav, project_id, master_rel, master_audio, sr)

        on_progress(0.55, "Extracting CPU librosa beat grid & sample timestamps...")
        grid = await asyncio.to_thread(
            analyze_beat_grid,
            master_audio,
            sr,
            req.bpm,
            req.musical_key,
            "4/4",
        )

        tracks: list[TrackSchema] = [
            TrackSchema(
                track_id="trk_master_01",
                name="Full Mix (Reference)",
                track_type="audio",
                stem_role="full_mix",
                parent_track_id=None,
                file_rel_path=master_rel,
                revision=1,
                sample_rate=sr,
                start_sample=0,
                end_sample=total_samples,
                muted=req.separate_stems,
            )
        ]

        if req.separate_stems:
            on_progress(0.65, "Swapping to Demucs (htdemucs) for 4-stem separation...")
            stems = await separate_stems(master_audio, sample_rate=sr)
            on_progress(0.88, "Saving 4 separated stems (vocals, drums, bass, other) to Drive...")
            stem_display = {
                "vocals": "Vocals",
                "drums": "Drums",
                "bass": "Bass",
                "other": "Other / Synths",
            }
            for role in ("vocals", "drums", "bass", "other"):
                stem_buf = stems.get(role)
                if stem_buf is None:
                    continue
                stem_rel = _unique_wav_rel("stems", role, 1)
                await asyncio.to_thread(storage.write_wav, project_id, stem_rel, stem_buf, sr)
                tracks.append(
                    TrackSchema(
                        track_id=f"trk_{role}_01",
                        name=stem_display[role],
                        track_type="stem",
                        stem_role=role,  # type: ignore[arg-type]
                        parent_track_id="trk_master_01",
                        file_rel_path=stem_rel,
                        revision=1,
                        sample_rate=sr,
                        start_sample=0,
                        end_sample=total_samples,
                        muted=False,
                    )
                )

        def _apply(proj: ProjectSchema) -> None:
            proj.grid = grid
            proj.producer_spec = ProducerSpecSchema(
                raw_prompt=req.raw_prompt,
                style_tags=req.style_tags,
                structured_lyrics=req.structured_lyrics,
                seed=req.seed,
                guidance_scale=req.guidance_scale,
                inference_steps=req.inference_steps,
            )
            proj.tracks = tracks
            proj.repaint_history = []

        saved = await _commit_project_with_lock(project_id, _apply)
        return {"project": saved.model_dump(mode="json")}

    record = job_queue.enqueue(project_id, "generate", _run)
    return JobCreatedResponse(job_id=record.job_id, project_id=project_id, job_type="generate")


@router.post("/repaint", response_model=JobCreatedResponse, status_code=202)
async def trigger_repaint_region(project_id: str, req: RepaintRegionRequest) -> JobCreatedResponse:
    project = await _load_project_or_404(project_id)
    master_track = next((t for t in project.tracks if t.stem_role == "full_mix"), None)
    if master_track is None or not master_track.file_rel_path:
        raise HTTPException(status_code=400, detail="Project has no master mix yet. Generate a song first.")

    if req.end_sample <= req.start_sample or req.end_sample > project.grid.total_samples:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sample bounds [{req.start_sample}, {req.end_sample}] for total_samples={project.grid.total_samples}.",
        )

    async def _run(on_progress: ProgressCallback) -> dict[str, Any]:
        proj = await asyncio.to_thread(storage.load_project, project_id)
        if req.end_sample <= req.start_sample or req.end_sample > proj.grid.total_samples:
            raise ValueError(
                f"Project changed before repaint execution: invalid sample bounds [{req.start_sample}, {req.end_sample}]."
            )

        m_track = next((t for t in proj.tracks if t.stem_role == "full_mix"), None)
        if m_track is None or not m_track.file_rel_path:
            raise ValueError("Project no longer contains a master mix track.")

        sr = proj.grid.sample_rate or settings.sample_rate
        total_samples = proj.grid.total_samples
        old_rev = m_track.revision
        new_rev = old_rev + 1
        seed = _resolve_job_seed(req.seed)

        previous_files: dict[str, str] = {}
        previous_track_revisions: dict[str, int] = {}
        for t in proj.tracks:
            if t.file_rel_path:
                previous_files[t.track_id] = t.file_rel_path
                previous_track_revisions[t.track_id] = t.revision

        on_progress(0.12, "Reconstructing raw stem sum for phase-coherent conditioning...")
        source_mix = await asyncio.to_thread(sum_raw_stems, project_id, proj.tracks, total_samples, sr)
        if source_mix is None:
            source_mix, _ = await asyncio.to_thread(storage.read_wav, project_id, m_track.file_rel_path or "", sr)

        tags = req.style_tags if req.style_tags is not None else proj.producer_spec.style_tags
        if req.prompt_override:
            tags = [req.prompt_override] + [t for t in tags if t != req.prompt_override]
        lyrics = req.lyrics_override if req.lyrics_override is not None else proj.producer_spec.structured_lyrics

        on_progress(0.28, "Running ACE-Step 1.5 sample-accurate regional repaint...")
        new_slice = await repaint_region(
            source_mix=source_mix,
            start_sample=req.start_sample,
            end_sample=req.end_sample,
            bpm=proj.grid.bpm,
            musical_key=proj.grid.musical_key,
            style_tags=tags,
            lyrics_override=lyrics,
            seed=seed,
            inference_steps=proj.producer_spec.inference_steps,
            guidance_scale=proj.producer_spec.guidance_scale,
            target_layer_role="full",
        )

        on_progress(0.50, "Applying 50ms equal-power cosine crossfade splice...")
        spliced_mix = await asyncio.to_thread(
            equal_power_crossfade_splice,
            source_mix,
            new_slice,
            req.start_sample,
            req.end_sample,
            settings.crossfade_samples,
            "equal_power",
        )
        new_master_rel = _unique_wav_rel("raw", "master_mix", new_rev)
        await asyncio.to_thread(storage.write_wav, project_id, new_master_rel, spliced_mix, sr)

        stem_updates: dict[str, tuple[str, int]] = {}
        stem_tracks = [
            t for t in proj.tracks if t.track_type == "stem" and t.stem_role in ("vocals", "drums", "bass", "other")
        ]
        if req.restem_window and stem_tracks:
            on_progress(0.65, "Swapping to Demucs (htdemucs) for windowed stem re-sync...")
            win_start, win_end, inner_start, inner_end = compute_context_window(
                req.start_sample,
                req.end_sample,
                total_samples,
                settings.demucs_context_samples,
            )
            windowed_mix = spliced_mix[win_start:win_end]
            windowed_stems = await separate_stems(windowed_mix, sample_rate=sr)

            on_progress(0.85, "Applying equal-gain linear crossfade into stems (bit-identical outside splice)...")
            for st_track in stem_tracks:
                role = st_track.stem_role
                if role not in windowed_stems or not st_track.file_rel_path:
                    continue
                existing_stem, _ = await asyncio.to_thread(storage.read_wav, project_id, st_track.file_rel_path, sr)
                clean_slice = windowed_stems[role][inner_start:inner_end]
                spliced_stem = await asyncio.to_thread(
                    equal_power_crossfade_splice,
                    existing_stem,
                    clean_slice,
                    req.start_sample,
                    req.end_sample,
                    settings.crossfade_samples,
                    "equal_gain",
                )
                st_new_rev = st_track.revision + 1
                new_stem_rel = _unique_wav_rel("stems", role, st_new_rev)
                await asyncio.to_thread(storage.write_wav, project_id, new_stem_rel, spliced_stem, sr)
                stem_updates[st_track.track_id] = (new_stem_rel, st_new_rev)

        def _apply(latest: ProjectSchema) -> None:
            for tr in latest.tracks:
                if tr.track_id == m_track.track_id:
                    tr.file_rel_path = new_master_rel
                    tr.revision = new_rev
                elif tr.track_id in stem_updates:
                    tr.file_rel_path, tr.revision = stem_updates[tr.track_id]

            latest.repaint_history.append(
                RepaintHistoryEntry(
                    repaint_id=f"rep_{uuid.uuid4().hex[:8]}",
                    mode="repaint_and_restem",
                    start_sample=req.start_sample,
                    end_sample=req.end_sample,
                    start_sec=round(req.start_sample / float(sr), 6),
                    end_sec=round(req.end_sample / float(sr), 6),
                    crossfade_samples=settings.crossfade_samples,
                    prompt_override=req.prompt_override,
                    lyrics_override=req.lyrics_override,
                    previous_revision=old_rev,
                    new_revision=new_rev,
                    previous_files=previous_files,
                    previous_track_revisions=previous_track_revisions,
                )
            )

        saved = await _commit_project_with_lock(project_id, _apply)
        return {"project": saved.model_dump(mode="json")}

    record = job_queue.enqueue(project_id, "repaint", _run)
    return JobCreatedResponse(job_id=record.job_id, project_id=project_id, job_type="repaint")


@router.post("/repaint-track", response_model=JobCreatedResponse, status_code=202)
async def trigger_single_track_repaint(project_id: str, req: SingleTrackRepaintRequest) -> JobCreatedResponse:
    project = await _load_project_or_404(project_id)
    target = next((t for t in project.tracks if t.track_id == req.target_track_id), None)
    if target is None or not target.file_rel_path:
        raise HTTPException(status_code=404, detail=f"Target track '{req.target_track_id}' not found.")
    if req.end_sample <= req.start_sample or req.end_sample > project.grid.total_samples:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sample bounds [{req.start_sample}, {req.end_sample}] for total_samples={project.grid.total_samples}.",
        )

    async def _run(on_progress: ProgressCallback) -> dict[str, Any]:
        proj = await asyncio.to_thread(storage.load_project, project_id)
        t_track = next((t for t in proj.tracks if t.track_id == req.target_track_id), None)
        if t_track is None or not t_track.file_rel_path:
            raise ValueError(f"Target track '{req.target_track_id}' is missing.")
        if req.end_sample <= req.start_sample or req.end_sample > proj.grid.total_samples:
            raise ValueError(
                f"Invalid sample bounds [{req.start_sample}, {req.end_sample}] for project length."
            )

        sr = proj.grid.sample_rate or settings.sample_rate
        total_samples = proj.grid.total_samples
        old_rev = t_track.revision
        new_rev = old_rev + 1
        seed = _resolve_job_seed(req.seed)
        prev_rel = t_track.file_rel_path

        previous_files: dict[str, str] = {t_track.track_id: prev_rel}
        previous_track_revisions: dict[str, int] = {t_track.track_id: old_rev}

        on_progress(0.20, "Summing untouched raw stems for harmonic/rhythmic conditioning...")
        context_mix = np.zeros((total_samples, 2), dtype=np.float32)
        other_tracks = [
            t for t in proj.tracks
            if t.track_id != t_track.track_id and t.file_rel_path and t.stem_role != "full_mix"
        ]
        if not other_tracks:
            other_tracks = [t for t in proj.tracks if t.file_rel_path]
        for ot in other_tracks:
            buf, _ = await asyncio.to_thread(storage.read_wav, project_id, ot.file_rel_path or "", sr)
            usable = min(total_samples, buf.shape[0])
            context_mix[:usable] += buf[:usable]

        tags = list(proj.producer_spec.style_tags)
        if req.prompt_override:
            tags.insert(0, req.prompt_override)
        lyrics = req.lyrics_override if req.lyrics_override is not None else proj.producer_spec.structured_lyrics

        on_progress(0.45, f"Running ACE-Step 1.5 to repaint only '{t_track.name}'...")
        new_slice = await repaint_region(
            source_mix=context_mix,
            start_sample=req.start_sample,
            end_sample=req.end_sample,
            bpm=proj.grid.bpm,
            musical_key=proj.grid.musical_key,
            style_tags=tags,
            lyrics_override=lyrics,
            seed=seed,
            inference_steps=proj.producer_spec.inference_steps,
            guidance_scale=proj.producer_spec.guidance_scale,
            target_layer_role=t_track.stem_role,
            instrument_hint=req.prompt_override or t_track.name,
        )

        # In real GPU mode, isolate only the requested stem role via windowed Demucs extraction
        if not settings.mock_gpu and t_track.stem_role in ("vocals", "drums", "bass", "other"):
            on_progress(0.65, f"Isolating '{t_track.stem_role}' stem via windowed Demucs...")
            temp_mix = context_mix.copy()
            temp_mix[req.start_sample : req.end_sample] = new_slice
            win_start, win_end, inner_start, inner_end = compute_context_window(
                req.start_sample,
                req.end_sample,
                total_samples,
                settings.demucs_context_samples,
            )
            windowed_stems = await separate_stems(temp_mix[win_start:win_end], sample_rate=sr)
            extracted = windowed_stems.get(t_track.stem_role)
            if extracted is not None and extracted.shape[0] >= inner_end:
                new_slice = extracted[inner_start:inner_end]

        on_progress(0.80, f"Applying 50ms crossfade splice into '{t_track.name}' (other tracks untouched)...")
        orig_audio, _ = await asyncio.to_thread(storage.read_wav, project_id, prev_rel, sr)
        spliced = await asyncio.to_thread(
            equal_power_crossfade_splice,
            orig_audio,
            new_slice,
            req.start_sample,
            req.end_sample,
            settings.crossfade_samples,
            "equal_power",
        )
        safe_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", t_track.name.strip().lower()).strip("_") or "track"
        new_rel = (
            _unique_wav_rel("raw", "master_mix", new_rev)
            if t_track.stem_role == "full_mix"
            else _unique_wav_rel("stems", f"{safe_slug}_{t_track.track_id}", new_rev)
        )
        await asyncio.to_thread(storage.write_wav, project_id, new_rel, spliced, sr)

        # Keep Full Mix synchronized with raw sum(stems) when a stem track is edited
        master_update: tuple[str, str, int] | None = None
        if t_track.stem_role != "full_mix":
            m_track = next((t for t in proj.tracks if t.stem_role == "full_mix" and t.file_rel_path), None)
            if m_track and m_track.file_rel_path:
                previous_files[m_track.track_id] = m_track.file_rel_path
                previous_track_revisions[m_track.track_id] = m_track.revision
                temp_tracks = [
                    (t.model_copy(update={"file_rel_path": new_rel, "revision": new_rev}) if t.track_id == t_track.track_id else t)
                    for t in proj.tracks
                ]
                rebuilt = await _rebuild_and_write_master_from_stems(project_id, temp_tracks, total_samples, sr, m_track)
                if rebuilt is not None:
                    master_update = (m_track.track_id, rebuilt[0], rebuilt[1])

        def _apply(latest: ProjectSchema) -> None:
            for tr in latest.tracks:
                if tr.track_id == t_track.track_id:
                    tr.file_rel_path = new_rel
                    tr.revision = new_rev
                elif master_update and tr.track_id == master_update[0]:
                    tr.file_rel_path = master_update[1]
                    tr.revision = master_update[2]

            latest.repaint_history.append(
                RepaintHistoryEntry(
                    repaint_id=f"rep_{uuid.uuid4().hex[:8]}",
                    mode="single_track_repaint",
                    start_sample=req.start_sample,
                    end_sample=req.end_sample,
                    start_sec=round(req.start_sample / float(sr), 6),
                    end_sec=round(req.end_sample / float(sr), 6),
                    crossfade_samples=settings.crossfade_samples,
                    prompt_override=req.prompt_override,
                    lyrics_override=req.lyrics_override,
                    target_track_id=t_track.track_id,
                    previous_revision=old_rev,
                    new_revision=new_rev,
                    previous_files=previous_files,
                    previous_track_revisions=previous_track_revisions,
                )
            )

        saved = await _commit_project_with_lock(project_id, _apply)
        return {"project": saved.model_dump(mode="json")}

    record = job_queue.enqueue(project_id, "repaint_track", _run)
    return JobCreatedResponse(job_id=record.job_id, project_id=project_id, job_type="repaint_track")


@router.post("/add-instrument", response_model=JobCreatedResponse, status_code=202)
async def trigger_add_instrument_layer(project_id: str, req: AddInstrumentLayerRequest) -> JobCreatedResponse:
    project = await _load_project_or_404(project_id)
    if req.end_sample <= req.start_sample or req.end_sample > project.grid.total_samples:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sample bounds [{req.start_sample}, {req.end_sample}] for total_samples={project.grid.total_samples}.",
        )
    if not project.tracks:
        raise HTTPException(status_code=400, detail="Generate a base track before adding an instrument layer.")

    async def _run(on_progress: ProgressCallback) -> dict[str, Any]:
        proj = await asyncio.to_thread(storage.load_project, project_id)
        if req.end_sample <= req.start_sample or req.end_sample > proj.grid.total_samples:
            raise ValueError(f"Invalid sample bounds [{req.start_sample}, {req.end_sample}].")

        sr = proj.grid.sample_rate or settings.sample_rate
        total_samples = proj.grid.total_samples
        max_rev = max((t.revision for t in proj.tracks), default=1)
        seed = _resolve_job_seed(req.seed)

        # Condition on raw unity-gain stem sum (ignoring mute/volume) so existing edits are heard
        on_progress(0.15, f"Conditioning '{req.track_name}' on raw mix stem context...")
        ref_track = next((t for t in proj.tracks if t.stem_role == "full_mix" and t.file_rel_path), None)
        cond_mix = await asyncio.to_thread(sum_raw_stems, project_id, proj.tracks, total_samples, sr)
        if cond_mix is None and ref_track and ref_track.file_rel_path:
            cond_mix, _ = await asyncio.to_thread(storage.read_wav, project_id, ref_track.file_rel_path, sr)
        if cond_mix is None:
            cond_mix = np.zeros((total_samples, 2), dtype=np.float32)

        tags = [req.instrument_prompt] + list(proj.producer_spec.style_tags)
        on_progress(0.45, f"Generating '{req.track_name}' via ACE-Step 1.5 across selected region...")
        raw_slice = await repaint_region(
            source_mix=cond_mix,
            start_sample=req.start_sample,
            end_sample=req.end_sample,
            bpm=proj.grid.bpm,
            musical_key=proj.grid.musical_key,
            style_tags=tags,
            lyrics_override="",
            seed=seed,
            inference_steps=proj.producer_spec.inference_steps,
            guidance_scale=proj.producer_spec.guidance_scale,
            target_layer_role="custom",
            instrument_hint=req.instrument_prompt,
        )

        on_progress(0.80, f"Applying 50ms fade edges into silent 44.1kHz track '{req.track_name}'...")
        faded_slice = await asyncio.to_thread(apply_region_fade_edges, raw_slice, settings.crossfade_samples)
        full_buf = np.zeros((total_samples, 2), dtype=np.float32)
        full_buf[req.start_sample : req.end_sample] = faded_slice

        track_id = f"trk_inst_{uuid.uuid4().hex[:6]}"
        safe_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", req.track_name.strip().lower()).strip("_") or "instrument"
        rel_path = _unique_wav_rel("stems", f"layer_{safe_slug}_{track_id}", 1)
        await asyncio.to_thread(storage.write_wav, project_id, rel_path, full_buf, sr)

        parent_ref = ref_track.track_id if ref_track else None
        new_track = TrackSchema(
            track_id=track_id,
            name=req.track_name.strip(),
            track_type="stem",
            stem_role="custom",
            parent_track_id=parent_ref,
            file_rel_path=rel_path,
            revision=1,
            sample_rate=sr,
            start_sample=0,
            end_sample=total_samples,
            muted=False,
            plugin_metadata={
                "instrument_prompt": req.instrument_prompt,
                "region_start_sample": req.start_sample,
                "region_end_sample": req.end_sample,
            },
        )

        previous_files: dict[str, str] = {}
        previous_track_revisions: dict[str, int] = {}
        master_update: tuple[str, str, int] | None = None
        if ref_track and ref_track.file_rel_path:
            previous_files[ref_track.track_id] = ref_track.file_rel_path
            previous_track_revisions[ref_track.track_id] = ref_track.revision
            rebuilt = await _rebuild_and_write_master_from_stems(
                project_id,
                list(proj.tracks) + [new_track],
                total_samples,
                sr,
                ref_track,
            )
            if rebuilt is not None:
                master_update = (ref_track.track_id, rebuilt[0], rebuilt[1])

        def _apply(latest: ProjectSchema) -> None:
            latest.tracks.append(new_track)
            if master_update:
                for tr in latest.tracks:
                    if tr.track_id == master_update[0]:
                        tr.file_rel_path = master_update[1]
                        tr.revision = master_update[2]
                        break
            latest.repaint_history.append(
                RepaintHistoryEntry(
                    repaint_id=f"rep_{uuid.uuid4().hex[:8]}",
                    mode="add_instrument_layer",
                    start_sample=req.start_sample,
                    end_sample=req.end_sample,
                    start_sec=round(req.start_sample / float(sr), 6),
                    end_sec=round(req.end_sample / float(sr), 6),
                    crossfade_samples=settings.crossfade_samples,
                    prompt_override=req.instrument_prompt,
                    target_track_id=track_id,
                    previous_revision=max_rev,
                    new_revision=max_rev + 1,
                    previous_files=previous_files,
                    previous_track_revisions=previous_track_revisions,
                )
            )

        saved = await _commit_project_with_lock(project_id, _apply)
        return {"project": saved.model_dump(mode="json")}

    record = job_queue.enqueue(project_id, "add_instrument", _run)
    return JobCreatedResponse(job_id=record.job_id, project_id=project_id, job_type="add_instrument")


@router.post("/lego", response_model=JobCreatedResponse, status_code=202)
async def trigger_lego_layer(project_id: str, req: LegoLayerRequest) -> JobCreatedResponse:
    project = await _load_project_or_404(project_id)
    ref_track = next((t for t in project.tracks if t.track_id == req.reference_track_id), None)
    if ref_track is None or not ref_track.file_rel_path:
        raise HTTPException(status_code=404, detail=f"Reference track '{req.reference_track_id}' not found.")
    if req.end_sample <= req.start_sample or req.end_sample > project.grid.total_samples:
        raise HTTPException(status_code=400, detail="Invalid sample range for Lego conditioning.")

    async def _run(on_progress: ProgressCallback) -> dict[str, Any]:
        proj = await asyncio.to_thread(storage.load_project, project_id)
        sr = proj.grid.sample_rate or settings.sample_rate
        total_samples = proj.grid.total_samples
        target_track = next(
            (t for t in proj.tracks if t.stem_role == req.target_stem_role and t.track_type == "stem"),
            None,
        )
        ref_t = next((t for t in proj.tracks if t.track_id == req.reference_track_id), None)
        if ref_t is None or not ref_t.file_rel_path:
            raise ValueError(f"Reference track '{req.reference_track_id}' is missing.")
        ref_audio, _ = await asyncio.to_thread(storage.read_wav, project_id, ref_t.file_rel_path, sr)
        prev_rev = max((t.revision for t in proj.tracks), default=1)
        new_rev = prev_rev + 1
        seed = _resolve_job_seed(req.seed)

        on_progress(0.30, f"Generating conditioned '{req.target_stem_role}' layer via ACE-Step 1.5...")
        tags = list(proj.producer_spec.style_tags)
        if req.prompt_override:
            tags.insert(0, req.prompt_override)

        new_slice = await repaint_region(
            source_mix=ref_audio,
            start_sample=req.start_sample,
            end_sample=req.end_sample,
            bpm=proj.grid.bpm,
            musical_key=proj.grid.musical_key,
            style_tags=tags,
            lyrics_override=proj.producer_spec.structured_lyrics,
            seed=seed,
            target_layer_role=req.target_stem_role,
        )

        previous_files: dict[str, str] = {}
        previous_track_revisions: dict[str, int] = {}
        new_lego_track: TrackSchema | None = None
        updated_rel = ""

        if target_track and target_track.file_rel_path:
            previous_files[target_track.track_id] = target_track.file_rel_path
            previous_track_revisions[target_track.track_id] = target_track.revision
            orig_stem, _ = await asyncio.to_thread(storage.read_wav, project_id, target_track.file_rel_path, sr)
            spliced = await asyncio.to_thread(
                equal_power_crossfade_splice,
                orig_stem,
                new_slice,
                req.start_sample,
                req.end_sample,
                settings.crossfade_samples,
                "equal_power",
            )
            updated_rel = _unique_wav_rel("stems", req.target_stem_role, new_rev)
            await asyncio.to_thread(storage.write_wav, project_id, updated_rel, spliced, sr)
            target_id = target_track.track_id
        else:
            full_buf = np.zeros((total_samples, 2), dtype=np.float32)
            full_buf[req.start_sample : req.end_sample] = new_slice
            rel_path = _unique_wav_rel("stems", f"lego_{req.target_stem_role}", new_rev)
            await asyncio.to_thread(storage.write_wav, project_id, rel_path, full_buf, sr)
            target_id = f"trk_lego_{uuid.uuid4().hex[:6]}"
            new_lego_track = TrackSchema(
                track_id=target_id,
                name=f"Lego Layer ({req.target_stem_role})",
                track_type="stem",
                stem_role=req.target_stem_role,
                parent_track_id=req.reference_track_id,
                file_rel_path=rel_path,
                revision=new_rev,
                sample_rate=sr,
                start_sample=0,
                end_sample=total_samples,
            )

        def _apply(latest: ProjectSchema) -> None:
            if target_track and updated_rel:
                for tr in latest.tracks:
                    if tr.track_id == target_track.track_id:
                        tr.file_rel_path = updated_rel
                        tr.revision = new_rev
                        break
            elif new_lego_track is not None:
                latest.tracks.append(new_lego_track)

            latest.repaint_history.append(
                RepaintHistoryEntry(
                    repaint_id=f"rep_{uuid.uuid4().hex[:8]}",
                    mode="lego_layer",
                    start_sample=req.start_sample,
                    end_sample=req.end_sample,
                    start_sec=round(req.start_sample / float(sr), 6),
                    end_sec=round(req.end_sample / float(sr), 6),
                    crossfade_samples=settings.crossfade_samples,
                    prompt_override=req.prompt_override,
                    target_track_id=target_id,
                    previous_revision=prev_rev,
                    new_revision=new_rev,
                    previous_files=previous_files,
                    previous_track_revisions=previous_track_revisions,
                )
            )

        saved = await _commit_project_with_lock(project_id, _apply)
        return {"project": saved.model_dump(mode="json")}

    record = job_queue.enqueue(project_id, "lego", _run)
    return JobCreatedResponse(job_id=record.job_id, project_id=project_id, job_type="lego")


@router.post("/undo-repaint", response_model=ProjectSchema)
async def undo_last_repaint(project_id: str) -> ProjectSchema:
    async with storage.get_project_lock(project_id):
        project = await _load_project_or_404(project_id)
        if not project.repaint_history:
            raise HTTPException(status_code=400, detail="No repaint history entries to undo.")

        last_entry = project.repaint_history.pop()
        if last_entry.mode == "add_instrument_layer" and last_entry.target_track_id:
            project.tracks = [t for t in project.tracks if t.track_id != last_entry.target_track_id]

        prev_rev = last_entry.previous_revision
        for track in project.tracks:
            if last_entry.previous_files and track.track_id in last_entry.previous_files:
                candidate_rel = last_entry.previous_files[track.track_id]
                restored_rev = last_entry.previous_track_revisions.get(track.track_id, prev_rev)
                target_path = storage.resolve_project_file(project_id, candidate_rel)
                if target_path.is_file():
                    track.file_rel_path = candidate_rel
                    track.revision = restored_rev

        _enforce_reference_mix_muted(project)
        return await asyncio.to_thread(storage.save_project, project)


@router.post("/export", response_model=JobCreatedResponse, status_code=202)
async def trigger_export_project(project_id: str, req: ExportProjectRequest) -> JobCreatedResponse:
    project = await _load_project_or_404(project_id)
    if req.format == "single_track_wav":
        if not req.track_id or not any(t.track_id == req.track_id for t in project.tracks):
            raise HTTPException(status_code=404, detail=f"Track '{req.track_id}' not found.")
    if (req.start_sample is not None) != (req.end_sample is not None):
        raise HTTPException(status_code=400, detail="Provide both start_sample and end_sample for region export.")
    if req.start_sample is not None and req.end_sample is not None:
        if req.end_sample <= req.start_sample or req.end_sample > project.grid.total_samples:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid export bounds [{req.start_sample}, {req.end_sample}] for total_samples={project.grid.total_samples}.",
            )

    async def _run(on_progress: ProgressCallback) -> dict[str, Any]:
        proj = await asyncio.to_thread(storage.load_project, project_id)
        on_progress(0.25, "Applying per-track Pedalboard EQ, Compressor & Reverb...")
        export_paths = await asyncio.to_thread(
            export_project_audio,
            proj,
            req.format,
            req.include_muted,
            req.normalize_lufs,
            req.track_id,
            req.start_sample,
            req.end_sample,
        )
        on_progress(0.95, "Finalizing lossless export in Drive exports/...")
        return {
            "format": req.format,
            "download_rel_path": export_paths["download_rel_path"],
            "master_rel_path": export_paths["master_rel_path"],
            "download_url": f"/api/v1/projects/{project_id}/audio/{export_paths['download_rel_path']}",
        }

    record = job_queue.enqueue(project_id, "export", _run)
    return JobCreatedResponse(job_id=record.job_id, project_id=project_id, job_type="export")
