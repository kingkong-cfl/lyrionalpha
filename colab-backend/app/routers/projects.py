"""Project CRUD and HTTP Byte-Range WAV streaming endpoints."""

from __future__ import annotations

import asyncio
import mimetypes
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse

from app.core.storage import storage
from app.schemas import CreateProjectRequest, ProjectSchema, TrackSchema, UpdateProjectRequest

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])

_RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")


def merge_track_mixer_updates(project: ProjectSchema, incoming_tracks: list[TrackSchema]) -> None:
    """Apply only client mixer/DSP/label fields onto server tracks without touching server file paths or revisions."""
    incoming_by_id = {t.track_id: t for t in incoming_tracks}
    for srv_track in project.tracks:
        patch = incoming_by_id.get(srv_track.track_id)
        if patch is None:
            continue
        srv_track.name = patch.name.strip() or srv_track.name
        srv_track.muted = patch.muted
        srv_track.soloed = patch.soloed
        srv_track.volume_db = patch.volume_db
        srv_track.pan = patch.pan
        srv_track.dsp_chain = patch.dsp_chain
        srv_track.plugin_metadata = patch.plugin_metadata


@router.get("", response_model=list[ProjectSchema])
async def list_projects() -> list[ProjectSchema]:
    return await asyncio.to_thread(storage.list_projects)


@router.post("", response_model=ProjectSchema, status_code=201)
async def create_project(req: CreateProjectRequest) -> ProjectSchema:
    return await asyncio.to_thread(
        storage.create_project,
        name=req.name,
        bpm=req.bpm,
        musical_key=req.musical_key,
        raw_prompt=req.raw_prompt,
    )


@router.get("/{project_id}", response_model=ProjectSchema)
async def get_project(project_id: str) -> ProjectSchema:
    try:
        return await asyncio.to_thread(storage.load_project, project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/{project_id}", response_model=ProjectSchema)
async def update_project(project_id: str, req: UpdateProjectRequest) -> ProjectSchema:
    async with storage.get_project_lock(project_id):
        try:
            project = await asyncio.to_thread(storage.load_project, project_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if req.name is not None:
            project.name = req.name.strip() or project.name
        if req.producer_spec is not None:
            project.producer_spec = req.producer_spec
        if req.tracks is not None:
            merge_track_mixer_updates(project, req.tracks)

        return await asyncio.to_thread(storage.save_project, project)


def _iter_file_range(path: Path, start: int, end: int, chunk_size: int = 65536):
    with path.open("rb") as fh:
        fh.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = fh.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def build_audio_file_response(
    project_id: str,
    rel_path: str,
    request_url: str,
    range_header: str | None,
    preview: bool = False,
) -> Response:
    """Resolve project audio/export file (with optional compact preview transcoding) and Byte-Range handling."""
    try:
        file_path = storage.resolve_project_file(project_id, rel_path)
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio or export file '{rel_path}' not found.")

    media_type, _ = mimetypes.guess_type(str(file_path))
    if preview and file_path.suffix.lower() == ".wav":
        file_path, media_type = storage.get_or_create_preview_audio(project_id, rel_path)
    elif file_path.suffix.lower() == ".wav":
        media_type = "audio/wav"
    elif file_path.suffix.lower() == ".flac":
        media_type = "audio/flac"
    elif file_path.suffix.lower() == ".zip":
        media_type = "application/zip"
    media_type = media_type or "application/octet-stream"

    file_size = file_path.stat().st_size
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=31536000, immutable" if "rev=" in request_url else "no-cache",
    }

    if range_header:
        match = _RANGE_RE.match(range_header.strip())
        if not match:
            raise HTTPException(status_code=416, detail="Invalid Range header")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else file_size - 1
        if start >= file_size or end < start:
            raise HTTPException(
                status_code=416,
                detail="Requested range not satisfiable",
                headers={"Content-Range": f"bytes */{file_size}"},
            )
        end = min(end, file_size - 1)
        content_length = end - start + 1
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        headers["Content-Length"] = str(content_length)
        return StreamingResponse(
            _iter_file_range(file_path, start, end),
            status_code=206,
            media_type=media_type,
            headers=headers,
        )

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name if file_path.suffix.lower() in (".zip", ".wav") else None,
        headers=headers,
    )


@router.get("/{project_id}/audio/{rel_path:path}")
async def stream_project_audio(
    project_id: str,
    rel_path: str,
    request: Request,
    preview: bool = False,
) -> Response:
    return await asyncio.to_thread(
        build_audio_file_response,
        project_id,
        rel_path,
        str(request.url),
        request.headers.get("range"),
        preview,
    )
