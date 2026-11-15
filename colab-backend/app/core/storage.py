"""Atomic Google Drive storage manager and canonical 44.1kHz stereo WAV I/O."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from app.config import settings
from app.schemas import ProjectSchema, utc_now_iso

try:
    import soundfile as sf  # type: ignore
except ImportError:  # pragma: no cover - local CPU fallback
    sf = None  # type: ignore


class StorageManager:
    """Persists project.json, raw mixes, stems, and exports."""

    SUBDIRS = ("raw", "stems", "midi", "exports")

    def __init__(self, drive_root: Path | None = None) -> None:
        self.drive_root = (drive_root or settings.drive_root).resolve()
        self.projects_root = self.drive_root / "projects"
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self._project_locks: dict[str, asyncio.Lock] = {}
        self._writable_cache: tuple[float, bool] | None = None

    def get_project_lock(self, project_id: str) -> asyncio.Lock:
        return self._project_locks.setdefault(project_id, asyncio.Lock())

    def is_writable(self, ttl_sec: float = 60.0) -> bool:
        now = time.monotonic()
        if self._writable_cache is not None:
            cached_at, cached_result = self._writable_cache
            if (now - cached_at) < ttl_sec:
                return cached_result
        try:
            probe = self.drive_root / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            ok = True
        except OSError:
            ok = False
        self._writable_cache = (now, ok)
        return ok

    def project_dir(self, project_id: str) -> Path:
        safe_id = Path(project_id).name
        if not safe_id or safe_id != project_id:
            raise ValueError(f"Invalid project_id: {project_id}")
        return self.projects_root / safe_id

    def ensure_project_dirs(self, project_id: str) -> Path:
        pdir = self.project_dir(project_id)
        pdir.mkdir(parents=True, exist_ok=True)
        for sub in self.SUBDIRS:
            (pdir / sub).mkdir(parents=True, exist_ok=True)
        return pdir

    def resolve_project_file(self, project_id: str, rel_path: str) -> Path:
        pdir = self.project_dir(project_id).resolve()
        candidate = (pdir / rel_path).resolve()
        if pdir not in candidate.parents and candidate != pdir:
            raise PermissionError("Path traversal outside project directory is forbidden.")
        return candidate

    def create_project(self, name: str, bpm: float = 120.0, musical_key: str = "A minor", raw_prompt: str = "") -> ProjectSchema:
        project_id = f"proj_{uuid.uuid4().hex[:10]}"
        self.ensure_project_dirs(project_id)
        project = ProjectSchema(
            project_id=project_id,
            name=name.strip(),
        )
        project.grid.bpm = bpm
        project.grid.musical_key = musical_key
        project.producer_spec.raw_prompt = raw_prompt
        self.save_project(project)
        return project

    def load_project(self, project_id: str) -> ProjectSchema:
        manifest_path = self.project_dir(project_id) / "project.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        return ProjectSchema.model_validate(raw)

    def save_project(self, project: ProjectSchema) -> ProjectSchema:
        pdir = self.ensure_project_dirs(project.project_id)
        project.updated_at = utc_now_iso()
        manifest_path = pdir / "project.json"
        tmp_path = pdir / "project.json.tmp"
        payload = json.dumps(project.model_dump(mode="json"), indent=2)
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, manifest_path)
        return project

    def list_projects(self) -> list[ProjectSchema]:
        projects: list[ProjectSchema] = []
        if not self.projects_root.exists():
            return projects
        for child in self.projects_root.iterdir():
            manifest = child / "project.json"
            if child.is_dir() and manifest.is_file():
                try:
                    projects.append(self.load_project(child.name))
                except Exception:
                    continue
        projects.sort(key=lambda p: p.updated_at, reverse=True)
        return projects

    def write_wav(
        self,
        project_id: str,
        rel_path: str,
        audio: np.ndarray,
        sample_rate: int = settings.sample_rate,
    ) -> Path:
        """Write canonical float32 stereo audio normalized to 16-bit PCM WAV."""
        target = self.resolve_project_file(project_id, rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        stereo = self.ensure_stereo_float32(audio)
        if sample_rate != settings.sample_rate:
            stereo = self.resample_audio(stereo, sample_rate, settings.sample_rate)
            sample_rate = settings.sample_rate

        pcm16 = np.clip(np.round(stereo * 32768.0), -32768.0, 32767.0).astype(np.int16)
        tmp_path = target.with_suffix(".wav.tmp")

        if sf is not None:
            sf.write(str(tmp_path), pcm16, sample_rate, subtype="PCM_16", format="WAV")
        else:
            wavfile.write(str(tmp_path), sample_rate, pcm16)
        os.replace(tmp_path, target)
        return target

    def read_wav(
        self,
        project_id: str,
        rel_path: str,
        target_sr: int = settings.sample_rate,
    ) -> tuple[np.ndarray, int]:
        """Read a project WAV file as float32 (num_samples, 2) at `target_sr`."""
        src = self.resolve_project_file(project_id, rel_path)
        if not src.is_file():
            raise FileNotFoundError(f"Audio file '{rel_path}' not found in project '{project_id}'.")

        if sf is not None:
            data, sr = sf.read(str(src), dtype="float32", always_2d=True)
        else:
            sr, raw = wavfile.read(str(src))
            data = self._pcm_to_float32(raw)

        stereo = self.ensure_stereo_float32(data)
        if sr != target_sr:
            stereo = self.resample_audio(stereo, sr, target_sr)
            sr = target_sr
        return stereo, sr

    @staticmethod
    def _pcm_to_float32(raw: np.ndarray) -> np.ndarray:
        if raw.dtype == np.int16:
            arr = raw.astype(np.float32) / 32768.0
        elif raw.dtype == np.int32:
            arr = raw.astype(np.float32) / 2147483648.0
        elif raw.dtype == np.uint8:
            arr = (raw.astype(np.float32) - 128.0) / 128.0
        else:
            arr = raw.astype(np.float32)
        return arr

    @staticmethod
    def ensure_stereo_float32(audio: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.stack([arr, arr], axis=-1)
        elif arr.ndim == 2:
            # Handle (channels, samples) vs (samples, channels)
            if arr.shape[0] in (1, 2) and arr.shape[1] > arr.shape[0] and arr.shape[1] > 8:
                arr = arr.T
            if arr.shape[1] == 1:
                arr = np.repeat(arr, 2, axis=1)
            elif arr.shape[1] > 2:
                arr = arr[:, :2]
        else:
            raise ValueError(f"Unsupported audio array shape: {arr.shape}")
        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr or audio.shape[0] == 0:
            return audio
        divisor = math.gcd(orig_sr, target_sr)
        up = target_sr // divisor
        down = orig_sr // divisor
        resampled = resample_poly(audio, up=up, down=down, axis=0)
        return np.ascontiguousarray(resampled, dtype=np.float32)


storage = StorageManager()
