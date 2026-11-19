"""Server-side DSP & mastering export using Spotify Pedalboard + FFmpeg (with SciPy fallback)."""

from __future__ import annotations

import logging
import math
import re
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import lfilter, sosfilt

from app.config import settings
from app.core.storage import storage
from app.schemas import ProjectSchema, TrackDspChain, TrackSchema

logger = logging.getLogger("lyrion.dsp_service")

try:
    from pedalboard import (  # type: ignore
        Compressor,
        Gain,
        HighShelfFilter,
        Limiter,
        LowShelfFilter,
        PeakFilter,
        Pedalboard,
        Reverb,
    )

    HAS_PEDALBOARD = True
except ImportError:  # pragma: no cover - local CPU fallback
    HAS_PEDALBOARD = False


def _biquad_low_shelf_sos(cutoff_hz: float, gain_db: float, sr: int, q: float = 0.707) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * cutoff_hz / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    two_sqrt_a_alpha = 2.0 * math.sqrt(a) * alpha
    b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0)
    b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0)
    a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _biquad_high_shelf_sos(cutoff_hz: float, gain_db: float, sr: int, q: float = 0.707) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * cutoff_hz / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    two_sqrt_a_alpha = 2.0 * math.sqrt(a) * alpha
    b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)
    b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)
    a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _biquad_peak_sos(center_hz: float, gain_db: float, sr: int, q: float = 1.2) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * min(center_hz, sr * 0.45) / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _apply_dsp_scipy(audio: np.ndarray, dsp: TrackDspChain, sr: int) -> np.ndarray:
    out = audio.astype(np.float64, copy=True)
    if abs(dsp.eq.low_shelf_db) > 0.05:
        out = sosfilt(_biquad_low_shelf_sos(140.0, dsp.eq.low_shelf_db, sr), out, axis=0)
    if abs(dsp.eq.mid_gain_db) > 0.05:
        out = sosfilt(_biquad_peak_sos(dsp.eq.mid_freq_hz, dsp.eq.mid_gain_db, sr), out, axis=0)
    if abs(dsp.eq.high_shelf_db) > 0.05:
        out = sosfilt(_biquad_high_shelf_sos(4500.0, dsp.eq.high_shelf_db, sr), out, axis=0)

    if dsp.compressor.enabled:
        thresh = 10.0 ** (dsp.compressor.threshold_db / 20.0)
        ratio = max(1.0, float(dsp.compressor.ratio))
        peak_env = np.max(np.abs(out), axis=1)
        rel_coeff = math.exp(-1.0 / max(1.0, sr * max(0.005, float(dsp.compressor.release_ms) / 1000.0)))
        atk_coeff = math.exp(-1.0 / max(1.0, sr * max(0.002, float(dsp.compressor.attack_ms) / 1000.0)))
        smooth_env = lfilter([1.0 - rel_coeff], [1.0, -rel_coeff], peak_env)
        target_gain = np.ones_like(smooth_env)
        over = smooth_env > thresh
        if np.any(over):
            compressed_env = thresh + (smooth_env[over] - thresh) / ratio
            target_gain[over] = compressed_env / np.maximum(smooth_env[over], 1e-8)
        smooth_gain = lfilter([1.0 - atk_coeff], [1.0, -atk_coeff], target_gain, zi=[1.0 * atk_coeff])[0]
        out = out * smooth_gain[:, None]

    if dsp.reverb.enabled and dsp.reverb.wet_level > 0.0:
        wet = min(0.85, max(0.0, float(dsp.reverb.wet_level)))
        delay_1 = int(0.029 * sr * (0.5 + dsp.reverb.room_size))
        delay_2 = int(0.043 * sr * (0.5 + dsp.reverb.room_size))
        tail = np.zeros_like(out)
        if out.shape[0] > delay_1:
            tail[delay_1:] += 0.45 * out[:-delay_1]
        if out.shape[0] > delay_2:
            tail[delay_2:, 0] += 0.32 * out[:-delay_2, 1]
            tail[delay_2:, 1] += 0.32 * out[:-delay_2, 0]
        damp = 1.0 - 0.5 * dsp.reverb.damping
        out = (1.0 - wet * 0.4) * out + (wet * damp) * tail

    return np.ascontiguousarray(out, dtype=np.float32)


def process_track_dsp(audio: np.ndarray, track: TrackSchema, sr: int = settings.sample_rate) -> np.ndarray:
    """Apply non-destructive EQ, Compressor, Reverb, Volume (dB), and Pan to a stereo buffer."""
    dsp = track.dsp_chain
    if HAS_PEDALBOARD:
        board = Pedalboard()
        if abs(dsp.eq.low_shelf_db) > 0.05:
            board.append(LowShelfFilter(cutoff_frequency_hz=140.0, gain_db=dsp.eq.low_shelf_db))
        if abs(dsp.eq.mid_gain_db) > 0.05:
            board.append(PeakFilter(cutoff_frequency_hz=dsp.eq.mid_freq_hz, gain_db=dsp.eq.mid_gain_db, q=1.2))
        if abs(dsp.eq.high_shelf_db) > 0.05:
            board.append(HighShelfFilter(cutoff_frequency_hz=4500.0, gain_db=dsp.eq.high_shelf_db))
        if dsp.compressor.enabled:
            board.append(
                Compressor(
                    threshold_db=dsp.compressor.threshold_db,
                    ratio=max(1.0, dsp.compressor.ratio),
                    attack_ms=dsp.compressor.attack_ms,
                    release_ms=dsp.compressor.release_ms,
                )
            )
        if dsp.reverb.enabled and dsp.reverb.wet_level > 0.0:
            board.append(
                Reverb(
                    room_size=min(1.0, max(0.0, dsp.reverb.room_size)),
                    damping=min(1.0, max(0.0, dsp.reverb.damping)),
                    wet_level=min(1.0, max(0.0, dsp.reverb.wet_level)),
                    dry_level=1.0 - min(0.5, dsp.reverb.wet_level * 0.4),
                )
            )
        board.append(Gain(gain_db=track.volume_db))
        # Pedalboard expects (channels, samples)
        processed = board(audio.T.astype(np.float32), sr).T
    else:
        processed = _apply_dsp_scipy(audio, dsp, sr)
        linear_gain = 10.0 ** (track.volume_db / 20.0)
        processed = processed * linear_gain

    # Web Audio StereoPannerNode-compatible true-stereo pan law (-1.0 left .. +1.0 right)
    pan = max(-1.0, min(1.0, float(track.pan)))
    left = processed[:, 0].copy()
    right = processed[:, 1].copy()
    if pan <= 0.0:
        x = (pan + 1.0) * (math.pi / 2.0)
        processed[:, 0] = left + right * math.cos(x)
        processed[:, 1] = right * math.sin(x)
    else:
        x = pan * (math.pi / 2.0)
        processed[:, 0] = left * math.cos(x)
        processed[:, 1] = right + left * math.sin(x)
    return np.ascontiguousarray(processed, dtype=np.float32)


def _master_bus_limit(
    mix: np.ndarray,
    normalize_lufs: bool = True,
    sr: int = settings.sample_rate,
) -> np.ndarray:
    out = mix.astype(np.float32, copy=True)
    if normalize_lufs and out.size > 0:
        rms = float(np.sqrt(np.mean(out**2)))
        if rms > 1e-6:
            # Target approx -14 LUFS RMS equivalent (-14 dBFS ~ 0.20 RMS)
            target_rms = 0.20
            gain = min(4.0, target_rms / rms)
            out = out * gain

    if HAS_PEDALBOARD:
        limiter = Pedalboard([Limiter(threshold_db=-1.0, release_ms=60.0)])
        out = limiter(out.T, sr).T

    peak = float(np.max(np.abs(out))) if out.size else 1.0
    if peak > 0.98:
        out = out * (0.98 / peak)
    return np.ascontiguousarray(out, dtype=np.float32)


def _safe_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
    return slug or "track"


def _slice_region_buffer(
    buf: np.ndarray,
    start_sample: Optional[int],
    end_sample: Optional[int],
    total_samples: int,
    sr: int,
) -> np.ndarray:
    if start_sample is None or end_sample is None:
        return buf
    if start_sample < 0 or end_sample > total_samples or start_sample >= end_sample:
        raise ValueError(
            f"Invalid export region [{start_sample}, {end_sample}] for total_samples={total_samples}."
        )
    sliced = np.ascontiguousarray(buf[start_sample:end_sample], dtype=np.float32).copy()
    # Apply subtle 5ms edge de-click when cropping a sub-region
    fade_len = min(int(0.005 * sr), sliced.shape[0] // 4)
    if fade_len > 1:
        ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)[:, None]
        if start_sample > 0:
            sliced[:fade_len] *= ramp
        if end_sample < total_samples:
            sliced[-fade_len:] *= ramp[::-1]
    return sliced


def export_project_audio(
    project: ProjectSchema,
    export_format: str = "wav",
    include_muted: bool = False,
    normalize_lufs: bool = True,
    track_id: Optional[str] = None,
    start_sample: Optional[int] = None,
    end_sample: Optional[int] = None,
) -> dict[str, str]:
    """Render processed stems, single track, or master mix (full or region) into exports/."""
    sr = project.grid.sample_rate or settings.sample_rate
    total_samples = max(1, project.grid.total_samples)
    is_region = start_sample is not None and end_sample is not None
    region_tag = f"_s{start_sample}_{end_sample}" if is_region else ""

    if export_format == "single_track_wav":
        if not track_id:
            raise ValueError("track_id is required when export format is 'single_track_wav'.")
        target_track = next((t for t in project.tracks if t.track_id == track_id), None)
        if target_track is None or not target_track.file_rel_path:
            raise FileNotFoundError(f"Track '{track_id}' not found or has no audio file.")

        audio, _ = storage.read_wav(project.project_id, target_track.file_rel_path, target_sr=sr)
        if audio.shape[0] < total_samples:
            pad = np.zeros((total_samples - audio.shape[0], 2), dtype=np.float32)
            audio = np.concatenate([audio, pad], axis=0)
        elif audio.shape[0] > total_samples:
            audio = audio[:total_samples]

        proc = process_track_dsp(audio, target_track, sr)
        sliced = _slice_region_buffer(proc, start_sample, end_sample, total_samples, sr)
        # Preserve mixer fader volume_db on single-track export (only apply safety ceiling limiter)
        rendered = _master_bus_limit(sliced, normalize_lufs=False, sr=sr)
        track_slug = _safe_slug(target_track.name)
        out_rel = f"exports/{project.project_id}_{track_slug}{region_tag}.wav"
        storage.write_wav(project.project_id, out_rel, rendered, sample_rate=sr)
        return {"download_rel_path": out_rel, "master_rel_path": out_rel}

    has_solo = any(t.soloed for t in project.tracks)
    active_tracks: list[TrackSchema] = []
    for t in project.tracks:
        if not t.file_rel_path:
            continue
        if not include_muted and t.muted:
            continue
        if has_solo and not t.soloed:
            continue
        active_tracks.append(t)

    master_mix = np.zeros((total_samples, 2), dtype=np.float32)
    processed_stems: list[tuple[str, np.ndarray]] = []
    used_zip_names: set[str] = set()

    for track in active_tracks:
        audio, _ = storage.read_wav(project.project_id, track.file_rel_path or "", target_sr=sr)
        if audio.shape[0] < total_samples:
            pad = np.zeros((total_samples - audio.shape[0], 2), dtype=np.float32)
            audio = np.concatenate([audio, pad], axis=0)
        elif audio.shape[0] > total_samples:
            audio = audio[:total_samples]

        proc = process_track_dsp(audio, track, sr)
        base_name = _safe_slug(track.name) if track.stem_role == "custom" else (track.stem_role or _safe_slug(track.name))
        stem_name = base_name
        idx = 2
        while stem_name in used_zip_names:
            stem_name = f"{base_name}_{idx}"
            idx += 1
        used_zip_names.add(stem_name)

        sliced_stem = _slice_region_buffer(proc, start_sample, end_sample, total_samples, sr)
        processed_stems.append((stem_name, sliced_stem))
        # Avoid double-summing full_mix when stems are already active
        if track.stem_role != "full_mix" or len(active_tracks) == 1:
            master_mix += proc

    sliced_master = _slice_region_buffer(master_mix, start_sample, end_sample, total_samples, sr)
    rendered_master = _master_bus_limit(sliced_master, normalize_lufs=normalize_lufs, sr=sr)

    master_rel = f"exports/{project.project_id}_master{region_tag}.wav"
    storage.write_wav(project.project_id, master_rel, rendered_master, sample_rate=sr)

    if export_format == "stems_zip":
        zip_rel = f"exports/{project.project_id}_stems{region_tag}.zip"
        zip_path = storage.resolve_project_file(project.project_id, zip_rel)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            master_abs = storage.resolve_project_file(project.project_id, master_rel)
            zf.write(master_abs, arcname="master_mix.wav")
            for stem_name, buf in processed_stems:
                stem_rel = f"exports/_tmp_{stem_name}.wav"
                stem_abs = storage.write_wav(project.project_id, stem_rel, _master_bus_limit(buf, False, sr=sr), sample_rate=sr)
                zf.write(stem_abs, arcname=f"{stem_name}.wav")
                stem_abs.unlink(missing_ok=True)
        return {"download_rel_path": zip_rel, "master_rel_path": master_rel}

    return {"download_rel_path": master_rel, "master_rel_path": master_rel}
