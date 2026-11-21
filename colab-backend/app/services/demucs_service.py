"""Demucs (htdemucs) 4-stem separation service managed by ModelManager."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.config import settings
from app.core.model_manager import ModelType, model_manager
from app.core.storage import StorageManager

logger = logging.getLogger("lyrion.demucs")

STEM_ORDER = ("drums", "bass", "other", "vocals")


def _load_demucs_model() -> Any:
    if settings.mock_gpu:
        return {"backend": "mock_htdemucs"}

    import torch  # type: ignore
    from demucs.pretrained import get_model  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    preferred_model = os.environ.get("DEMUCS_MODEL", "htdemucs_ft").strip() or "htdemucs_ft"
    loaded_name = preferred_model
    try:
        model = get_model(preferred_model)
    except Exception as exc:
        if preferred_model == "htdemucs":
            raise
        logger.warning("Failed to load Demucs '%s' (%s); falling back to 'htdemucs'.", preferred_model, exc)
        loaded_name = "htdemucs"
        model = get_model("htdemucs")
    model.to(device)
    model.eval()
    try:
        setattr(model, "_lyrion_display_name", loaded_name)
    except Exception:
        pass
    return model


def _mock_separate_stems(audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
    """
    Complementary frequency + mid/side + transient splitter for MOCK_GPU=true.
    Guarantees vocals + drums + bass + other == audio accurately at exact sample count.
    """
    n_samples = audio.shape[0]
    if n_samples < 64:
        zero = np.zeros_like(audio, dtype=np.float32)
        return {"vocals": zero.copy(), "drums": zero.copy(), "bass": zero.copy(), "other": audio.copy()}

    nyq = 0.5 * sample_rate
    # 1. Bass (< 170 Hz)
    sos_low = butter(4, min(0.95, 170.0 / nyq), btype="lowpass", output="sos")
    bass = sosfiltfilt(sos_low, audio, axis=0).astype(np.float32)

    residual = audio - bass

    # 2. Upper air / percussion shimmer (> 3400 Hz)
    sos_high = butter(4, min(0.95, 3400.0 / nyq), btype="highpass", output="sos")
    highs = sosfiltfilt(sos_high, residual, axis=0).astype(np.float32)

    midband = residual - highs

    # 3. Mid-center vs Mid-side split (Vocals sit centered in midband; pads/keys sit wide)
    mid_mono = 0.5 * (midband[:, 0:1] + midband[:, 1:2])
    mid_center = np.repeat(mid_mono, 2, axis=1).astype(np.float32)
    mid_side = (midband - mid_center).astype(np.float32)

    # Extract fast percussive bursts from mid_center into drums, leaving tonal center as vocals
    energy = np.abs(mid_center)
    smooth = np.convolve(energy[:, 0], np.ones(1024, dtype=np.float32) / 1024.0, mode="same")[:, None]
    transient_mask = np.clip((energy - smooth * 1.45) * 3.0, 0.0, 1.0)
    drums_mid = (mid_center * transient_mask).astype(np.float32)
    vocals = (mid_center - drums_mid).astype(np.float32)

    drums = (drums_mid + 0.55 * highs).astype(np.float32)
    other = (mid_side + 0.45 * highs).astype(np.float32)

    return {
        "vocals": np.ascontiguousarray(vocals, dtype=np.float32),
        "drums": np.ascontiguousarray(drums, dtype=np.float32),
        "bass": np.ascontiguousarray(bass, dtype=np.float32),
        "other": np.ascontiguousarray(other, dtype=np.float32),
    }


async def separate_stems(audio: np.ndarray, sample_rate: int = settings.sample_rate) -> dict[str, np.ndarray]:
    """
    Separate stereo float32 audio (N, 2) into 4 stems (`vocals`, `drums`, `bass`, `other`).
    Enforces strict ModelManager locking so ACE-Step 1.5 is unloaded first.
    """
    stereo = StorageManager.ensure_stereo_float32(audio)
    n_samples = stereo.shape[0]

    async with model_manager.acquire(ModelType.DEMUCS, _load_demucs_model) as model:
        if settings.mock_gpu:
            await asyncio.sleep(0.12)
            return await asyncio.to_thread(_mock_separate_stems, stereo, sample_rate)

        def _run_real_demucs() -> dict[str, np.ndarray]:
            import torch  # type: ignore
            from demucs.apply import apply_model  # type: ignore

            device = next(model.parameters()).device
            # Demucs native sample rate is usually 44100 Hz
            model_sr = int(getattr(model, "samplerate", 44100))
            work = stereo
            if sample_rate != model_sr:
                work = StorageManager.resample_audio(stereo, sample_rate, model_sr)

            wav_tensor = torch.from_numpy(work.T).unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                sources = apply_model(model, wav_tensor, shifts=1, split=True, overlap=0.25, progress=False)[0]

            source_names = list(getattr(model, "sources", STEM_ORDER))
            stems_out: dict[str, np.ndarray] = {}
            for idx, name in enumerate(source_names):
                arr = sources[idx].detach().cpu().numpy().T
                if model_sr != sample_rate:
                    arr = StorageManager.resample_audio(arr, model_sr, sample_rate)
                if arr.shape[0] > n_samples:
                    arr = arr[:n_samples]
                elif arr.shape[0] < n_samples:
                    pad = np.zeros((n_samples - arr.shape[0], 2), dtype=np.float32)
                    arr = np.concatenate([arr, pad], axis=0)
                stems_out[name] = np.ascontiguousarray(arr, dtype=np.float32)

            return {k: stems_out[k] for k in ("vocals", "drums", "bass", "other") if k in stems_out}

        return await asyncio.to_thread(_run_real_demucs)
