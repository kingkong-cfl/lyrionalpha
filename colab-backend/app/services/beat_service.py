"""CPU-based beat grid & sample-accurate bar/beat index extraction."""

from __future__ import annotations

import logging
import numpy as np

from app.config import settings
from app.schemas import BeatGridSchema

logger = logging.getLogger("lyrion.beat_service")

try:
    import librosa  # type: ignore
except ImportError:  # pragma: no cover - local CPU fallback
    librosa = None  # type: ignore


def analyze_beat_grid(
    audio: np.ndarray,
    sample_rate: int = settings.sample_rate,
    target_bpm: float = 120.0,
    musical_key: str = "A minor",
    time_signature: str = "4/4",
) -> BeatGridSchema:
    """
    Build sample-accurate beat_timestamps[] and beat_samples[].
    Uses librosa.beat.beat_track on CPU when available in non-mock GPU mode,
    otherwise computes exact sample-locked beats aligned to target_bpm.
    """
    total_samples = int(audio.shape[0])
    duration_sec = round(total_samples / float(sample_rate), 6) if sample_rate > 0 else 0.0
    beats_per_bar = int(time_signature.split("/")[0]) if "/" in time_signature else 4

    detected_bpm = float(target_bpm)
    beat_times: list[float] = []

    if not settings.mock_gpu and librosa is not None and total_samples > sample_rate:
        try:
            mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
            tempo, beat_frames = librosa.beat.beat_track(
                y=mono.astype(np.float32),
                sr=sample_rate,
                start_bpm=float(target_bpm),
                units="frames",
            )
            tempo_val = float(np.atleast_1d(tempo)[0])
            if 45.0 <= tempo_val <= 220.0:
                detected_bpm = round(tempo_val, 2)
            raw_times = librosa.frames_to_time(beat_frames, sr=sample_rate).tolist()
            if len(raw_times) >= 2:
                beat_times = [round(float(t), 6) for t in raw_times if 0.0 <= t < duration_sec]
        except Exception as exc:
            logger.warning("librosa beat_track fallback triggered: %s", exc)

    if len(beat_times) < 2:
        beat_period_sec = 60.0 / max(40.0, detected_bpm)
        n_beats = max(1, int(duration_sec / beat_period_sec) + 1)
        beat_times = []
        for idx in range(n_beats):
            t = round(idx * beat_period_sec, 6)
            if t < duration_sec:
                beat_times.append(t)

    beat_samples = [min(total_samples, max(0, int(round(t * sample_rate)))) for t in beat_times]
    # Deduplicate monotonic sample indices while keeping order
    dedup_times: list[float] = []
    dedup_samples: list[int] = []
    last_s = -1
    for t, s in zip(beat_times, beat_samples):
        if s > last_s:
            dedup_times.append(t)
            dedup_samples.append(s)
            last_s = s

    downbeat_indices = [i for i in range(0, len(dedup_samples), max(1, beats_per_bar))]

    return BeatGridSchema(
        sample_rate=sample_rate,
        total_samples=total_samples,
        duration_sec=duration_sec,
        bpm=round(detected_bpm, 2),
        time_signature=time_signature,
        musical_key=musical_key,
        beat_timestamps=dedup_times,
        beat_samples=dedup_samples,
        downbeat_beat_indices=downbeat_indices,
    )
