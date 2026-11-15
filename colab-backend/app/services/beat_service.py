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


def _correct_octave_tempo(
    tempo_val: float,
    beat_times: list[float],
    target_bpm: float,
    duration_sec: float,
) -> tuple[float, list[float]]:
    """Correct half-time or double-time tempo detection relative to target_bpm."""
    valid_beats = [float(t) for t in beat_times if 0.0 <= t < duration_sec]
    corrected_tempo = float(tempo_val)
    cur_diff = abs(corrected_tempo - target_bpm)

    double_tempo = corrected_tempo * 2.0
    half_tempo = corrected_tempo * 0.5

    if 45.0 <= double_tempo <= 220.0 and abs(double_tempo - target_bpm) < cur_diff * 0.6:
        corrected_tempo = double_tempo
        if len(valid_beats) >= 2:
            subdivided: list[float] = []
            for idx in range(len(valid_beats) - 1):
                t0 = valid_beats[idx]
                t1 = valid_beats[idx + 1]
                subdivided.append(t0)
                mid = (t0 + t1) * 0.5
                if mid < duration_sec:
                    subdivided.append(mid)
            subdivided.append(valid_beats[-1])
            half_step = (valid_beats[-1] - valid_beats[-2]) * 0.5
            trailing = valid_beats[-1] + half_step
            if half_step > 0.0 and trailing < duration_sec:
                subdivided.append(trailing)
            valid_beats = subdivided
    elif 45.0 <= half_tempo <= 220.0 and abs(half_tempo - target_bpm) < cur_diff * 0.6:
        corrected_tempo = half_tempo
        if len(valid_beats) >= 2:
            valid_beats = valid_beats[::2]

    return round(corrected_tempo, 2), [round(float(t), 6) for t in valid_beats if 0.0 <= t < duration_sec]


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
            raw_times = librosa.frames_to_time(beat_frames, sr=sample_rate).tolist()
            corrected_bpm, corrected_times = _correct_octave_tempo(
                tempo_val, raw_times, float(target_bpm), duration_sec
            )
            if 45.0 <= corrected_bpm <= 220.0:
                detected_bpm = corrected_bpm
            if len(corrected_times) >= 2:
                beat_times = corrected_times
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
