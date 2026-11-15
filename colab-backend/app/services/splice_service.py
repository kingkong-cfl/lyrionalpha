"""Sample-accurate equal-power crossfade splicing and windowed Demucs re-sync."""

from __future__ import annotations

import numpy as np

from app.config import settings


def equal_power_crossfade_splice(
    original_audio: np.ndarray,
    new_segment: np.ndarray,
    start_sample: int,
    end_sample: int,
    crossfade_samples: int = settings.crossfade_samples,
    mode: str = "equal_power",
) -> np.ndarray:
    """
    Splice `new_segment` (length == end_sample - start_sample) into `original_audio`
    using cosine/sine (`mode="equal_power"`) or linear (`mode="equal_gain"`, ideal
    for correlated Demucs re-stem boundaries) crossfade windows at internal boundaries.
    Samples outside [start_sample, end_sample) remain 100% bit-identical.
    """
    total_samples = int(original_audio.shape[0])
    if start_sample < 0 or end_sample > total_samples or start_sample >= end_sample:
        raise ValueError(
            f"Invalid sample slice [{start_sample}, {end_sample}] for buffer length {total_samples}."
        )

    region_len = end_sample - start_sample
    if new_segment.shape[0] != region_len:
        raise ValueError(
            f"new_segment length ({new_segment.shape[0]}) must match slice length ({region_len})."
        )

    out = original_audio.copy()
    # Replace core region
    out[start_sample:end_sample] = new_segment

    # Compute safe crossfade length (<= 40% of region so left and right fades never overlap)
    fade_len = min(int(crossfade_samples), max(0, (region_len * 2) // 5))
    if fade_len <= 1:
        return np.ascontiguousarray(out, dtype=np.float32)

    use_equal_gain = mode == "equal_gain"

    # Left boundary crossfade (unless anchored at sample 0)
    if start_sample > 0:
        t = np.linspace(0.0, 1.0, fade_len, endpoint=True, dtype=np.float32)[:, None]
        if use_equal_gain:
            g_old = 1.0 - t
            g_new = t
        else:
            g_old = np.cos(0.5 * np.pi * t)
            g_new = np.sin(0.5 * np.pi * t)
        s0 = start_sample
        s1 = start_sample + fade_len
        out[s0:s1] = original_audio[s0:s1] * g_old + new_segment[:fade_len] * g_new

    # Right boundary crossfade (unless anchored at end of track)
    if end_sample < total_samples:
        t = np.linspace(0.0, 1.0, fade_len, endpoint=True, dtype=np.float32)[:, None]
        if use_equal_gain:
            g_new = 1.0 - t
            g_old = t
        else:
            g_new = np.cos(0.5 * np.pi * t)
            g_old = np.sin(0.5 * np.pi * t)
        e0 = end_sample - fade_len
        e1 = end_sample
        out[e0:e1] = new_segment[-fade_len:] * g_new + original_audio[e0:e1] * g_old

    return np.ascontiguousarray(out, dtype=np.float32)


def apply_region_fade_edges(
    segment: np.ndarray,
    fade_samples: int = settings.crossfade_samples,
) -> np.ndarray:
    """
    Apply cosine fade-in and fade-out edges (`fade_samples`, default 50 ms)
    to a standalone instrument segment placed over silence so edges never click.
    """
    out = np.ascontiguousarray(segment, dtype=np.float32).copy()
    n_samples = int(out.shape[0])
    fade_len = min(int(fade_samples), n_samples // 2)
    if fade_len <= 1:
        return out

    t = np.linspace(0.0, 1.0, fade_len, endpoint=True, dtype=np.float32)[:, None]
    fade_in = np.sin(0.5 * np.pi * t)
    fade_out = np.cos(0.5 * np.pi * t)
    out[:fade_len] *= fade_in
    out[-fade_len:] *= fade_out
    return np.ascontiguousarray(out, dtype=np.float32)


def compute_context_window(
    start_sample: int,
    end_sample: int,
    total_samples: int,
    context_samples: int = settings.demucs_context_samples,
) -> tuple[int, int, int, int]:
    """
    Return (win_start, win_end, inner_start, inner_end) so Demucs runs on
    [win_start:win_end] and the clean inner region maps to [inner_start:inner_end].
    """
    win_start = max(0, start_sample - context_samples)
    win_end = min(total_samples, end_sample + context_samples)
    inner_start = start_sample - win_start
    inner_end = inner_start + (end_sample - start_sample)
    return win_start, win_end, inner_start, inner_end
