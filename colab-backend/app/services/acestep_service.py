"""ACE-Step 1.5 service (bfloat16 on Colab L4 GPU, musical synthesis in MOCK_GPU mode)."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

from app.config import settings
from app.core.model_manager import ModelType, model_manager
from app.core.storage import StorageManager, storage

logger = logging.getLogger("lyrion.acestep")


_KEY_ROOT_HZ = {
    "C": 261.63,
    "C#": 277.18,
    "Db": 277.18,
    "D": 293.66,
    "D#": 311.13,
    "Eb": 311.13,
    "E": 329.63,
    "F": 349.23,
    "F#": 369.99,
    "Gb": 369.99,
    "G": 392.00,
    "G#": 415.30,
    "Ab": 415.30,
    "A": 220.00,
    "A#": 233.08,
    "Bb": 233.08,
    "B": 246.94,
}


def _parse_key_scale(musical_key: str) -> list[float]:
    parts = musical_key.strip().split()
    root_name = parts[0] if parts else "A"
    is_minor = any("min" in p.lower() for p in parts[1:]) or "m" == root_name[-1:]
    root_clean = root_name.rstrip("m") if len(root_name) > 1 and root_name.endswith("m") else root_name
    base_hz = _KEY_ROOT_HZ.get(root_clean, 220.0)
    semitones = [0, 2, 3, 5, 7, 8, 10, 12] if is_minor else [0, 2, 4, 5, 7, 9, 11, 12]
    return [base_hz * (2.0 ** (st / 12.0)) for st in semitones]


def _synthesize_musical_mix(
    total_samples: int,
    sample_rate: int,
    bpm: float,
    musical_key: str,
    seed: int,
    style_tags: list[str],
    layer_mode: str = "full",
    instrument_hint: str = "",
) -> np.ndarray:
    """Generate a rich, phase-coherent stereo arrangement for local MOCK_GPU=true development."""
    rng = np.random.default_rng(abs(int(seed)) % (2**31 - 1))
    t = np.arange(total_samples, dtype=np.float32) / float(sample_rate)
    beat_sec = 60.0 / max(45.0, float(bpm))
    beat_pos = t / beat_sec
    beat_idx = np.floor(beat_pos).astype(np.int32)
    bar_idx = beat_idx // 4
    sixteenth_idx = np.floor(beat_pos * 4.0).astype(np.int32)
    sixteenth_frac = (beat_pos * 4.0) % 1.0
    step16_mod = sixteenth_idx % 16
    step8_mod = (np.floor(beat_pos * 2.0).astype(np.int32)) % 8

    scale = _parse_key_scale(musical_key)
    progression_degrees = rng.choice([0, 2, 3, 4, 5], size=4, replace=True)

    # 1. Bassline (sub-150Hz fundamental + seed-varied groove & harmonic warmth)
    bass_mono = np.zeros(total_samples, dtype=np.float32)
    if layer_mode in ("full", "bass"):
        bass_subdiv = float(rng.choice([2.0, 4.0]))
        bass_oct = np.where(rng.random(8) < 0.30, 0.5, 0.25).astype(np.float32)
        bass_gate = np.where(rng.random(8) < 0.82, 1.0, 0.25).astype(np.float32)
        bass_gate[0] = 1.0
        bass_decay = float(rng.uniform(2.2, 4.8))
        bass_h2 = float(rng.uniform(0.05, 0.14))

        root_seq = (
            np.array([scale[progression_degrees[b % 4]] for b in bar_idx], dtype=np.float32)
            * bass_oct[step8_mod]
        )
        bass_phase_pos = (beat_pos * bass_subdiv) % 1.0
        bass_env = np.exp(-bass_decay * bass_phase_pos).astype(np.float32) * bass_gate[step8_mod]
        phase_bass = 2.0 * np.pi * np.cumsum(root_seq / sample_rate)
        bass_mono = (0.30 * np.sin(phase_bass) + bass_h2 * np.sin(2.0 * phase_bass)) * bass_env

    # 2. Drums (seed-varied syncopated kick/snare groove & 16th hi-hat accents)
    drums_l = np.zeros(total_samples, dtype=np.float32)
    drums_r = np.zeros(total_samples, dtype=np.float32)
    if layer_mode in ("full", "drums"):
        kick_mask = np.zeros(16, dtype=np.float32)
        kick_mask[0] = 1.0
        kick_mask[8] = 1.0
        for sync_step in (2, 3, 6, 10, 14):
            if rng.random() < 0.50:
                kick_mask[sync_step] = float(rng.uniform(0.75, 1.0))
        kick_base_hz = float(rng.uniform(44.0, 60.0))
        kick_punch_hz = float(rng.uniform(75.0, 120.0))
        kick_decay = float(rng.uniform(5.5, 9.5))

        kick_freq = kick_base_hz + kick_punch_hz * np.exp(-16.0 * sixteenth_frac)
        kick_env = np.exp(-kick_decay * sixteenth_frac) * kick_mask[step16_mod]
        kick = 0.42 * np.sin(2.0 * np.pi * np.cumsum(kick_freq / sample_rate)) * kick_env

        snare_mask = np.zeros(16, dtype=np.float32)
        snare_mask[4] = 1.0
        snare_mask[12] = 1.0
        for ghost_step in (2, 7, 10, 15):
            if rng.random() < 0.45:
                snare_mask[ghost_step] = float(rng.uniform(0.30, 0.55))
        snare_body_hz = float(rng.uniform(175.0, 240.0))
        snare_decay = float(rng.uniform(9.0, 16.0))

        noise = rng.standard_normal(total_samples).astype(np.float32)
        snare_amp = snare_mask[step16_mod]
        snare_env = np.exp(-snare_decay * sixteenth_frac) * snare_amp
        snare_body = np.sin(2.0 * np.pi * snare_body_hz * t) * np.exp(-14.0 * sixteenth_frac) * snare_amp
        snare = (0.16 * noise * snare_env) + (0.14 * snare_body)

        hat_accents = rng.uniform(0.35, 1.30, size=16).astype(np.float32)
        hat_mask = rng.choice([0.0, 1.0], size=16, p=[0.18, 0.82]).astype(np.float32)
        hat_decay = float(rng.uniform(18.0, 42.0))
        hat_noise = np.diff(noise, prepend=0.0)
        hat_env = np.exp(-hat_decay * sixteenth_frac) * hat_accents[step16_mod] * hat_mask[step16_mod]
        hats = 0.06 * hat_noise * hat_env

        drums_l = kick + snare + hats * 1.1
        drums_r = kick + snare + hats * 0.9

    # 3. Stereo Pad / Synth Harmony ("other" stem, >3.8kHz & wide stereo)
    other_l = np.zeros(total_samples, dtype=np.float32)
    other_r = np.zeros(total_samples, dtype=np.float32)
    if layer_mode in ("full", "other", "custom"):
        chord_mod = float(rng.choice([0.25, 1.0, 2.0, 4.0]))
        voicing_third = float(rng.choice([1.189, 1.260, 1.335]))
        voicing_fifth = float(rng.choice([1.498, 1.682, 1.888]))
        lfo_rate = float(rng.uniform(0.25, 1.15))
        bright_hz = float(rng.uniform(3600.0, 5400.0))
        detune_phases = rng.uniform(0.15, 1.1, size=3)

        chord_freqs = np.array([scale[progression_degrees[b % 4]] for b in bar_idx], dtype=np.float32)
        third_freqs = chord_freqs * voicing_third
        fifth_freqs = chord_freqs * voicing_fifth
        pad_frac = (beat_pos * chord_mod) % 1.0
        pad_env = np.minimum(1.0, pad_frac * 8.0) * np.exp(-1.1 * pad_frac)
        p1 = 2.0 * np.pi * np.cumsum(chord_freqs / sample_rate)
        p2 = 2.0 * np.pi * np.cumsum(third_freqs / sample_rate)
        p3 = 2.0 * np.pi * np.cumsum(fifth_freqs / sample_rate)
        lfo = 0.5 + 0.5 * np.sin(2.0 * np.pi * lfo_rate * t)
        bright = 0.05 * np.sin(2.0 * np.pi * bright_hz * t) * (0.4 + 0.6 * lfo)
        other_l = 0.10 * (
            np.sin(p1) + 0.7 * np.sin(p2 + float(detune_phases[0])) + 0.6 * np.sin(p3)
        ) * pad_env + bright
        other_r = 0.10 * (
            np.sin(p1 + float(detune_phases[1])) + 0.7 * np.sin(p2) + 0.6 * np.sin(p3 + float(detune_phases[2]))
        ) * pad_env - bright * 0.8

    # 4. Vocal-like melodic lead (centered mid-band 450Hz-2400Hz with seed-varied vibrato/melody)
    vocal_mono = np.zeros(total_samples, dtype=np.float32)
    if layer_mode in ("full", "vocals"):
        vocal_div = float(rng.choice([2.0, 3.0, 4.0]))
        step_idx = np.floor(beat_pos * vocal_div).astype(np.int32)
        melody_pattern = rng.integers(0, len(scale), size=16)
        vocal_gate = rng.choice([0.0, 1.0], size=16, p=[0.22, 0.78]).astype(np.float32)
        vocal_gate[0] = 1.0
        vocal_gate[8] = 1.0
        vibrato_rate = float(rng.uniform(4.6, 6.6))
        vibrato_depth = float(rng.uniform(0.005, 0.013))
        h2_amp = float(rng.uniform(0.05, 0.12))
        h3_amp = float(rng.uniform(0.02, 0.06))

        step_frac = (beat_pos * vocal_div) % 1.0
        step_mod = step_idx % len(melody_pattern)
        note_freqs = np.array([scale[melody_pattern[s]] * 2.0 for s in step_mod], dtype=np.float32)
        vibrato = 1.0 + vibrato_depth * np.sin(2.0 * np.pi * vibrato_rate * t)
        vocal_phase = 2.0 * np.pi * np.cumsum((note_freqs * vibrato) / sample_rate)
        phrase_gate = ((bar_idx % 4) < 3).astype(np.float32) * vocal_gate[step_mod]
        note_env = np.minimum(1.0, step_frac * 14.0) * np.exp(-2.2 * step_frac) * phrase_gate
        vocal_mono = (
            0.16 * np.sin(vocal_phase)
            + h2_amp * np.sin(2.0 * vocal_phase)
            + h3_amp * np.sin(3.0 * vocal_phase)
        ) * note_env

    # 5. Custom standalone instrument layer (electric guitar / lead synth / brass / keys)
    inst_l = np.zeros(total_samples, dtype=np.float32)
    inst_r = np.zeros(total_samples, dtype=np.float32)
    if layer_mode == "custom":
        hint = (instrument_hint + " " + " ".join(style_tags)).lower()
        riff_div = float(rng.choice([2.0, 4.0]))
        step_idx = np.floor(beat_pos * riff_div).astype(np.int32)
        riff_pattern = rng.integers(0, len(scale), size=16)
        riff_gate = rng.choice([0.0, 1.0], size=16, p=[0.20, 0.80]).astype(np.float32)
        riff_gate[0] = 1.0
        riff_gate[4] = 1.0
        drive = float(rng.uniform(1.8, 3.2))
        detune_rad = float(rng.uniform(0.12, 0.45))

        step_frac = (beat_pos * riff_div) % 1.0
        step_mod = step_idx % len(riff_pattern)
        octave = 2.0 if any(k in hint for k in ("lead", "synth", "flute", "violin")) else 1.0
        note_freqs = np.array([scale[riff_pattern[s]] * octave for s in step_mod], dtype=np.float32)
        phase = 2.0 * np.pi * np.cumsum(note_freqs / sample_rate)
        gate_mult = riff_gate[step_mod]

        if "guitar" in hint:
            pluck_env = np.minimum(1.0, step_frac * 24.0) * np.exp(-4.5 * step_frac) * gate_mult
            raw_wave = np.tanh(drive * (np.sin(phase) + 0.5 * np.sin(2.0 * phase) + 0.3 * np.sin(3.0 * phase)))
            inst_l = 0.22 * raw_wave * pluck_env
            inst_r = 0.22 * np.tanh(drive * np.sin(phase + detune_rad)) * pluck_env
        elif any(k in hint for k in ("brass", "horn", "sax")):
            swell_env = np.minimum(1.0, step_frac * 6.0) * np.exp(-1.5 * step_frac) * gate_mult
            raw_wave = 0.6 * np.sin(phase) + 0.4 * np.sin(2.0 * phase) + 0.25 * np.sin(3.0 * phase)
            inst_l = 0.22 * raw_wave * swell_env
            inst_r = 0.20 * np.sin(phase + detune_rad) * swell_env + 0.08 * raw_wave * swell_env
        else:
            arp_env = np.minimum(1.0, step_frac * 18.0) * np.exp(-3.0 * step_frac) * gate_mult
            raw_wave = 0.65 * np.sin(phase) + 0.35 * np.sin(2.0 * phase + detune_rad) + 0.15 * np.sin(4.0 * phase)
            inst_l = 0.22 * raw_wave * arp_env
            inst_r = 0.22 * np.sin(phase + detune_rad * 1.5) * arp_env

    mix_l = bass_mono + drums_l + other_l + vocal_mono + inst_l
    mix_r = bass_mono + drums_r + other_r + vocal_mono + inst_r
    stereo = np.stack([mix_l, mix_r], axis=-1).astype(np.float32)

    # Soft limiter headroom
    peak = float(np.max(np.abs(stereo))) if stereo.size else 1.0
    if peak > 0.92:
        stereo = stereo * (0.92 / peak)

    # Gentle 10ms fade-in/out on full song endpoints
    fade = min(441, total_samples // 10)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
        stereo[:fade] *= ramp
        stereo[-fade:] *= ramp[::-1]

    return np.ascontiguousarray(stereo, dtype=np.float32)


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter candidate kwargs to those accepted by `callable_obj` unless it accepts **kwargs."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _read_and_fit_stereo(
    audio_path: Path,
    target_sr: int,
    total_samples: int | None,
    max_duration_sec: float = 240.0,
) -> np.ndarray:
    """Read an audio file as float32 stereo at `target_sr`, fitting or preserving natural length."""
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except ImportError:
        sr, raw = wavfile.read(str(audio_path))
        data = StorageManager._pcm_to_float32(raw)

    stereo = StorageManager.ensure_stereo_float32(data)
    if sr != target_sr:
        stereo = StorageManager.resample_audio(stereo, sr, target_sr)
    if total_samples is None:
        max_samples = max(1, int(round(max_duration_sec * target_sr)))
        if stereo.shape[0] > max_samples:
            stereo = stereo[:max_samples]
    elif stereo.shape[0] > total_samples:
        stereo = stereo[:total_samples]
    elif stereo.shape[0] < total_samples:
        pad = np.zeros((total_samples - stereo.shape[0], 2), dtype=np.float32)
        stereo = np.concatenate([stereo, pad], axis=0)
    return np.ascontiguousarray(stereo, dtype=np.float32)


def _build_acestep_prompt(
    style_tags: list[str],
    bpm: float | None,
    musical_key: str | None,
    extra_prefix: list[str] | None = None,
) -> str:
    """Build caption string omitting optional BPM/Key tokens when left in Auto mode."""
    parts: list[str] = []
    if extra_prefix:
        parts.extend(p.strip() for p in extra_prefix if p and p.strip())
    parts.extend(t.strip() for t in style_tags if t and t.strip())
    if bpm is not None:
        parts.append(f"{bpm:g} bpm")
    clean_key = (musical_key or "").strip()
    if clean_key:
        parts.append(clean_key)
    return ", ".join(parts)


_TURKISH_DIACRITICS = set("çğıİöüşÇĞIÖÜŞ")
_TURKISH_INDICATOR_WORDS = frozenset(
    {
        "ve",
        "bir",
        "ben",
        "sen",
        "gibi",
        "icin",
        "ama",
        "cok",
        "yok",
        "var",
        "gece",
        "kalp",
        "ask",
        "hayat",
        "zaman",
        "beni",
        "seni",
        "biz",
        "kadar",
        "neden",
        "nasil",
        "bu",
        "o",
        "ile",
        "gibi",
        "belki",
        "simdi",
    }
)


def resolve_vocal_language(vocal_language: str | None, lyrics: str) -> tuple[str, bool]:
    """Return (vocal_language, use_cot_language) with automatic Turkish detection when Auto."""
    explicit = (vocal_language or "").strip().lower()
    if explicit in ("instrumental", "none"):
        return "unknown", False
    if explicit and explicit not in ("auto", "unknown"):
        return explicit, False

    raw_lyrics = lyrics or ""
    if any(ch in _TURKISH_DIACRITICS for ch in raw_lyrics):
        return "tr", False

    words = set(re.findall(r"[a-z]+", raw_lyrics.lower()))
    if len(words & _TURKISH_INDICATOR_WORDS) >= 2:
        return "tr", False

    return "unknown", True


def resolve_model_inference_profile(
    config_path: str,
    inference_steps: int,
    guidance_scale: float,
    shift: float,
) -> tuple[int, float, float]:
    """Upgrade turbo-default args automatically when SFT or Base checkpoints are active."""
    cfg = (config_path or "").lower()
    if ("sft" in cfg or "base" in cfg) and inference_steps == 8 and abs(guidance_scale - 1.0) < 1e-6:
        sft_steps = int(os.environ.get("ACESTEP_SFT_STEPS", "32"))
        sft_guidance = float(os.environ.get("ACESTEP_SFT_GUIDANCE_SCALE", "7.0"))
        return sft_steps, sft_guidance, float(shift)
    return int(inference_steps), float(guidance_scale), float(shift)


def _resolve_acestep_project_root() -> str:
    explicit = os.environ.get("ACESTEP_PROJECT_ROOT", "").strip()
    if explicit:
        return explicit
    colab_dir = Path("/content/ACE-Step-1.5")
    if colab_dir.is_dir():
        return str(colab_dir)
    return os.getcwd()


def _check_acestep_init_result(init_res: Any) -> None:
    if isinstance(init_res, tuple) and len(init_res) >= 2 and init_res[1] is False:
        raise RuntimeError(f"ACE-Step 1.5 initialize_service failed: {init_res[0]}")


def _load_acestep_model() -> Any:
    if settings.mock_gpu:
        return {"backend": "mock_acestep_1_5", "dtype": "bfloat16"}

    import torch  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from acestep.handler import AceStepHandler  # type: ignore
        from acestep.inference import (  # type: ignore
            GenerationConfig,
            GenerationParams,
            generate_music,
        )

        project_root = _resolve_acestep_project_root()
        config_path = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo").strip() or "acestep-v15-turbo"
        offload_to_cpu = os.environ.get("ACESTEP_OFFLOAD_TO_CPU", "false").lower() in ("1", "true", "yes")
        dit_handler = AceStepHandler()
        init_kwargs = _filter_kwargs(
            dit_handler.initialize_service,
            {
                "project_root": project_root,
                "config_path": config_path,
                "device": device,
                "offload_to_cpu": offload_to_cpu,
            },
        )
        init_res = dit_handler.initialize_service(**init_kwargs)
        _check_acestep_init_result(init_res)

        llm_handler = None
        lm_model_path = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B").strip()
        try:
            from acestep.llm_inference import LLMHandler  # type: ignore

            lm_backend = os.environ.get("ACESTEP_LM_BACKEND", "pt").strip() or "pt"
            checkpoint_dir = str(Path(project_root) / "checkpoints")
            candidate_llm = LLMHandler()
            llm_kwargs = _filter_kwargs(
                candidate_llm.initialize,
                {
                    "checkpoint_dir": checkpoint_dir,
                    "lm_model_path": lm_model_path,
                    "backend": lm_backend,
                    "device": device,
                    "offload_to_cpu": offload_to_cpu,
                },
            )
            llm_res = candidate_llm.initialize(**llm_kwargs)
            _check_acestep_init_result(llm_res)
            llm_handler = candidate_llm
        except Exception as exc:
            logger.warning("ACE-Step 1.5 LLMHandler unavailable, continuing with DiT-only mode: %s", exc)
            llm_handler = None

        lm_short = lm_model_path.removeprefix("acestep-5Hz-") or lm_model_path
        display_name = f"{config_path} + {lm_short}" if llm_handler is not None else f"{config_path} + lm=OFF"
        logger.info(
            "Loaded ACE-Step 1.5 (%s, device=%s, offload_to_cpu=%s)",
            display_name,
            device,
            offload_to_cpu,
        )

        return {
            "version": "1.5",
            "dit_handler": dit_handler,
            "llm_handler": llm_handler,
            "config_path": config_path,
            "lm_model_path": lm_model_path,
            "offload_to_cpu": offload_to_cpu,
            "display_name": display_name,
            "generate_music": generate_music,
            "GenerationParams": GenerationParams,
            "GenerationConfig": GenerationConfig,
        }
    except ImportError:
        from acestep.pipeline_ace_step import ACEStepPipeline  # type: ignore

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        pipeline = ACEStepPipeline(dtype=dtype)
        pipeline.load_checkpoint()
        return {"version": "1.0", "pipeline": pipeline}


def _invoke_acestep_real_sync(
    pipe_bundle: Any,
    *,
    task_type: str,
    prompt: str,
    lyrics: str,
    bpm: float | None,
    musical_key: str | None,
    duration_sec: float | None,
    seed: int | None,
    inference_steps: int,
    guidance_scale: float,
    target_sr: int,
    total_samples: int | None,
    shift: float = 3.0,
    vocal_language: str | None = None,
    src_audio: np.ndarray | None = None,
    repaint_start_sec: float = 0.0,
    repaint_end_sec: float = 0.0,
) -> np.ndarray:
    """Synchronous bridge supporting ACE-Step 1.5 (`AceStepHandler` + `generate_music`) with v1 fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path: Path | None = None
        if src_audio is not None:
            src_path = Path(tmpdir) / "source.wav"
            float_src = np.ascontiguousarray(StorageManager.ensure_stereo_float32(src_audio), dtype=np.float32)
            try:
                import soundfile as sf  # type: ignore

                sf.write(str(src_path), float_src, target_sr, subtype="FLOAT", format="WAV")
            except ImportError:
                wavfile.write(str(src_path), target_sr, float_src)

        if isinstance(pipe_bundle, dict) and pipe_bundle.get("version") == "1.5":
            params_cls = pipe_bundle["GenerationParams"]
            config_cls = pipe_bundle["GenerationConfig"]
            gen_fn = pipe_bundle["generate_music"]
            dit_handler = pipe_bundle["dit_handler"]
            llm_handler = pipe_bundle.get("llm_handler")
            config_path = str(
                pipe_bundle.get("config_path") or os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
            )
            display_name = str(pipe_bundle.get("display_name") or config_path)
            use_thinking = bool(llm_handler is not None and task_type == "text2music")

            eff_steps, eff_guidance, eff_shift = resolve_model_inference_profile(
                config_path, inference_steps, guidance_scale, shift
            )
            resolved_lang, use_cot_lang = resolve_vocal_language(vocal_language, lyrics)
            lm_cfg_scale = float(os.environ.get("ACESTEP_LM_CFG_SCALE", "2.0"))

            clean_key = (musical_key or "").strip()
            target_dur = float(duration_sec) if duration_sec is not None else -1.0
            param_kwargs: dict[str, Any] = {
                "task_type": task_type,
                "caption": prompt,
                "prompt": prompt,
                "lyrics": lyrics,
                "bpm": int(round(bpm)) if bpm is not None else None,
                "keyscale": clean_key,
                "duration": target_dur,
                "vocal_language": resolved_lang,
                "inference_steps": eff_steps,
                "guidance_scale": eff_guidance,
                "shift": eff_shift,
                "thinking": use_thinking,
                "use_cot_metas": use_thinking,
                "use_cot_caption": use_thinking,
                "use_cot_language": bool(use_thinking and use_cot_lang),
                "use_constrained_decoding": use_thinking,
                "lm_cfg_scale": lm_cfg_scale,
            }
            if seed is not None:
                param_kwargs["seed"] = int(seed)
            if src_path is not None:
                param_kwargs["src_audio"] = str(src_path)
                param_kwargs["repainting_start"] = float(repaint_start_sec)
                param_kwargs["repainting_end"] = float(repaint_end_sec)

            logger.info(
                "ACE-Step 1.5 [%s] task=%s lang=%s (cot_lang=%s) dur=%s bpm=%s key=%s steps=%s cfg=%s shift=%s thinking=%s lm_cfg=%s",
                display_name,
                task_type,
                resolved_lang,
                bool(use_thinking and use_cot_lang),
                target_dur,
                param_kwargs["bpm"],
                clean_key or "Auto",
                eff_steps,
                eff_guidance,
                eff_shift,
                use_thinking,
                lm_cfg_scale,
            )

            params = params_cls(**_filter_kwargs(params_cls, param_kwargs))
            config = config_cls(
                **_filter_kwargs(
                    config_cls,
                    {
                        "batch_size": 1,
                        "audio_format": "wav",
                        "use_random_seed": seed is None,
                        "seeds": [int(seed)] if seed is not None else None,
                    },
                )
            )
            result = gen_fn(
                **_filter_kwargs(
                    gen_fn,
                    {
                        "dit_handler": dit_handler,
                        "llm_handler": llm_handler if use_thinking else None,
                        "params": params,
                        "config": config,
                        "save_dir": str(tmpdir),
                    },
                )
            )
            if not getattr(result, "success", True):
                raise RuntimeError(f"ACE-Step 1.5 inference failed: {getattr(result, 'error', 'unknown error')}")

            out_file: Path | None = None
            cot_meta: Any = getattr(result, "extra_outputs", None) or getattr(result, "metas", None)
            audios = getattr(result, "audios", None)
            if isinstance(audios, (list, tuple)) and len(audios) > 0:
                first = audios[0]
                if isinstance(first, dict) and first.get("path"):
                    out_file = Path(str(first["path"]))
                    if cot_meta is None and "params" in first:
                        cot_meta = first["params"]
                elif hasattr(first, "path") and getattr(first, "path"):
                    out_file = Path(str(getattr(first, "path")))
                elif isinstance(first, (str, Path)):
                    out_file = Path(str(first))
            if out_file is None or not out_file.is_file():
                resolved_src = src_path.resolve() if src_path is not None else None
                candidates = [
                    p
                    for p in (sorted(Path(tmpdir).rglob("*.wav")) + sorted(Path(tmpdir).rglob("*.flac")))
                    if p.resolve() != resolved_src
                ]
                if not candidates:
                    raise FileNotFoundError("ACE-Step 1.5 inference produced no output audio file.")
                out_file = candidates[0]

            stereo_out = _read_and_fit_stereo(out_file, target_sr, total_samples)
            logger.info(
                "ACE-Step 1.5 output ready: %.2fs (%d samples @ %d Hz), cot_meta=%s",
                stereo_out.shape[0] / float(target_sr),
                stereo_out.shape[0],
                target_sr,
                cot_meta,
            )
            return stereo_out

        pipe = pipe_bundle["pipeline"] if isinstance(pipe_bundle, dict) and "pipeline" in pipe_bundle else pipe_bundle
        out_path = Path(tmpdir) / "out.wav"
        pipe_kwargs: dict[str, Any] = {
            "audio_duration": float(duration_sec) if duration_sec is not None else 30.0,
            "prompt": prompt,
            "lyrics": lyrics,
            "infer_step": int(inference_steps),
            "guidance_scale": float(guidance_scale),
            "save_path": str(out_path),
        }
        if seed is not None:
            pipe_kwargs["manual_seeds"] = [int(seed)]
        if src_path is not None:
            pipe_kwargs["task"] = task_type
            pipe_kwargs["repaint_start"] = float(repaint_start_sec)
            pipe_kwargs["repaint_end"] = float(repaint_end_sec)
            pipe_kwargs["src_audio_path"] = str(src_path)
        pipe(**pipe_kwargs)
        return _read_and_fit_stereo(out_path, target_sr, total_samples)


def resolve_mock_generation_params(
    *,
    bpm: float | None,
    musical_key: str | None,
    duration_sec: float | None,
    seed: int | None,
) -> tuple[float, str, float, int]:
    """Resolve deterministic synthetic defaults when MOCK_GPU=true receives Auto (None) fields."""
    resolved_bpm = float(bpm) if bpm is not None else 116.0
    resolved_key = (musical_key or "").strip() or "A minor"
    resolved_dur = float(duration_sec) if duration_sec is not None else 28.0
    resolved_seed = int(seed) if seed is not None else 421337
    return resolved_bpm, resolved_key, resolved_dur, resolved_seed


async def generate_full_song(
    *,
    bpm: float | None = None,
    musical_key: str | None = None,
    duration_sec: float | None = None,
    style_tags: list[str],
    structured_lyrics: str,
    vocal_language: str | None = None,
    seed: int | None = None,
    guidance_scale: float = 1.0,
    inference_steps: int = 8,
    shift: float = 3.0,
) -> tuple[np.ndarray, int]:
    """Generate a full stereo mix at canonical 44,100 Hz using ACE-Step 1.5 in bfloat16."""
    target_sr = settings.sample_rate

    async with model_manager.acquire(ModelType.ACE_STEP, _load_acestep_model) as pipe:
        if settings.mock_gpu:
            await asyncio.sleep(0.15)
            mock_bpm, mock_key, mock_dur, mock_seed = resolve_mock_generation_params(
                bpm=bpm,
                musical_key=musical_key,
                duration_sec=duration_sec,
                seed=seed,
            )
            total_samples = max(target_sr, int(round(mock_dur * target_sr)))
            audio = await asyncio.to_thread(
                _synthesize_musical_mix,
                total_samples,
                target_sr,
                mock_bpm,
                mock_key,
                mock_seed,
                style_tags,
                "full",
            )
            return audio, target_sr

        prompt = _build_acestep_prompt(style_tags, bpm, musical_key)
        total_samples = max(target_sr, int(round(duration_sec * target_sr))) if duration_sec is not None else None
        stereo = await asyncio.to_thread(
            _invoke_acestep_real_sync,
            pipe,
            task_type="text2music",
            prompt=prompt,
            lyrics=structured_lyrics,
            bpm=bpm,
            musical_key=musical_key,
            duration_sec=duration_sec,
            seed=seed,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            target_sr=target_sr,
            total_samples=total_samples,
            shift=shift,
            vocal_language=vocal_language,
        )
        return stereo, target_sr


async def repaint_region(
    *,
    source_mix: np.ndarray,
    start_sample: int,
    end_sample: int,
    bpm: float,
    musical_key: str,
    style_tags: list[str],
    lyrics_override: str,
    seed: int,
    inference_steps: int = 8,
    guidance_scale: float = 1.0,
    shift: float = 3.0,
    target_layer_role: str = "full",
    instrument_hint: str = "",
) -> np.ndarray:
    """
    Generate a repainted buffer covering [0 .. total_samples] (or matching slice)
    with ACE-Step 1.5 regional repaint / audio2audio conditioning.
    """
    target_sr = settings.sample_rate
    total_samples = int(source_mix.shape[0])
    region_len = max(1, end_sample - start_sample)

    async with model_manager.acquire(ModelType.ACE_STEP, _load_acestep_model) as pipe:
        if settings.mock_gpu:
            await asyncio.sleep(0.15)
            # Synthesize fresh musical segment with requested seed & layer role
            fresh = await asyncio.to_thread(
                _synthesize_musical_mix,
                total_samples,
                target_sr,
                bpm,
                musical_key,
                seed,
                style_tags,
                target_layer_role if target_layer_role != "full_mix" else "full",
                instrument_hint,
            )
            return fresh[start_sample:end_sample]

        prompt = _build_acestep_prompt(
            style_tags,
            bpm,
            musical_key,
            extra_prefix=[instrument_hint] if instrument_hint else None,
        )

        stereo = await asyncio.to_thread(
            _invoke_acestep_real_sync,
            pipe,
            task_type="repaint",
            prompt=prompt,
            lyrics=lyrics_override,
            bpm=bpm,
            musical_key=musical_key,
            duration_sec=total_samples / float(target_sr),
            seed=seed,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            target_sr=target_sr,
            total_samples=total_samples,
            shift=shift,
            src_audio=source_mix,
            repaint_start_sec=start_sample / float(target_sr),
            repaint_end_sec=end_sample / float(target_sr),
        )
        seg = stereo[start_sample:end_sample]
        if seg.shape[0] < region_len:
            pad = np.zeros((region_len - seg.shape[0], 2), dtype=np.float32)
            seg = np.concatenate([seg, pad], axis=0)
        return np.ascontiguousarray(seg[:region_len], dtype=np.float32)
