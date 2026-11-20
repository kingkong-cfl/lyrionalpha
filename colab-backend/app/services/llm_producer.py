"""Zero-VRAM AI Producer using external free LLM APIs with deterministic fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

from app.config import settings
from app.schemas import EnhanceMode, ProducerEnhanceResponse

logger = logging.getLogger("lyrion.llm_producer")

SYSTEM_PROMPT = """You are an expert music producer and songwriter for ACE-Step 1.5.
Convert the user's raw musical idea into a valid JSON object with ONLY these keys:
{
  "bpm": <number between 60 and 180>,
  "musical_key": "<e.g. A minor, C major, F# minor>",
  "time_signature": "4/4",
  "duration_sec": <number between 16 and 180>,
  "style_tags": ["<genre>", "<mood>", "<instrumentation>", "<vocal style>", "<tempo feel>"],
  "structured_lyrics": "<formatted lyrics with [Intro], [Verse 1], [Chorus], [Bridge], [Outro] tags>"
}
Return strict JSON without markdown fences."""

SYSTEM_PROMPT_TAGS = """You are an expert music producer for ACE-Step 1.5.
Based on the user's musical idea and any provided existing lyrics, propose ONLY rich, accurate musical style tags.
Return strict JSON without markdown fences:
{
  "style_tags": ["<genre>", "<subgenre/era>", "<mood>", "<lead instrumentation>", "<bass/drums texture>", "<vocal character>", "<mix feel>"]
}"""

SYSTEM_PROMPT_LYRICS = """You are an expert songwriter for ACE-Step 1.5.
Based on the user's musical idea, optional style tags, and optional existing draft lyrics, write or refine singable structured lyrics using [Verse 1], [Chorus], [Verse 2], [Bridge] headers.
Return strict JSON without markdown fences:
{
  "structured_lyrics": "<formatted lyrics with section headers>"
}"""


def _system_prompt_for_mode(mode: EnhanceMode) -> str:
    if mode == "tags":
        return SYSTEM_PROMPT_TAGS
    if mode == "lyrics":
        return SYSTEM_PROMPT_LYRICS
    return SYSTEM_PROMPT


def _compose_producer_user_text(
    raw_prompt: str,
    preferred_duration: float | None,
    mode: EnhanceMode,
    existing_style_tags: list[str] | None,
    existing_lyrics: str | None,
) -> str:
    lines = [raw_prompt.strip()]
    if mode == "all" and preferred_duration is not None:
        lines.append(f"Target duration_sec: {preferred_duration}")
    if existing_style_tags:
        lines.append(f"Existing style tags: {', '.join(t for t in existing_style_tags if t.strip())}")
    if existing_lyrics and existing_lyrics.strip():
        lines.append(f"Existing lyrics:\n{existing_lyrics.strip()}")
    return "\n\n".join(lines)


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            cleaned = cleaned[brace_start : brace_end + 1]
    return json.loads(cleaned)


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float = 6.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call_gemini(
    raw_prompt: str,
    model_name: str,
    preferred_duration: float | None,
    mode: EnhanceMode = "all",
    existing_style_tags: list[str] | None = None,
    existing_lyrics: str | None = None,
) -> ProducerEnhanceResponse:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        f"?key={settings.gemini_api_key}"
    )
    user_text = _compose_producer_user_text(
        raw_prompt, preferred_duration, mode, existing_style_tags, existing_lyrics
    )

    payload = {
        "system_instruction": {"parts": [{"text": _system_prompt_for_mode(mode)}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "temperature": 0.7,
            "responseMimeType": "application/json",
        },
    }
    data = _http_post_json(url, payload, {"Content-Type": "application/json"})
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    parsed = _extract_json_object(text)
    return _normalize_response(
        parsed,
        provider=f"gemini:{model_name}",
        preferred_duration=preferred_duration,
        mode=mode,
        existing_style_tags=existing_style_tags,
        existing_lyrics=existing_lyrics,
    )


def _call_openai_compat(
    base_url: str,
    api_key: str,
    model_name: str,
    provider_label: str,
    raw_prompt: str,
    preferred_duration: float | None,
    mode: EnhanceMode = "all",
    existing_style_tags: list[str] | None = None,
    existing_lyrics: str | None = None,
) -> ProducerEnhanceResponse:
    user_text = _compose_producer_user_text(
        raw_prompt, preferred_duration, mode, existing_style_tags, existing_lyrics
    )

    payload = {
        "model": model_name,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": _system_prompt_for_mode(mode)},
            {"role": "user", "content": user_text},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = _http_post_json(base_url, payload, headers)
    text = data["choices"][0]["message"]["content"]
    parsed = _extract_json_object(text)
    return _normalize_response(
        parsed,
        provider=provider_label,
        preferred_duration=preferred_duration,
        mode=mode,
        existing_style_tags=existing_style_tags,
        existing_lyrics=existing_lyrics,
    )


def _normalize_response(
    parsed: dict[str, Any],
    provider: str,
    preferred_duration: float | None,
    mode: EnhanceMode = "all",
    existing_style_tags: list[str] | None = None,
    existing_lyrics: str | None = None,
) -> ProducerEnhanceResponse:
    bpm = float(parsed.get("bpm", 120.0))
    bpm = max(50.0, min(220.0, bpm))
    duration = float(preferred_duration or parsed.get("duration_sec", 30.0))
    duration = max(8.0, min(240.0, duration))

    if mode == "lyrics":
        tags = [str(t).strip() for t in (existing_style_tags or []) if str(t).strip()]
    else:
        tags_raw = parsed.get("style_tags") or ["electronic", "melodic", "studio mix"]
        tags = [str(t).strip() for t in tags_raw if str(t).strip()][:10]

    if mode == "tags":
        lyrics = (existing_lyrics or "").strip()
    else:
        lyrics = str(
            parsed.get("structured_lyrics")
            or "[Verse 1]\nNeon shadows move across the floor\n\n[Chorus]\nEchoes in the neon light"
        ).strip()

    return ProducerEnhanceResponse(
        provider=provider,
        mode=mode,
        bpm=round(bpm, 1),
        musical_key=str(parsed.get("musical_key") or "A minor"),
        time_signature=str(parsed.get("time_signature") or "4/4"),
        duration_sec=round(duration, 1),
        style_tags=tags,
        structured_lyrics=lyrics,
    )


def _deterministic_fallback(
    raw_prompt: str,
    preferred_duration: float | None,
    mode: EnhanceMode = "all",
    existing_style_tags: list[str] | None = None,
    existing_lyrics: str | None = None,
) -> ProducerEnhanceResponse:
    lower = f"{raw_prompt} {existing_lyrics or ''} {' '.join(existing_style_tags or [])}".lower()
    bpm_match = re.search(r"(\d{2,3})\s*bpm", lower)
    if bpm_match:
        bpm = float(max(60, min(190, int(bpm_match.group(1)))))
    elif any(w in lower for w in ("dnb", "drum and bass", "fast")):
        bpm = 172.0
    elif any(w in lower for w in ("house", "techno", "dance", "club")):
        bpm = 126.0
    elif any(w in lower for w in ("lofi", "lo-fi", "chill", "ballad", "ambient")):
        bpm = 86.0
    elif any(w in lower for w in ("trap", "hip hop", "hip-hop")):
        bpm = 140.0
    else:
        bpm = 118.0

    key = "A minor"
    if any(w in lower for w in ("happy", "bright", "sunny", "pop", "uplifting")):
        key = "C major"
    elif any(w in lower for w in ("dark", "cyber", "techno", "cinematic")):
        key = "D minor"
    elif any(w in lower for w in ("jazz", "soul", "r&b", "neo-soul")):
        key = "F major"

    if mode == "lyrics":
        tags = [str(t).strip() for t in (existing_style_tags or []) if str(t).strip()]
    else:
        genre_map = [
            ("synthwave", ["synthwave", "retrowave", "analog synth", "80s", "retro"]),
            ("lofi", ["lo-fi", "warm vinyl", "chillhop", "dusty drums"]),
            ("rock", ["alt-rock", "electric guitar", "live drums", "overdrive"]),
            ("techno", ["melodic techno", "driving bassline", "analog arpeggio"]),
            ("hip hop", ["hip-hop", "punchy 808", "crisp snare", "atmospheric pads"]),
            ("acoustic", ["indie acoustic", "fingerpicked guitar", "warm upright bass"]),
            ("cinematic", ["cinematic orchestral", "deep brass", "wide strings"]),
        ]
        tags = []
        for keyword, matched_tags in genre_map:
            if keyword in lower:
                tags.extend(matched_tags)
        if not tags:
            tags = ["modern electronic pop", "warm analog bass", "crisp percussion", "wide stereo mix"]

        if "female" in lower:
            tags.append("female vocals")
        elif "male" in lower:
            tags.append("male vocals")
        elif "instrumental" in lower:
            tags.append("instrumental")
        else:
            tags.append("expressive lead vocals")
        tags = tags[:8]

    if mode == "tags":
        lyrics = (existing_lyrics or "").strip()
    elif existing_lyrics and existing_lyrics.strip() and "[" not in existing_lyrics:
        draft_lines = [ln.strip() for ln in existing_lyrics.strip().splitlines() if ln.strip()]
        mid = max(1, len(draft_lines) // 2)
        verse = "\n".join(draft_lines[:mid])
        chorus = "\n".join(draft_lines[mid:]) if mid < len(draft_lines) else verse
        lyrics = f"[Verse 1]\n{verse}\n\n[Chorus]\n{chorus}"
    else:
        theme = raw_prompt.strip().rstrip(".")
        lyrics = (
            f"[Verse 1]\nMoving through the static as the skyline fades to gold\n"
            f"Every beat is counting out the stories of {theme or 'tonight'}\n\n"
            f"[Chorus]\nTurn the signal higher in the midnight glow\n"
            f"Riding on the rhythm where the bright waves flow\n\n"
            f"[Bridge]\nHold the chord suspended till the silence breaks apart\n\n"
            f"[Outro]\n(Warm synth fade at {int(bpm)} BPM in {key})"
        )

    return ProducerEnhanceResponse(
        provider="local-rule-producer",
        mode=mode,
        bpm=bpm,
        musical_key=key,
        time_signature="4/4",
        duration_sec=float(preferred_duration or 30.0),
        style_tags=tags,
        structured_lyrics=lyrics,
    )


async def enhance_prompt(
    raw_prompt: str,
    preferred_duration_sec: float | None = None,
    mode: EnhanceMode = "all",
    existing_style_tags: list[str] | None = None,
    existing_lyrics: str | None = None,
) -> ProducerEnhanceResponse:
    """Call Gemini (2.5-flash -> 2.5-flash-lite), fallback to Groq/NIM, then deterministic parser."""
    if settings.gemini_api_key:
        for model_name in (settings.gemini_model, settings.gemini_fallback_model):
            if not model_name:
                continue
            try:
                return await asyncio.to_thread(
                    _call_gemini,
                    raw_prompt,
                    model_name,
                    preferred_duration_sec,
                    mode,
                    existing_style_tags,
                    existing_lyrics,
                )
            except Exception as exc:
                logger.warning("Gemini model '%s' failed (%s); trying fallback...", model_name, exc)

    if settings.groq_api_key:
        try:
            return await asyncio.to_thread(
                _call_openai_compat,
                "https://api.groq.com/openai/v1/chat/completions",
                settings.groq_api_key,
                settings.groq_model,
                f"groq:{settings.groq_model}",
                raw_prompt,
                preferred_duration_sec,
                mode,
                existing_style_tags,
                existing_lyrics,
            )
        except Exception as exc:
            logger.warning("Groq model '%s' failed (%s); trying fallback...", settings.groq_model, exc)

    if settings.nvidia_nim_api_key:
        try:
            return await asyncio.to_thread(
                _call_openai_compat,
                "https://integrate.api.nvidia.com/v1/chat/completions",
                settings.nvidia_nim_api_key,
                settings.nvidia_nim_model,
                f"nvidia-nim:{settings.nvidia_nim_model}",
                raw_prompt,
                preferred_duration_sec,
                mode,
                existing_style_tags,
                existing_lyrics,
            )
        except Exception as exc:
            logger.warning("NVIDIA NIM failed (%s); falling back to local producer.", exc)

    return _deterministic_fallback(
        raw_prompt,
        preferred_duration_sec,
        mode=mode,
        existing_style_tags=existing_style_tags,
        existing_lyrics=existing_lyrics,
    )
