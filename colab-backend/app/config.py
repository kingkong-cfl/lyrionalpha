"""Runtime configuration for Lyrion Colab L4 Backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_local_env() -> None:
    """Load key=value pairs from .env without requiring python-dotenv."""
    env_candidates = [
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in env_candidates:
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
        break


_load_local_env()


def _resolve_drive_root() -> Path:
    explicit = os.environ.get("DRIVE_ROOT", "").strip()
    colab_drive = Path("/content/drive/MyDrive")
    if explicit:
        # If running on Colab and DRIVE_ROOT was left as local ./data default, upgrade to Drive mount
        if explicit.startswith("./") and colab_drive.exists():
            return colab_drive / "AIMusicStudio"
        return Path(explicit).expanduser().resolve()
    if colab_drive.exists():
        return colab_drive / "AIMusicStudio"
    return (Path(__file__).resolve().parent.parent / "data" / "AIMusicStudio").resolve()


@dataclass(frozen=True)
class Settings:
    app_name: str = "Lyrion AI Music Studio Backend"
    version: str = "1.0.0"
    sample_rate: int = 44100
    crossfade_samples: int = 2205       # 50 ms equal-power splice window
    demucs_context_samples: int = 88200  # 2.0 s context margin for windowed re-stem
    mock_gpu: bool = os.environ.get("MOCK_GPU", "true").lower() in ("1", "true", "yes")
    drive_root: Path = _resolve_drive_root()
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "").strip()
    gemini_model: str = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()
    gemini_fallback_model: str = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite").strip()
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "").strip()
    groq_model: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    nvidia_nim_api_key: str = os.environ.get("NVIDIA_NIM_API_KEY", "").strip()
    nvidia_nim_model: str = os.environ.get("NVIDIA_NIM_MODEL", "meta/llama-3.3-70b-instruct").strip()
    api_token: str = os.environ.get("LYRION_API_TOKEN", "").strip()
    allowed_origins: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        raw_origins = os.environ.get("ALLOWED_ORIGINS", "*").strip()
        origins = [o.strip() for o in raw_origins.split(",") if o.strip()] or ["*"]
        object.__setattr__(self, "allowed_origins", origins)


settings = Settings()
settings.drive_root.mkdir(parents=True, exist_ok=True)
(settings.drive_root / "projects").mkdir(parents=True, exist_ok=True)
