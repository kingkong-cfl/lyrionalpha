"""Strict single-model VRAM lifecycle manager for 24GB NVIDIA L4 GPUs."""

from __future__ import annotations

import asyncio
import gc
import logging
import types
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Callable

from app.config import settings

logger = logging.getLogger("lyrion.model_manager")

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - local CPU mode
    torch = None  # type: ignore


class ModelType(str, Enum):
    NONE = "none"
    ACE_STEP = "ace_step_1_5"  # Extensible for future ACE_STEP_BASE / lego on A100
    DEMUCS = "htdemucs"
    BASIC_PITCH = "basic_pitch"
    ORPHEUS = "orpheus"
    RVC = "rvc"


_SIMULATED_VRAM_MB: dict[ModelType, float] = {
    ModelType.NONE: 420.0,
    ModelType.ACE_STEP: 11850.0,
    ModelType.DEMUCS: 3940.0,
    ModelType.BASIC_PITCH: 980.0,
    ModelType.ORPHEUS: 6400.0,
    ModelType.RVC: 2150.0,
}


@dataclass(frozen=True)
class GpuTelemetry:
    gpu_name: str
    vram_used_mb: float
    vram_total_mb: float
    active_model: str
    mock_gpu: bool


_KNOWN_HEAVY_ATTRS = (
    "model",
    "llm",
    "vae",
    "text_encoder",
    "dcae",
    "vocoder",
    "dit",
    "lm_model",
    "silence_latent",
    "quantizer",
)


def _release_model_object(obj: Any, _seen: set[int] | None = None) -> None:
    """Recursively detach model handlers and tensor attributes so CUDA memory can be reclaimed."""
    if obj is None or isinstance(obj, (type, str, bytes, int, float, bool, types.ModuleType, types.FunctionType, types.BuiltinFunctionType)):
        return
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return
    _seen.add(obj_id)

    if isinstance(obj, dict):
        for val in list(obj.values()):
            _release_model_object(val, _seen)
        obj.clear()
        return

    unload_fn = getattr(obj, "unload", None)
    if callable(unload_fn):
        try:
            unload_fn()
        except Exception:
            pass

    for attr in _KNOWN_HEAVY_ATTRS:
        if hasattr(obj, attr):
            try:
                setattr(obj, attr, None)
            except Exception:
                pass

    obj_dict = getattr(obj, "__dict__", None)
    if isinstance(obj_dict, dict):
        for key in list(obj_dict.keys()):
            if key not in _KNOWN_HEAVY_ATTRS and not isinstance(
                obj_dict[key], (str, bytes, int, float, bool)
            ):
                obj_dict.pop(key, None)


class ModelManager:
    """Enforces mutual exclusion in GPU VRAM so heavy models never OOM an L4."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._active_type: ModelType = ModelType.NONE
        self._active_model: Any = None
        self.unload_count: int = 0

    @property
    def active_type(self) -> ModelType:
        return self._active_type

    @asynccontextmanager
    async def acquire(
        self,
        target: ModelType,
        loader_fn: Callable[[], Any],
        unload_after: bool = False,
    ) -> AsyncIterator[Any]:
        """Acquire exclusive access to the GPU with `target` loaded, unloading any prior model."""
        async with self._lock:
            if self._active_type != target:
                if self._active_type != ModelType.NONE:
                    logger.info("Swapping VRAM model: unloading %s -> loading %s", self._active_type.value, target.value)
                    self._unload_current()
                self._active_model = await asyncio.to_thread(loader_fn)
                self._active_type = target
            try:
                yield self._active_model
            finally:
                if unload_after:
                    self._unload_current()

    async def force_unload(self) -> None:
        async with self._lock:
            self._unload_current()

    def _unload_current(self) -> None:
        if self._active_model is not None:
            _release_model_object(self._active_model)
            del self._active_model
            self._active_model = None
        if self._active_type != ModelType.NONE:
            self.unload_count += 1
        self._active_type = ModelType.NONE
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def _active_model_label(self) -> str:
        if self._active_type == ModelType.NONE:
            return ModelType.NONE.value
        if isinstance(self._active_model, dict) and self._active_model.get("display_name"):
            return str(self._active_model["display_name"])
        custom_label = getattr(self._active_model, "_lyrion_display_name", "")
        if custom_label:
            return str(custom_label)
        return self._active_type.value

    def get_telemetry(self) -> GpuTelemetry:
        label = self._active_model_label()
        if not settings.mock_gpu and torch is not None and torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_idx)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            used_mb = round((total_bytes - free_bytes) / (1024 * 1024), 1)
            total_mb = round(props.total_memory / (1024 * 1024), 1)
            return GpuTelemetry(
                gpu_name=props.name,
                vram_used_mb=used_mb,
                vram_total_mb=total_mb,
                active_model=label,
                mock_gpu=False,
            )

        return GpuTelemetry(
            gpu_name="NVIDIA L4 (Mocked Local)" if settings.mock_gpu else "CPU Fallback",
            vram_used_mb=_SIMULATED_VRAM_MB.get(self._active_type, 420.0),
            vram_total_mb=24576.0,
            active_model=label,
            mock_gpu=settings.mock_gpu,
        )


model_manager = ModelManager()
