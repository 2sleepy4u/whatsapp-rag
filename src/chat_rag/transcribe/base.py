from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..config import Settings


@dataclass
class TranscriptResult:
    text: str
    language: str | None = None
    duration: float | None = None


@runtime_checkable
class TranscriptionEngine(Protocol):
    name: str
    model: str

    def transcribe(self, path: Path) -> TranscriptResult:
        ...


def build_engine(settings: Settings, engine: str | None = None, model: str | None = None) -> TranscriptionEngine:
    """Pick a transcription engine.

    ``engine`` may be ``auto`` (default), ``faster-whisper`` or ``whisper.cpp``.
    ``auto`` prefers whisper.cpp when a binary+model are configured (GPU on the
    desktop), otherwise falls back to faster-whisper (CPU).
    """
    choice = (engine or settings.transcribe_engine or "auto").lower()
    cpp_ready = bool(settings.whisper_cpp_bin and settings.whisper_cpp_model)

    if choice in {"auto", "whisper.cpp", "whisper_cpp"} and (cpp_ready or choice != "auto"):
        from .whisper_cpp_engine import WhisperCppEngine

        return WhisperCppEngine(
            bin_path=settings.whisper_cpp_bin or "",
            model_path=settings.whisper_cpp_model or "",
            language=settings.whisper_language,
        )

    from .faster_whisper_engine import FasterWhisperEngine

    return FasterWhisperEngine(
        model=model or settings.whisper_model,
        language=settings.whisper_language,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
