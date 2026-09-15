from __future__ import annotations

from pathlib import Path

from .base import TranscriptResult


class FasterWhisperEngine:
    """CPU transcription via faster-whisper (CTranslate2).

    Install with ``uv sync --extra voice``. On the i5-12400 prefer
    ``model="small"`` or ``"medium"`` with ``compute_type="int8"``; for GPU
    acceleration on the RX 6650 XT use whisper.cpp + Vulkan instead.
    """

    name = "faster-whisper"

    def __init__(
        self,
        model: str = "small",
        language: str = "it",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "faster-whisper is not installed. Run: uv sync --extra voice"
            ) from exc

        self.model = model
        self.language = language
        self.beam_size = beam_size
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, path: Path) -> TranscriptResult:
        segments, info = self._model.transcribe(
            str(path),
            language=self.language or None,
            beam_size=self.beam_size,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return TranscriptResult(
            text=text,
            language=getattr(info, "language", self.language),
            duration=getattr(info, "duration", None),
        )
