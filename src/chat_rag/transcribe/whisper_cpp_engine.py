from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .base import TranscriptResult


class WhisperCppEngine:
    """GPU transcription via a whisper.cpp binary.

    Build whisper.cpp with Vulkan and point the env vars at the result:

        CHAT_RAG_WHISPER_CPP_BIN=/path/to/whisper-cli
        CHAT_RAG_WHISPER_CPP_MODEL=/path/to/ggml-medium.bin

    The binary needs to decode Opus/M4A; build it with ffmpeg or convert the
    media to 16 kHz WAV first.
    """

    name = "whisper.cpp"

    def __init__(self, bin_path: str, model_path: str, language: str = "it", threads: int | None = None) -> None:
        self.bin = Path(bin_path)
        self.model_path = Path(model_path)
        self.model = self.model_path.name
        self.language = language
        self.threads = threads

        if not bin_path or not self.bin.exists():
            raise FileNotFoundError(f"whisper.cpp binary not found: {bin_path!r}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"whisper.cpp model not found: {model_path!r}")

    def transcribe(self, path: Path) -> TranscriptResult:
        with tempfile.TemporaryDirectory() as tmp:
            out_prefix = Path(tmp) / "out"
            cmd = [
                str(self.bin),
                "-m", str(self.model_path),
                "-f", str(path),
                "-l", self.language,
                "-oj",
                "-of", str(out_prefix),
            ]
            if self.threads:
                cmd += ["-t", str(self.threads)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"whisper.cpp failed ({proc.returncode}): {proc.stderr.strip()[:300]}")

            data = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))

        parts: list[str] = []
        duration: float | None = None
        for item in data.get("transcription", []):
            text = (item.get("text") or "").strip()
            if text:
                parts.append(text)
            offsets = item.get("offsets") or {}
            if offsets.get("to"):
                duration = offsets["to"] / 1000.0
        return TranscriptResult(text=" ".join(parts).strip(), language=self.language, duration=duration)
