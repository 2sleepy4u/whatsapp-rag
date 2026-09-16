from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "CHAT_RAG_"


def _env(name: str, default: str) -> str:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_opt_bool(name: str) -> bool | None:
    raw = os.environ.get(ENV_PREFIX + name)
    if raw is None or not raw.strip():
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    tz: str
    ollama_host: str
    llm_model: str
    embed_model: str
    llm_num_predict: int
    llm_think: bool | None = None
    llm_temperature: float = 0.0
    system_prompt_file: str | None = None
    embed_context: str = "none"
    transcribe_engine: str = "auto"
    whisper_model: str = "small"
    whisper_language: str = "it"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_cpp_bin: str | None = None
    whisper_cpp_model: str | None = None

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "chat.db"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    def ensure_dirs(self) -> None:
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    return Settings(
        data_dir=Path(_env("DATA_DIR", "./data")).expanduser(),
        tz=_env("TZ", "Europe/Rome"),
        ollama_host=_env("OLLAMA_HOST", "http://localhost:11434"),
        llm_model=_env("LLM_MODEL", "qwen2.5:7b"),
        embed_model=_env("EMBED_MODEL", "bge-m3"),
        llm_num_predict=int(_env("LLM_NUM_PREDICT", "1024")),
        llm_think=_env_opt_bool("LLM_THINK"),
        llm_temperature=float(_env("LLM_TEMPERATURE", "0.0")),
        system_prompt_file=_env("SYSTEM_PROMPT_FILE", "") or None,
        embed_context=_env("EMBED_CONTEXT", "none"),
        transcribe_engine=_env("TRANSCRIBE_ENGINE", "auto"),
        whisper_model=_env("WHISPER_MODEL", "small"),
        whisper_language=_env("WHISPER_LANGUAGE", "it"),
        whisper_device=_env("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=_env("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_cpp_bin=_env("WHISPER_CPP_BIN", "") or None,
        whisper_cpp_model=_env("WHISPER_CPP_MODEL", "") or None,
    )
