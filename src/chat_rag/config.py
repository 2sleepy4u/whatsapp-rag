from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "CHAT_RAG_"


def _env(name: str, default: str) -> str:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    tz: str
    ollama_host: str
    llm_model: str
    embed_model: str
    llm_num_predict: int
    llm_think: bool

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
        llm_think=_env_bool("LLM_THINK", False),
    )
