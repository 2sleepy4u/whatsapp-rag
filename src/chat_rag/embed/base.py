from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """Anything that can turn a batch of texts into vectors."""

    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...
