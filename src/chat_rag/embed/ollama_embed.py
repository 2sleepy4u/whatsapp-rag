from __future__ import annotations

import ollama


class OllamaEmbedder:
    """Embedding client backed by a local Ollama server (GPU on the desktop)."""

    def __init__(self, model: str, host: str, keep_alive: str | int = "30m") -> None:
        self.model = model
        self.keep_alive = keep_alive
        self._client = ollama.Client(host=host)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embed(
            model=self.model, input=texts, keep_alive=self.keep_alive
        )
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")
        if embeddings is None:
            raise RuntimeError(f"Unexpected Ollama embed response: {response!r}")
        return [list(vec) for vec in embeddings]
