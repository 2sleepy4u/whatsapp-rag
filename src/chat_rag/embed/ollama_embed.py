from __future__ import annotations

import ollama


class OllamaEmbedder:
    """Embedding client backed by a local Ollama server (GPU on the desktop)."""

    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self._client = ollama.Client(host=host)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embed(model=self.model, input=texts)
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")
        if embeddings is None:
            raise RuntimeError(f"Unexpected Ollama embed response: {response!r}")
        return [list(vec) for vec in embeddings]
