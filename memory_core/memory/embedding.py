"""Optional offline embedding adapter. Never downloads models or calls a hosted API."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from .models import EmbeddingSpec


class LocalSentenceEmbedder:
    def __init__(
        self,
        model_path: str | Path,
        *,
        model_id: str,
        revision: str,
        dimensions: int,
        batch_size: int = 32,
    ):
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError("model_path must be an existing local model directory")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        # Optional import: the core engine has no ML runtime dependency.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            str(path), local_files_only=True, trust_remote_code=False, device="cpu"
        )
        actual = self._model.get_sentence_embedding_dimension()
        if actual != dimensions:
            raise ValueError(
                f"expected {dimensions} embedding dimensions, got {actual}"
            )
        self.spec = EmbeddingSpec(
            model_id=model_id, revision=revision, dimensions=dimensions
        )
        self.batch_size = batch_size
        self._lock = threading.Lock()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        def run():
            with self._lock:
                return self._model.encode(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()

        return await asyncio.to_thread(run)
