"""Protocols the MemorySystem depends on: the vector index and the embedder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from memory_core.core.models import MemoryCandidate, MemoryTrace


class MemoryIndex(Protocol):
    """Abstract vector index for long-term memory traces.

    Implementations provide methods to ingest traces and to search for
    relevant memories given a query embedding and simple filters.
    """

    def ingest(self, trace: MemoryTrace, embedding: Iterable[float]) -> None:
        """Store or update a trace in the index."""
        ...

    def search(
        self,
        *,
        user_id: str,
        text: str,
        tags: list[str],
        limit: int,
        query_embedding: Iterable[float],
    ) -> list[MemoryCandidate]:
        """Search a user's memories; returns ranked candidates."""
        ...

    def mark_accessed(self, trace_uid: str) -> None:
        """Record that a trace has been accessed."""
        ...

    def remove(self, trace_uid: str) -> None:
        """Remove a trace from the index."""
        ...


class Embedder(Protocol):
    """Turns text into embedding vectors.

    MemoryCore takes no position on the model — pass any implementation (a local
    ONNX model, an embeddings API) to ``MemorySystem`` and ``remember``/``recall``
    can then be called with raw text, no vectors required at call sites.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each text; one vector per input, all the same dimension."""
        ...
