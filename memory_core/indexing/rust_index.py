"""Thin wrapper over the Rust engine (``memory_core._native``)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from memory_core._native import PyMemoryCandidate, PyMemoryEngine
from memory_core.core.models import MemoryCandidate, MemoryTrace

if TYPE_CHECKING:
    from collections.abc import Iterable


def _candidate_from_rust(c: PyMemoryCandidate) -> MemoryCandidate:
    """Convert the Rust binding object into our pydantic DTO."""
    # PyMemoryCandidate.created_at is i64 unix timestamp from Rust
    return MemoryCandidate(
        trace_uid=c.trace_uid,
        score=c.score,
        summary=c.summary,
        tags=list(c.tags),
        created_at=datetime.fromtimestamp(c.created_at, UTC),
    )


class RustMemoryIndex:
    """Thin wrapper over the Rust engine that satisfies the MemoryIndex protocol."""

    def __init__(self, engine: PyMemoryEngine | None = None) -> None:
        self._engine = engine or PyMemoryEngine()

    @property
    def engine(self) -> PyMemoryEngine:
        return self._engine

    def ingest(self, trace: MemoryTrace, embedding: Iterable[float]) -> None:
        """Insert or update a trace + embedding pair in the Rust engine."""
        args = trace.to_rust_args()
        self._engine.ingest_trace(*args, list(embedding))

    def mark_accessed(self, trace_uid: str) -> None:
        """Increment the access counter for the trace."""
        self._engine.mark_accessed(trace_uid)

    def remove(self, trace_uid: str) -> None:
        """Delete a trace from the index."""
        self._engine.remove_trace(trace_uid)

    def search(
        self,
        *,
        user_id: str,
        text: str,
        tags: list[str],
        limit: int,
        query_embedding: Iterable[float],
    ) -> list[MemoryCandidate]:
        """Query the index for relevant traces using the Rust search bindings."""
        raw = self._engine.search_candidates(
            user_id=user_id,
            text=text,
            tags=tags,
            limit=limit,
            query_embedding=list(query_embedding),
        )
        return [_candidate_from_rust(c) for c in raw]
