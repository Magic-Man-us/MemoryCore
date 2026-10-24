"""Thin wrapper over the Rust engine (``memory_core._native``)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from memory_core._native import PyMemoryCandidate, PyMemoryEngine
from memory_core.core.models import MemoryCandidate, MemoryTrace, ScoringWeights

if TYPE_CHECKING:
    from collections.abc import Iterable


def _candidate_from_rust(c: PyMemoryCandidate) -> MemoryCandidate:
    """Convert the Rust binding object into our pydantic DTO."""
    return MemoryCandidate(
        trace_uid=c.trace_uid,
        score=c.score,
        summary=c.summary,
        tags=list(c.tags),
        created_at=datetime.fromtimestamp(c.created_at, UTC),
        importance=c.importance,
        access_count=c.access_count,
    )


class RustMemoryIndex:
    """Thin wrapper over the Rust engine that satisfies the MemoryIndex protocol.

    The engine releases the GIL for every operation and guards its state with a
    read/write lock, so it is safe to share across worker threads.
    """

    def __init__(
        self,
        engine: PyMemoryEngine | None = None,
        *,
        weights: ScoringWeights | None = None,
    ) -> None:
        if engine is not None and weights is not None:
            raise ValueError("pass either a prebuilt engine or weights, not both")
        if engine is None:
            w = weights or ScoringWeights()
            engine = PyMemoryEngine(
                semantic_weight=w.semantic,
                importance_weight=w.importance,
                recency_weight=w.recency,
                access_weight=w.access,
                recency_half_life_days=w.recency_half_life_days,
            )
        self._engine = engine

    @property
    def engine(self) -> PyMemoryEngine:
        return self._engine

    def __len__(self) -> int:
        return len(self._engine)

    def check_embedding(self, embedding: list[float]) -> None:
        """Raise ValueError if ``ingest`` would reject this embedding."""
        if not embedding:
            raise ValueError("embedding cannot be empty")
        dim = self._engine.embedding_dim
        if dim is not None and len(embedding) != dim:
            raise ValueError(
                f"embedding dimension mismatch: expected {dim}, got {len(embedding)}"
            )

    def ingest(self, trace: MemoryTrace, embedding: Iterable[float]) -> None:
        """Insert or replace a trace + embedding pair in the Rust engine."""
        self._engine.ingest_trace(*trace.to_rust_args(), list(embedding))

    def mark_accessed(self, trace_uid: str) -> bool:
        """Increment the access counter for the trace."""
        return self._engine.mark_accessed(trace_uid)

    def remove(self, trace_uid: str) -> bool:
        """Delete a trace from the index and free its memory."""
        return self._engine.remove_trace(trace_uid)

    def search(
        self,
        *,
        user_id: str,
        text: str,
        tags: list[str],
        limit: int,
        query_embedding: Iterable[float],
        now: datetime | None = None,
    ) -> list[MemoryCandidate]:
        """Query the index for relevant traces using the Rust search bindings.

        ``now`` pins the clock for the recency term (tests, replays); it defaults to
        the system clock.
        """
        raw = self._engine.search_candidates(
            user_id=user_id,
            text=text,
            tags=tags,
            limit=limit,
            query_embedding=list(query_embedding),
            now=int(now.timestamp()) if now is not None else None,
        )
        return [_candidate_from_rust(c) for c in raw]
