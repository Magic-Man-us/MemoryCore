from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from memory_core_py.core.models import MemoryCandidate, MemoryTrace

"""Protocols and interfaces for memory indexing backends.

This module defines the MemoryIndex protocol, which is the abstraction
that MemorySystem depends on for long-term memory search.
"""


class MemoryIndex(Protocol):
    """Abstract vector index for long-term memory traces.

    Implementations provide methods to ingest traces and to search for
    relevant memories given a query embedding and simple filters.
    """

    def ingest(self, trace: MemoryTrace, embedding: Iterable[float]) -> None:
        """Store or update a trace in the index.

        Parameters
        ----------
        trace:
            The MemoryTrace metadata to index.
        embedding:
            The embedding vector for the trace.
        """

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
        """Search for candidate memories for a given user.

        Parameters
        ----------
        user_id:
            Identifier for the user whose memories should be searched.
        text:
            Optional free text associated with the query.
        tags:
            Tags that should be used as filters or hints.
        limit:
            Maximum number of candidates to return.
        query_embedding:
            Embedding vector that represents the query.

        Returns
        -------
        list[MemoryCandidate]
            Ranked candidate memories.
        """

        ...

    def mark_accessed(self, trace_uid: str) -> None:
        """Record that a trace has been accessed.

        Parameters
        ----------
        trace_uid:
            Unique identifier of the trace that was accessed.
        """

        ...

    def remove(self, trace_uid: str) -> None:
        """Remove a trace from the index.

        Parameters
        ----------
        trace_uid:
            Unique identifier of the trace to remove.
        """

        ...
