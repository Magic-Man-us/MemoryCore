"""Injectable local reasoning steps. The engine validates every returned decision."""

from __future__ import annotations

from typing import Protocol

from .models import DreamGroup, DreamProposal, Memory, MemoryHit, Reconciliation


class Reconciler(Protocol):
    async def __call__(
        self, memory: Memory, neighbors: list[MemoryHit]
    ) -> Reconciliation: ...


class DreamPlanner(Protocol):
    async def __call__(self, groups: list[DreamGroup]) -> DreamProposal: ...


class Embedder(Protocol):
    """Turns text into embedding vectors; one vector per input, all the same dimension.

    With an embedder, remember and recall accept raw text and no vectors are needed at
    call sites.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]: ...
