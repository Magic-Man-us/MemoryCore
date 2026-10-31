"""Injectable local reasoning steps. The engine validates every returned decision."""

from typing import Protocol

from .models import DreamGroup, DreamProposal, Memory, MemoryHit, Reconciliation


class Reconciler(Protocol):
    async def __call__(
        self, memory: Memory, neighbors: list[MemoryHit]
    ) -> Reconciliation: ...


class DreamPlanner(Protocol):
    async def __call__(self, groups: list[DreamGroup]) -> DreamProposal: ...
