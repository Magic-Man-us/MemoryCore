"""Injectable async providers. They receive bounded snapshots, never database handles."""

from typing import Protocol

from .models import (
    ContextContribution,
    ContextInput,
    Evidence,
    KnowledgeInput,
    KnowledgeResult,
)


class ContextResolver(Protocol):
    async def __call__(self, request: ContextInput) -> ContextContribution: ...


class Researcher(Protocol):
    async def __call__(self, request: KnowledgeInput) -> list[Evidence]: ...


class KnowledgeEvaluator(Protocol):
    async def __call__(self, request: KnowledgeInput) -> KnowledgeResult: ...
