"""Host-bound memory context for existing Claude, OpenAI, and local sessions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

from pydantic import Field, PrivateAttr, computed_field

from memory_core.memory.errors import ContextRequiredError
from memory_core.memory.models import (
    ContextResponse,
    MemoryHit,
    Model,
    Nonempty,
    TaskContext,
)
from memory_core.runtime.context import render_context
from memory_core.runtime.core import MemoryCore

TokenBudget = Annotated[int, Field(ge=1, le=1_000_000)]
TokenCount = Annotated[int, Field(ge=0)]
SupportingLimit = Annotated[int, Field(ge=1, le=100)]
TokenCounter = Callable[[str], int]
MemoryRevisionNumber = Annotated[int, Field(ge=1)]
SupportingCount = Annotated[int, Field(ge=0)]
_CONTEXT_PREFIX = (
    "Current MemoryCore context, supplied as ordinary task data. This snapshot replaces earlier "
    "MemoryCore preferences/support for this session; withdrawn revisions no longer apply.\n"
)


class ContextUpdate(StrEnum):
    INITIAL = "initial"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


class SessionContextRequest(Model):
    context: TaskContext
    max_tokens: TokenBudget
    query: Nonempty | None = None
    limit: SupportingLimit = 5


class MemoryRevision(Model):
    id: Nonempty
    revision: MemoryRevisionNumber


class SessionContextPayload(ContextResponse):
    context: TaskContext
    _previous_revisions: dict[str, int] = PrivateAttr(default_factory=dict)

    @computed_field
    @property
    def withdrawn(self) -> list[MemoryRevision]:
        current = _revisions(self, self.supporting)
        return [
            MemoryRevision(id=uid, revision=revision)
            for uid, revision in self._previous_revisions.items()
            if current.get(uid) != revision
        ]


class SessionContextResponse(ContextResponse):
    context: TaskContext
    withdrawn: list[MemoryRevision]
    update: ContextUpdate
    text: str
    used_tokens: TokenCount
    omitted_supporting: SupportingCount
    deduplicated_supporting: SupportingCount


@dataclass(frozen=True, slots=True)
class _Pending:
    response: SessionContextResponse
    context: TaskContext
    full: ContextResponse
    query: str | None
    supporting: tuple[MemoryHit, ...]


def _revisions(
    prepared: ContextResponse, supporting: list[MemoryHit] | tuple[MemoryHit, ...] = ()
) -> dict[str, int]:
    memories = [*prepared.preferences.effective, *(hit.memory for hit in supporting)]
    for conflict in prepared.preferences.conflicts:
        memories.extend(conflict.memories)
    return {memory.id: memory.revision for memory in memories}


def _same(
    left: _Pending, context: TaskContext, prepared: ContextResponse, query: str | None
) -> bool:
    return (
        left.context == context
        and left.query == query
        and left.full.preferences.effective == prepared.preferences.effective
        and left.full.preferences.conflicts == prepared.preferences.conflicts
        and left.supporting == tuple(prepared.supporting)
    )


class SessionContext:
    def __init__(
        self, memory: MemoryCore, *, owner_id: Nonempty, count_tokens: TokenCounter
    ) -> None:
        if not owner_id:
            raise ValueError("authenticated owner is required")
        self._memory = memory
        self._owner_id = owner_id
        self._count_tokens = count_tokens
        self._lock = asyncio.Lock()
        self._injected: _Pending | None = None
        self._pending: _Pending | None = None

    async def prepare(self, request: SessionContextRequest) -> SessionContextResponse:
        async with self._lock:
            context = request.context.model_copy(deep=True)
            prepared = await self._memory.prepare_task_context(
                context,
                owner_id=self._owner_id,
                query=request.query,
                limit=request.limit,
            )
            if not isinstance(prepared, ContextResponse):
                raise TypeError("prepare_task_context returned an unexpected response")
            previous = self._injected
            previous_revisions = (
                _revisions(previous.full, previous.supporting) if previous else {}
            )
            payload = SessionContextPayload(
                context=context,
                preferences=prepared.preferences,
                supporting=prepared.supporting,
                receipt=prepared.receipt,
            )
            payload._previous_revisions = previous_revisions
            rendering = render_context(
                payload,
                budget=request.max_tokens,
                measure=self._count_tokens,
                prefix=_CONTEXT_PREFIX,
            )
            unchanged = previous is not None and _same(
                previous, context, rendering.prepared, request.query
            )
            response = SessionContextResponse(
                context=context,
                preferences=prepared.preferences,
                supporting=rendering.prepared.supporting,
                receipt=prepared.receipt,
                withdrawn=rendering.prepared.withdrawn,
                update=ContextUpdate.UNCHANGED
                if unchanged
                else ContextUpdate.INITIAL
                if previous is None
                else ContextUpdate.UPDATED,
                text="" if unchanged else rendering.text,
                used_tokens=0 if unchanged else rendering.used,
                omitted_supporting=rendering.omitted_supporting,
                deduplicated_supporting=rendering.deduplicated_supporting,
            )
            self._pending = _Pending(
                response,
                context.model_copy(deep=True),
                prepared.model_copy(deep=True),
                request.query,
                tuple(
                    hit.model_copy(deep=True) for hit in rendering.prepared.supporting
                ),
            )
            return response

    def reset(self) -> None:
        self._injected = None
        self._pending = None

    def commit(self, response: SessionContextResponse) -> None:
        if self._pending is None or self._pending.response is not response:
            raise ContextRequiredError(
                "commit requires this session's latest prepared response"
            )
        self._injected = self._pending

    async def before_action(self, response: SessionContextResponse) -> None:
        pending = self._pending
        if pending is None or pending.response is not response:
            raise ContextRequiredError(
                "action requires this session's latest prepared response"
            )
        if self._injected is not pending:
            raise ContextRequiredError(
                "inject and commit prepared context before acting"
            )
        await self._memory.service.validate_context(
            pending.full.receipt, pending.context, owner_id=self._owner_id
        )
        if self._pending is not pending or self._injected is not pending:
            raise ContextRequiredError("prepared context changed while validating")
        if pending.full.preferences.conflicts:
            raise ContextRequiredError("resolve preference conflicts before acting")
