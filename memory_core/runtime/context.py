"""Budgeted rendering shared by persistent sessions and provider adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from memory_core.memory.errors import ContextBudgetError
from memory_core.memory.models import ContextResponse


@dataclass(frozen=True, slots=True)
class ContextRendering[ContextT: ContextResponse]:
    text: str
    prepared: ContextT
    used: int
    omitted_supporting: int
    deduplicated_supporting: int


def render_context[ContextT: ContextResponse](
    prepared: ContextT,
    *,
    budget: int,
    measure: Callable[[str], int],
    prefix: str,
) -> ContextRendering[ContextT]:
    represented = {memory.id for memory in prepared.preferences.effective}
    for conflict in prepared.preferences.conflicts:
        represented.update(memory.id for memory in conflict.memories)
    supporting = []
    for hit in prepared.supporting:
        if hit.memory.id not in represented:
            supporting.append(hit)
            represented.add(hit.memory.id)
    mandatory = prepared.model_copy(update={"supporting": []})
    rendered = prefix + mandatory.model_dump_json()
    used = measure(rendered)
    if used < 0:
        raise ValueError("context measurement must be nonnegative")
    if used > budget:
        raise ContextBudgetError(
            "mandatory preferences/knowledge exceed context budget"
        )
    selected = mandatory
    for hit in supporting:
        candidate = selected.model_copy(
            update={"supporting": [*selected.supporting, hit]}
        )
        text = prefix + candidate.model_dump_json()
        size = measure(text)
        if size < 0:
            raise ValueError("context measurement must be nonnegative")
        if size > budget:
            continue
        selected, rendered, used = candidate, text, size
    return ContextRendering(
        rendered,
        selected,
        used,
        len(supporting) - len(selected.supporting),
        len(prepared.supporting) - len(supporting),
    )
