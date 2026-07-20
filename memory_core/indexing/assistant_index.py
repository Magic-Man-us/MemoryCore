"""Wrapper around the Rust SMK assistant index (``memory_core._native``)."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from pydantic import BaseModel

from memory_core._native import PyAssistantMemoryIndex, PySmkQuery
from memory_core.types.smk_features import build_smk_features

if TYPE_CHECKING:
    from collections.abc import Iterable

    from memory_core.core.models import AssistantMemoryTrace
    from memory_core.types.smk_types import Level2Bits, MemoryKind, ToolFlag, TopicBucket


def stable_memory_id(trace_uid: str) -> int:
    """A deterministic 64-bit id for a trace uid.

    ``hash(str)`` is salted per process (PYTHONHASHSEED), which would orphan every
    id on restart; a digest is stable across processes and machines.
    """
    return int.from_bytes(hashlib.blake2b(trace_uid.encode(), digest_size=8).digest(), "big")


class AssistantMemoryHit(BaseModel):
    """Compact representation of an assistant memory match."""

    trace_uid: str
    score: float
    smk_raw: int


class AssistantMemoryIndex:
    """Wrapper around the Rust SMK assistant index."""

    def __init__(self, dim: int) -> None:
        self._inner = PyAssistantMemoryIndex(dim)
        self._id_to_uid: dict[int, str] = {}

    def ingest(self, trace: AssistantMemoryTrace, embedding: Iterable[float]) -> None:
        smk = build_smk_features(trace)
        emb = list(embedding)

        mem_id = stable_memory_id(trace.trace_uid)
        self._id_to_uid[mem_id] = trace.trace_uid

        self._inner.add(
            id=mem_id,
            topic=int(smk.topic.value),
            kind=int(smk.kind.value),
            tool_mask=int(smk.tool_mask),
            difficulty=int(smk.difficulty.value),
            generality=int(smk.generality.value),
            importance=int(smk.importance.value),
            embedding=emb,
        )

    def query(
        self,
        query_embedding: Iterable[float],
        k: int,
        *,
        topic: TopicBucket | None = None,
        required_tools: set[ToolFlag] | None = None,
        allowed_kinds: list[MemoryKind] | None = None,
        min_generality: Level2Bits | None = None,
        min_importance: Level2Bits | None = None,
    ) -> list[AssistantMemoryHit]:
        mask = 0
        if required_tools:
            for t in required_tools:
                mask |= int(t)

        kinds_raw = None
        if allowed_kinds:
            kinds_raw = [int(kind.value) for kind in allowed_kinds]

        smk_q = PySmkQuery(
            topic=int(topic.value) if topic is not None else None,
            required_tools_mask=mask,
            allowed_kinds=kinds_raw,
            min_generality=int(min_generality.value) if min_generality else None,
            min_importance=int(min_importance.value) if min_importance else None,
        )

        hits = self._inner.query_top_k_filtered(list(query_embedding), k, smk_q)

        out: list[AssistantMemoryHit] = []
        for mem_id, score, smk_raw in hits:
            uid = self._id_to_uid.get(mem_id, str(mem_id))
            out.append(
                AssistantMemoryHit(
                    trace_uid=uid,
                    score=float(score),
                    smk_raw=int(smk_raw),
                )
            )
        return out
