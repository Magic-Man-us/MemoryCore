"""Wrapper around the Rust SMK assistant index (``memory_core._native``)."""

from __future__ import annotations

import hashlib
import threading
from typing import TYPE_CHECKING

from pydantic import BaseModel

from memory_core._native import PyAssistantMemoryIndex, PySmkQuery, pack_smk
from memory_core.types.smk_features import build_smk_features

if TYPE_CHECKING:
    from collections.abc import Iterable

    from memory_core.core.models import AssistantMemoryTrace
    from memory_core.types.smk_types import (
        Level2Bits,
        MemoryKind,
        ToolFlag,
        TopicBucket,
    )


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
    """Wrapper around the Rust SMK assistant index.

    Ingest is an upsert keyed by ``trace_uid``: re-ingesting a trace replaces it.
    The uid <-> id map is rebuilt from the assistant store on hydration, so it needs
    no persistence of its own.
    """

    def __init__(self, dim: int) -> None:
        self._inner = PyAssistantMemoryIndex(dim)
        self._id_to_uid: dict[int, str] = {}
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._inner.dim

    def __len__(self) -> int:
        return len(self._inner)

    def prepare(self, trace: AssistantMemoryTrace, embedding: Iterable[float]) -> list[float]:
        """Validate a trace + embedding for ingest without touching the index.

        Checks the dimension, packs (and so validates) the SMK, and checks for an
        id collision. Sets ``trace.smk_raw`` and returns the embedding as a list, so
        callers can persist first and publish to the index only once durable.
        """
        emb = list(embedding)
        if len(emb) != self.dim:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.dim}, got {len(emb)}"
            )
        smk = build_smk_features(trace)
        smk_raw = pack_smk(
            int(smk.topic),
            int(smk.kind),
            int(smk.tool_mask),
            int(smk.difficulty),
            int(smk.generality),
            int(smk.importance),
        )
        mem_id = stable_memory_id(trace.trace_uid)
        with self._lock:
            owner = self._id_to_uid.get(mem_id)
        if owner is not None and owner != trace.trace_uid:
            raise ValueError(f"assistant index id collision: {trace.trace_uid!r} vs {owner!r}")
        trace.smk_raw = smk_raw
        return emb

    def ingest(self, trace: AssistantMemoryTrace, embedding: Iterable[float]) -> int:
        """Insert or replace a trace; sets and returns ``trace.smk_raw``.

        Raises ValueError on a dimension mismatch or (astronomically unlikely) a
        64-bit id collision with a different trace uid.
        """
        smk = build_smk_features(trace)
        mem_id = stable_memory_id(trace.trace_uid)
        with self._lock:
            owner = self._id_to_uid.get(mem_id)
            if owner is not None and owner != trace.trace_uid:
                raise ValueError(
                    f"assistant index id collision: {trace.trace_uid!r} vs {owner!r}"
                )
            smk_raw = self._inner.add(
                id=mem_id,
                topic=int(smk.topic),
                kind=int(smk.kind),
                tool_mask=int(smk.tool_mask),
                difficulty=int(smk.difficulty),
                generality=int(smk.generality),
                importance=int(smk.importance),
                embedding=list(embedding),
            )
            self._id_to_uid[mem_id] = trace.trace_uid
        trace.smk_raw = smk_raw
        return smk_raw

    def remove(self, trace_uid: str) -> bool:
        """Remove a trace. Returns False when it was not indexed."""
        mem_id = stable_memory_id(trace_uid)
        with self._lock:
            if self._id_to_uid.get(mem_id) != trace_uid:
                return False
            del self._id_to_uid[mem_id]
            return self._inner.remove(mem_id)

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
        for tool in required_tools or ():
            mask |= int(tool)

        smk_q = PySmkQuery(
            topic=int(topic) if topic is not None else None,
            required_tools_mask=mask,
            allowed_kinds=[int(kind) for kind in allowed_kinds] if allowed_kinds else None,
            min_generality=int(min_generality) if min_generality is not None else None,
            min_importance=int(min_importance) if min_importance is not None else None,
        )

        hits = self._inner.query_top_k_filtered(list(query_embedding), k, smk_q)
        with self._lock:
            uids = [self._id_to_uid.get(mem_id) for mem_id, _, _ in hits]
        # A uid missing here was removed between the native query and this lookup:
        # drop the hit rather than surface a bare numeric id.
        return [
            AssistantMemoryHit(trace_uid=uid, score=float(score), smk_raw=int(smk_raw))
            for uid, (_, score, smk_raw) in zip(uids, hits, strict=True)
            if uid is not None
        ]
