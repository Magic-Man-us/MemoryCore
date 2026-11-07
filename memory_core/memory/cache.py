"""Incremental per-owner indexes. Only changed rows are revalidated after hydration."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from heapq import nsmallest
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from memory_core._native import PyVectorIndex
from memory_core.indexing.semantic import HnswIndex, IndexBackend, IndexPolicy
from memory_core.storage.codec import decode_embedding

from .metadata import MetadataIndex, Registry, value_key
from .models import (
    MEMORY_ADAPTER,
    MemoryKind,
    PreferenceMemory,
    RetrievalPolicy,
    Status,
)
from .payloads import Payloads
from .reconciliation import equivalence_key

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from datetime import datetime

    from sqlalchemy.orm import Session

    from .models import Memory, MetadataValue


def tokens(text_value: str) -> set[str]:
    return set(re.findall(r"\w+", text_value.casefold()))


@dataclass(slots=True)
class Header:
    kind: str
    status: str
    project: str | None
    tokens: set[str]
    equivalent: str
    terms: list[tuple]


class OwnerCache:
    def __init__(
        self,
        dim: int,
        policy: RetrievalPolicy,
        *,
        include_retired: bool = False,
        index_policy: IndexPolicy | None = None,
    ) -> None:
        self.include_retired = include_retired
        self.policy = policy
        self.generation = -1
        self.registry = Registry([])
        self.registry_generation = -1
        self.index_policy = index_policy or IndexPolicy()
        self.memories = (
            {} if include_retired else Payloads(self.index_policy.payload_cache_size)
        )
        self.headers = {}
        self.restored_generation = -1
        self.vectors = (
            HnswIndex(dim, self.index_policy)
            if self.index_policy.backend == IndexBackend.HNSW
            else PyVectorIndex(dim)
        )
        self.metadata = MetadataIndex()
        self.lexical = defaultdict(set)
        self.by_kind = defaultdict(set)
        self.by_status = defaultdict(set)
        self.by_project = defaultdict(set)
        self.equivalents = defaultdict(set)
        self.preference_terms = defaultdict(set)
        self.preference_fields = {}
        self.unconditional_preferences = set()
        # Only records with a validity bound; unbounded records are always eligible.
        self.windows = {}

    def sync(self, session: Session, owner: str, generation: int) -> None:
        if isinstance(self.memories, Payloads):
            self.memories.bind(session, owner)
        if self.generation == generation:
            return
        rows = session.execute(
            text(
                """SELECT payload,embedding,generation FROM memories_v1 WHERE owner=:o AND generation>:g
                AND (:retired=1 OR json_extract(payload,'$.status')!='superseded')"""
            ),
            {
                "o": owner,
                "g": self.generation,
                "retired": int(self.include_retired or self.generation >= 0),
            },
        )
        self.update_rows(rows)
        for (uid,) in session.execute(
            text("SELECT id FROM memory_archives WHERE owner=:o AND generation>:g"),
            {"o": owner, "g": self.generation},
        ):
            self.discard(uid)
        self.generation = generation

    # Any: SQLAlchemy Row cells are untyped.
    def update_rows(self, rows: Iterable[Sequence[Any]]) -> None:
        for row in rows:
            payload, embedding = row[:2]
            row_generation = row[2] if len(row) > 2 else None
            memory = MEMORY_ADAPTER.validate_json(payload)
            if memory.id in self.headers:
                self.discard(memory.id)
            if memory.status == Status.SUPERSEDED and not self.include_retired:
                if memory.id in self.memories:
                    del self.memories[memory.id]
                self.metadata.remove(memory.id)
                self.vectors.remove(memory.id)
                continue
            self.memories[memory.id] = memory
            terms = [
                (field, value_key(v))
                for field, values in (
                    memory.applicability if isinstance(memory, PreferenceMemory) else {}
                ).items()
                for v in values
            ]
            self.headers[memory.id] = Header(
                memory.kind,
                memory.status,
                memory.scope.project_id,
                tokens(memory.summary),
                self.key(memory),
                terms,
            )
            self._index_content(memory)
            if memory.valid_from is not None or memory.valid_until is not None:
                self.windows[memory.id] = (memory.valid_from, memory.valid_until)
            self.metadata.upsert(
                memory.id,
                {
                    **memory.metadata,
                    "kind": memory.kind.value,
                    "status": memory.status.value,
                },
            )
            self.by_kind[memory.kind].add(memory.id)
            self.by_status[memory.status].add(memory.id)
            self.by_project[memory.scope.project_id].add(memory.id)
            for token in tokens(memory.summary):
                self.lexical[token].add(memory.id)
            vector = decode_embedding(embedding)
            if vector is None:
                self.vectors.remove(memory.id)
            elif row_generation is None or row_generation > self.restored_generation:
                self.vectors.upsert(memory.id, vector)

    def _index_content(self, memory: Memory) -> None:
        if memory.status == Status.ACTIVE:
            self.equivalents[self.key(memory)].add(memory.id)
        if memory.kind != MemoryKind.PREFERENCE or memory.status == Status.SUPERSEDED:
            return
        self.preference_fields[memory.id] = len(memory.applicability)
        if not memory.applicability:
            self.unconditional_preferences.add(memory.id)
        for field, values in memory.applicability.items():
            for value in values:
                self.preference_terms[field, value_key(value)].add(memory.id)

    def discard(self, uid: str) -> None:
        previous = self.headers.pop(uid, None)
        if previous:
            self.equivalents[previous.equivalent].discard(uid)
            if not self.equivalents[previous.equivalent]:
                del self.equivalents[previous.equivalent]
            for token in previous.tokens:
                self.lexical[token].discard(uid)
                if not self.lexical[token]:
                    del self.lexical[token]
            self.by_kind[previous.kind].discard(uid)
            self.by_status[previous.status].discard(uid)
            self.by_project[previous.project].discard(uid)
            self.preference_fields.pop(uid, None)
            self.unconditional_preferences.discard(uid)
            for term in previous.terms:
                self.preference_terms[term].discard(uid)
                if not self.preference_terms[term]:
                    del self.preference_terms[term]
        self.windows.pop(uid, None)
        self.metadata.remove(uid)
        if uid in self.memories:
            del self.memories[uid]
        self.vectors.remove(uid)

    def next_boundary(self, after: datetime) -> datetime | None:
        """Earliest validity bound later than ``after``, or None when none is pending."""
        return min(
            (
                bound
                for window in self.windows.values()
                for bound in window
                if bound is not None and bound > after
            ),
            default=None,
        )

    def outside_window(self, at: datetime) -> set[str]:
        return {
            uid
            for uid, (start, end) in self.windows.items()
            if (start is not None and at < start) or (end is not None and at >= end)
        }

    @staticmethod
    def key(memory: Memory) -> str:
        return hashlib.sha256(equivalence_key(memory).encode()).hexdigest()

    def matching_preferences(
        self, attributes: dict[str, MetadataValue], project_id: str | None
    ) -> set[str]:
        # OR within each field, AND across required fields. Only matching postings
        # are visited; unrelated preference payloads are never inspected.
        counts = defaultdict(int)
        for field, actual in attributes.items():
            matches = set()
            for value in actual if isinstance(actual, list) else [actual]:
                matches.update(self.preference_terms.get((field, value_key(value)), ()))
            for uid in matches:
                counts[uid] += 1
        candidates = self.unconditional_preferences | {
            uid for uid, count in counts.items() if count == self.preference_fields[uid]
        }
        return {
            uid
            for uid in candidates
            if uid in self.by_project[None] or uid in self.by_project[project_id]
        }

    def eligible(self, project_id: str | None, *, active: bool = True) -> set[str]:
        scope = self.by_project[None] | self.by_project[project_id]
        return scope & self.by_status["active"] if active else scope

    def search(
        self, query: str, vector: list[float], ids: set[str], limit: int
    ) -> list[tuple[str, float]]:
        # Candidate generation is independent; semantic hits cannot be excluded by
        # lexical seeding. Rank the union using cosine and query-token coverage.
        budget = max(
            limit * self.policy.candidate_multiplier, self.policy.candidate_floor
        )
        semantic = dict(self.vectors.search(vector, list(ids), min(len(ids), budget)))
        counts = defaultdict(int)
        query_tokens = tokens(query)
        for token in query_tokens:
            for uid in self.lexical.get(token, set()) & ids:
                counts[uid] += 1
        lexical = nsmallest(budget, counts, key=lambda uid: (-counts[uid], uid))
        missing = set(lexical) - semantic.keys()
        semantic.update(self.vectors.search(vector, sorted(missing), len(missing)))
        total_weight = self.policy.semantic_weight + self.policy.lexical_weight
        scores = {
            uid: (
                self.policy.semantic_weight * max(0.0, cosine)
                + self.policy.lexical_weight
                * counts.get(uid, 0)
                / max(len(query_tokens), 1)
            )
            / total_weight
            for uid, cosine in semantic.items()
        }
        return [
            (uid, scores[uid])
            for uid in sorted(scores, key=lambda uid: (-scores[uid], uid))
            if scores[uid] > self.policy.minimum_score
        ][:limit]
