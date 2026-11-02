"""Incremental per-owner indexes. Only changed rows are revalidated after hydration."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from heapq import nsmallest

from sqlalchemy import text

from memory_core._native import PyVectorIndex
from memory_core.indexing.semantic import HnswIndex, IndexPolicy
from memory_core.storage.codec import decode_embedding

from .metadata import MetadataIndex, Registry, value_key
from .models import MEMORY_ADAPTER, RetrievalPolicy
from .payloads import Payloads
from .reconciliation import equivalence_key


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
        include_retired=False,
        index_policy=None,
    ):
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
            if self.index_policy.backend == "hnsw"
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

    def sync(self, session, owner, generation):
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

    def update_rows(self, rows):
        for row in rows:
            payload, embedding = row[:2]
            row_generation = row[2] if len(row) > 2 else None
            memory = MEMORY_ADAPTER.validate_json(payload)
            if memory.id in self.headers:
                self.discard(memory.id)
            if memory.status == "superseded" and not self.include_retired:
                if memory.id in self.memories:
                    del self.memories[memory.id]
                self.metadata.remove(memory.id)
                self.vectors.remove(memory.id)
                continue
            self.memories[memory.id] = memory
            terms = [
                (field, value_key(v))
                for field, values in getattr(memory, "applicability", {}).items()
                for v in values
            ]
            self.headers[memory.id] = Header(
                memory.kind,
                memory.status,
                getattr(memory.scope, "project_id", None),
                tokens(memory.summary),
                self.key(memory),
                terms,
            )
            self._index_content(memory)
            self.metadata.upsert(
                memory.id,
                {**memory.metadata, "kind": memory.kind, "status": memory.status.value},
            )
            self.by_kind[memory.kind].add(memory.id)
            self.by_status[memory.status].add(memory.id)
            self.by_project[getattr(memory.scope, "project_id", None)].add(memory.id)
            for token in tokens(memory.summary):
                self.lexical[token].add(memory.id)
            if embedding and (
                row_generation is None or row_generation > self.restored_generation
            ):
                self.vectors.upsert(memory.id, decode_embedding(embedding))
            elif not embedding:
                self.vectors.remove(memory.id)

    def _index_content(self, memory):
        if memory.status == "active":
            self.equivalents[self.key(memory)].add(memory.id)
        if memory.kind != "preference" or memory.status == "superseded":
            return
        self.preference_fields[memory.id] = len(memory.applicability)
        if not memory.applicability:
            self.unconditional_preferences.add(memory.id)
        for field, values in memory.applicability.items():
            for value in values:
                self.preference_terms[field, value_key(value)].add(memory.id)

    def discard(self, uid):
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
        self.metadata.remove(uid)
        if uid in self.memories:
            del self.memories[uid]
        self.vectors.remove(uid)

    @staticmethod
    def key(memory):
        return hashlib.sha256(equivalence_key(memory).encode()).hexdigest()

    def matching_preferences(self, attributes, project_id):
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

    def eligible(self, project_id, *, active=True):
        scope = self.by_project[None] | self.by_project[project_id]
        return scope & self.by_status["active"] if active else scope

    def search(self, query, vector, ids, limit):
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
