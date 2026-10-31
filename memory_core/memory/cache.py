"""Incremental per-owner indexes. Only changed rows are revalidated after hydration."""

from __future__ import annotations

import re
from collections import defaultdict

from sqlalchemy import text

from memory_core._native import PyVectorIndex
from memory_core.storage.codec import decode_embedding

from .metadata import MetadataIndex
from .models import MEMORY_ADAPTER, RetrievalPolicy


def tokens(text_value: str) -> set[str]:
    return set(re.findall(r"\w+", text_value.casefold()))


class OwnerCache:
    def __init__(self, dim: int, policy: RetrievalPolicy):
        self.policy = policy
        self.generation = -1
        self.memories = {}
        self.vectors = PyVectorIndex(dim)
        self.metadata = MetadataIndex()
        self.lexical = defaultdict(set)
        self.by_kind = defaultdict(set)
        self.by_status = defaultdict(set)
        self.by_project = defaultdict(set)

    def sync(self, session, owner, generation):
        if self.generation == generation:
            return
        rows = session.execute(
            text(
                "SELECT payload,embedding FROM memories_v1 WHERE owner=:o AND generation>:g"
            ),
            {"o": owner, "g": self.generation},
        )
        for payload, embedding in rows:
            memory = MEMORY_ADAPTER.validate_json(payload)
            previous = self.memories.get(memory.id)
            if previous:
                for token in tokens(previous.summary):
                    self.lexical[token].discard(memory.id)
                self.by_kind[previous.kind].discard(memory.id)
                self.by_status[previous.status].discard(memory.id)
                self.by_project[getattr(previous.scope, "project_id", None)].discard(
                    memory.id
                )
            self.memories[memory.id] = memory
            self.metadata.upsert(
                memory.id,
                {**memory.metadata, "kind": memory.kind, "status": memory.status.value},
            )
            self.by_kind[memory.kind].add(memory.id)
            self.by_status[memory.status].add(memory.id)
            self.by_project[getattr(memory.scope, "project_id", None)].add(memory.id)
            for token in tokens(memory.summary):
                self.lexical[token].add(memory.id)
            if embedding:
                self.vectors.upsert(memory.id, decode_embedding(embedding))
            else:
                self.vectors.remove(memory.id)
        self.generation = generation

    def eligible(self, project_id, *, active=True):
        scope = self.by_project[None] | self.by_project[project_id]
        return scope & self.by_status["active"] if active else scope

    def search(self, query, vector, ids, limit):
        # Candidate generation is independent; semantic hits cannot be excluded by
        # lexical seeding. Rank the union using cosine and query-token coverage.
        budget = max(
            limit * self.policy.candidate_multiplier, self.policy.candidate_floor
        )
        semantic = dict(self.vectors.search(vector, sorted(ids), min(len(ids), budget)))
        counts = defaultdict(int)
        query_tokens = tokens(query)
        for token in query_tokens:
            for uid in self.lexical.get(token, set()) & ids:
                counts[uid] += 1
        lexical = sorted(counts, key=lambda uid: (-counts[uid], uid))[:budget]
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
