"""Explicit historical recall streams retained records without growing the hot cache."""

from __future__ import annotations

import zlib
from datetime import UTC, datetime
from heapq import nsmallest
from typing import TYPE_CHECKING

from sqlalchemy import text

from .cache import OwnerCache
from .models import MemoryHit

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from .metadata import Registry
    from .models import RecallRequest
    from .service import MemoryService


def recall_history(
    service: MemoryService,
    session: Session,
    owner: str,
    request: RecallRequest,
    vector: list[float],
    registry: Registry,
) -> list[MemoryHit]:
    if request.filter is not None:
        registry.check_filter(request.filter)
    rows = session.execute(
        text(
            "SELECT payload,embedding FROM memories_v1 WHERE owner=:o UNION ALL SELECT payload,embedding FROM memory_archives WHERE owner=:o"
        ),
        {"o": owner},
    )
    at = request.as_of or datetime.now(UTC)
    best = []
    while batch := rows.fetchmany(256):
        cache = OwnerCache(
            service.spec.dimensions, service.retrieval_policy, include_retired=True
        )
        cache.update_rows(
            [(zlib.decompress(p) if isinstance(p, bytes) else p, e) for p, e in batch]
        )
        ids = cache.eligible(request.project_id, active=False) & set.union(
            set(), *(cache.by_status[status] for status in request.statuses)
        )
        ids -= cache.outside_window(at)
        if request.filter is not None:
            ids = cache.metadata.select(request.filter, registry, ids)
        best.extend(
            MemoryHit(memory=cache.memories[uid], score=score)
            for uid, score in cache.search(request.query, vector, ids, request.limit)
        )
        best = nsmallest(
            request.limit, best, key=lambda hit: (-hit.score, hit.memory.id)
        )
    return best
