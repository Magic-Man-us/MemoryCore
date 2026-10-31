"""Explicit historical recall streams retained records without growing the hot cache."""

from heapq import nsmallest

from sqlalchemy import text

from .cache import OwnerCache
from .models import MemoryHit


def recall_history(service, session, owner, request, vector, registry):
    if request.filter is not None:
        registry.check_filter(request.filter)
    rows = session.execute(
        text("SELECT payload,embedding FROM memories_v1 WHERE owner=:o ORDER BY id"),
        {"o": owner},
    )
    best = []
    while batch := rows.fetchmany(256):
        cache = OwnerCache(
            service.spec.dimensions, service.retrieval_policy, include_retired=True
        )
        cache.update_rows(batch)
        ids = cache.eligible(request.project_id, active=False) & set.union(
            set(), *(cache.by_status[status] for status in request.statuses)
        )
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
