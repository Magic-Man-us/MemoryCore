"""Measured retrieval evaluation for dream recommendations and index/model experiments.

Run on a separate evaluation database with the same fixtures for both configurations.
This reports evidence; it never changes the live embedding identity or search policy.
"""

from __future__ import annotations

import statistics
import time

from pydantic import Field

from .models import Model, RecallRequest


class RetrievalCase(Model):
    query: str
    relevant_ids: set[str] = Field(min_length=1)
    project_id: str | None = None
    embedding: list[float] | None = None


class RetrievalEvaluation(Model):
    cases: int
    k: int
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_reciprocal_rank: float
    p50_ms: float
    p95_ms: float


async def evaluate_retrieval(
    service, cases: list[RetrievalCase], *, owner_id: str, k: int = 10
) -> RetrievalEvaluation:
    if not cases:
        raise ValueError("evaluation requires held-out query cases")
    recalls, precisions, reciprocals, timings = [], [], [], []
    for case in cases:
        started = time.perf_counter()
        response = await service.recall(
            RecallRequest(
                query=case.query,
                project_id=case.project_id,
                embedding=case.embedding,
                limit=k,
            ),
            owner_id=owner_id,
        )
        timings.append((time.perf_counter() - started) * 1000)
        ids = [hit.memory.id for hit in response.hits]
        matches = set(ids) & case.relevant_ids
        recalls.append(len(matches) / len(case.relevant_ids))
        precisions.append(len(matches) / k)
        reciprocals.append(
            next(
                (
                    1 / rank
                    for rank, uid in enumerate(ids, 1)
                    if uid in case.relevant_ids
                ),
                0,
            )
        )
    timings.sort()
    return RetrievalEvaluation(
        cases=len(cases),
        k=k,
        mean_recall_at_k=statistics.mean(recalls),
        mean_precision_at_k=statistics.mean(precisions),
        mean_reciprocal_rank=statistics.mean(reciprocals),
        p50_ms=statistics.median(timings),
        p95_ms=timings[min(len(timings) - 1, int(0.95 * len(timings)))],
    )
