"""Bounded local retention: lossless cold archives, explicit replay windows."""

from __future__ import annotations

import asyncio
import time
import zlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import Field
from sqlalchemy import text

from memory_core.storage.codec import JSON_VALUE

from .models import JobKind, Model

if TYPE_CHECKING:
    from .core import MemoryCore

type ReclaimReport = dict[str, tuple[int, ...] | bool]


class RetentionPolicy(Model):
    enabled: bool = False
    interval_seconds: float = Field(default=300, ge=1)
    batch_size: int = Field(default=100, ge=1, le=1000)
    superseded_after_days: int = Field(default=30, ge=0)
    completed_job_days: int = Field(default=7, ge=0)
    event_replay_days: int = Field(default=90, ge=1)


class RetentionResult(Model):
    archived_memories: int = 0
    compacted_jobs: int = 0
    archived_events: int = 0
    archived_plans: int = 0


async def maintain(runtime: MemoryCore, owner_id: str) -> RetentionResult:
    policy = runtime.retention
    if not policy.enabled:
        return RetentionResult()

    def run() -> RetentionResult:
        now = time.time()
        cutoff = (
            datetime.now(UTC) - timedelta(days=policy.superseded_after_days)
        ).isoformat()
        counts = {}
        with (
            runtime.service._owner_lock(owner_id),
            runtime.queue.transaction(owner_id) as session,
        ):
            records = session.execute(
                text("""SELECT id,payload,embedding FROM memories_v1
                WHERE owner=:o AND json_extract(payload,'$.status')='superseded'
                AND julianday(json_extract(payload,'$.updated_at'))<julianday(:cutoff)
                ORDER BY generation,id LIMIT :n"""),
                {"o": owner_id, "cutoff": cutoff, "n": policy.batch_size},
            ).all()
            if records:
                generation = runtime.service.store.bump(
                    session,
                    owner_id,
                    "archive_superseded",
                    {"ids": [r[0] for r in records]},
                    write=False,
                )
                for uid, payload, embedding in records:
                    session.execute(
                        text("INSERT INTO memory_archives VALUES (:o,:id,:g,:p,:e,:t)"),
                        {
                            "o": owner_id,
                            "id": uid,
                            "g": generation,
                            "p": zlib.compress(payload.encode()),
                            "e": embedding,
                            "t": now,
                        },
                    )
                    session.execute(
                        text("DELETE FROM memories_v1 WHERE owner=:o AND id=:id"),
                        {"o": owner_id, "id": uid},
                    )
                counts["archived_memories"] = len(records)
            # Only completed jobs are compacted; failed/review/running work is untouched.
            jobs = session.execute(
                text("""SELECT j.id,j.kind,j.result FROM memory_jobs j JOIN memory_job_retention r
                ON r.owner=j.owner AND r.id=j.id WHERE j.owner=:o AND j.status='done'
                AND r.compacted=0 AND r.completed_at<:cutoff ORDER BY r.completed_at,j.id LIMIT :n"""),
                {
                    "o": owner_id,
                    "cutoff": now - policy.completed_job_days * 86400,
                    "n": policy.batch_size,
                },
            ).all()
            for uid, kind, raw_result in jobs:
                if kind == JobKind.DREAM and raw_result:
                    plan_id = JSON_VALUE.validate_json(raw_result).get("plan_id")
                    plan = session.execute(
                        text(
                            "SELECT payload FROM memory_dreams WHERE owner=:o AND id=:id"
                        ),
                        {"o": owner_id, "id": plan_id},
                    ).scalar()
                    if plan and JSON_VALUE.validate_json(plan).get("applied"):
                        session.execute(
                            text(
                                "INSERT OR IGNORE INTO memory_dream_archives VALUES (:o,:id,:p)"
                            ),
                            {
                                "o": owner_id,
                                "id": plan_id,
                                "p": zlib.compress(plan.encode()),
                            },
                        )
                        session.execute(
                            text("DELETE FROM memory_dreams WHERE owner=:o AND id=:id"),
                            {"o": owner_id, "id": plan_id},
                        )
                        counts["archived_plans"] = counts.get("archived_plans", 0) + 1
                # Results and task/event references remain available; only duplicate input is removed.
                session.execute(
                    text(
                        "UPDATE memory_jobs SET payload='{}' WHERE owner=:o AND id=:id"
                    ),
                    {"o": owner_id, "id": uid},
                )
                session.execute(
                    text(
                        "UPDATE memory_job_retention SET compacted=1 WHERE owner=:o AND id=:id"
                    ),
                    {"o": owner_id, "id": uid},
                )
            counts["compacted_jobs"] = len(jobs)
            events = session.execute(
                text("""SELECT e.task_id,e.id,e.job_id,e.payload FROM memory_events e
                JOIN memory_jobs j ON j.owner=e.owner AND j.id=e.job_id
                JOIN memory_job_retention r ON r.owner=j.owner AND r.id=j.id
                JOIN memory_tasks t ON t.owner=e.owner AND t.id=e.task_id
                WHERE e.owner=:o AND j.status='done' AND t.completion IS NOT NULL AND r.completed_at<:cutoff
                AND NOT EXISTS (SELECT 1 FROM memory_jobs u WHERE u.owner=e.owner AND u.task_id=e.task_id AND u.status!='done')
                AND NOT EXISTS (SELECT 1 FROM memory_gaps g WHERE g.owner=e.owner AND g.task_id=e.task_id AND
                  (g.result IS NULL OR json_extract(g.result,'$.status')!='resolved'))
                ORDER BY r.completed_at,e.id LIMIT :n"""),
                {
                    "o": owner_id,
                    "cutoff": now - policy.event_replay_days * 86400,
                    "n": policy.batch_size,
                },
            ).all()
            for task, uid, job, payload in events:
                session.execute(
                    text(
                        "INSERT INTO memory_event_archives VALUES (:o,:task,:id,:job,:p,:t)"
                    ),
                    {
                        "o": owner_id,
                        "task": task,
                        "id": uid,
                        "job": job,
                        "p": zlib.compress(payload.encode()),
                        "t": now,
                    },
                )
                session.execute(
                    text(
                        "DELETE FROM memory_events WHERE owner=:o AND task_id=:task AND id=:id"
                    ),
                    {"o": owner_id, "task": task, "id": uid},
                )
            counts["archived_events"] = len(events)
        return RetentionResult(**counts)

    return await asyncio.to_thread(run)


async def reclaim(runtime: MemoryCore, *, pages: int = 256) -> ReclaimReport:
    """Explicit idle maintenance; never change auto_vacuum mode on an existing DB."""
    if type(pages) is not int or not 1 <= pages <= 10000:
        raise ValueError("pages must be between 1 and 10000")

    def run() -> ReclaimReport:
        with runtime.service.store.db.session_scope() as session:
            connection = session.connection()
            mode = connection.exec_driver_sql("PRAGMA auto_vacuum").scalar_one()
            checkpoint = tuple(
                connection.exec_driver_sql("PRAGMA wal_checkpoint(PASSIVE)").one()
            )
            if mode == 2:
                connection.exec_driver_sql(f"PRAGMA incremental_vacuum({pages})")
            return {"checkpoint": checkpoint, "incremental_vacuum_enabled": mode == 2}

    return await asyncio.to_thread(run)
