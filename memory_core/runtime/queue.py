"""Durable task events and leased maintenance; all callers use owner-scoped transactions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import text

from memory_core.memory.errors import ConflictError
from memory_core.memory.models import TaskContext
from memory_core.storage.codec import JSON_VALUE, canonical_json

from .models import JobKind, JobStatus, JobView, TaskResponse

DDL = (
    """CREATE TABLE IF NOT EXISTS memory_job_retention (
        owner TEXT NOT NULL,id TEXT NOT NULL,completed_at REAL NOT NULL,
        compacted INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(owner,id))""",
    """CREATE TABLE IF NOT EXISTS memory_event_archives (
        owner TEXT NOT NULL,task_id TEXT NOT NULL,id TEXT NOT NULL,job_id TEXT NOT NULL,
        payload BLOB NOT NULL,archived_at REAL NOT NULL,PRIMARY KEY(owner,task_id,id))""",
    """CREATE TABLE IF NOT EXISTS memory_tasks (
    owner TEXT NOT NULL, id TEXT NOT NULL, context TEXT NOT NULL, objective TEXT NOT NULL,
    sources TEXT NOT NULL, completion TEXT, PRIMARY KEY(owner,id))""",
    """CREATE TABLE IF NOT EXISTS memory_events (
    owner TEXT NOT NULL, task_id TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
    job_id TEXT NOT NULL, PRIMARY KEY(owner,task_id,id))""",
    """CREATE TABLE IF NOT EXISTS memory_jobs (
    owner TEXT NOT NULL, id TEXT NOT NULL, task_id TEXT, kind TEXT NOT NULL,
    payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
    available REAL NOT NULL DEFAULT 0, lease_until REAL, token TEXT, result TEXT, error TEXT,
    PRIMARY KEY(owner,id))""",
    "CREATE INDEX IF NOT EXISTS memory_job_retention_age ON memory_job_retention(owner,compacted,completed_at,id)",
    "CREATE INDEX IF NOT EXISTS memory_jobs_task ON memory_jobs(owner,task_id,status)",
    "CREATE INDEX IF NOT EXISTS memory_jobs_ready ON memory_jobs(status,available,lease_until)",
    """CREATE TABLE IF NOT EXISTS memory_gaps (
    owner TEXT NOT NULL, task_id TEXT NOT NULL, id TEXT NOT NULL, gap TEXT NOT NULL,
    result TEXT, resolved_at REAL, PRIMARY KEY(owner,task_id,id))""",
)


def canonical(model):
    return canonical_json(model.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class Claim:
    owner: str
    id: str
    task_id: str | None
    kind: str
    payload: dict
    token: str
    attempts: int
    result: dict | None


class Queue:
    def __init__(self, service, policy):
        self.service, self.policy = service, policy
        with service.store.db.session_scope() as session:
            for ddl in DDL:
                session.execute(text(ddl))
            session.commit()

    def transaction(self, owner):
        return self.service.store.transaction(owner)

    @staticmethod
    def task(session, owner, task_id):
        row = (
            session.execute(
                text("SELECT * FROM memory_tasks WHERE owner=:o AND id=:id"),
                {"o": owner, "id": task_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise ConflictError("unknown task")
        return TaskResponse(
            context=TaskContext.model_validate_json(row["context"]),
            objective=row["objective"],
            sources=JSON_VALUE.validate_json(row["sources"]),
            outcome=JSON_VALUE.validate_json(row["completion"])["outcome"]
            if row["completion"]
            else None,
        )

    @staticmethod
    def enqueue(session, owner, kind, payload, task_id=None):
        job_id = str(uuid4())
        session.execute(
            text("""INSERT INTO memory_jobs(owner,id,task_id,kind,payload,available)
            VALUES (:o,:id,:task,:kind,:payload,:available)"""),
            {
                "o": owner,
                "id": job_id,
                "task": task_id,
                "kind": kind,
                "payload": JSON_VALUE.dump_json(payload).decode(),
                "available": time.time(),
            },
        )
        return job_id

    def capture(self, session, owner, task_id, event_id, payload):
        raw = canonical_json(payload)
        old = session.execute(
            text("""SELECT payload,job_id FROM memory_events
            WHERE owner=:o AND task_id=:t AND id=:id"""),
            {"o": owner, "t": task_id, "id": event_id},
        ).first()
        if old:
            if old[0] != raw:
                raise ConflictError("event ID reused with different content")
            return old[1]
        expired = session.execute(
            text(
                "SELECT 1 FROM memory_event_archives WHERE owner=:o AND task_id=:t AND id=:id"
            ),
            {"o": owner, "t": task_id, "id": event_id},
        ).first()
        if expired:
            raise ConflictError(
                "event replay window expired; evidence remains archived"
            )
        task = self.task(session, owner, task_id)
        if task.outcome is not None:
            raise ConflictError("task is already complete")
        job_id = self.enqueue(session, owner, JobKind.CAPTURE, payload, task_id)
        session.execute(
            text("INSERT INTO memory_events VALUES (:o,:t,:id,:p,:j)"),
            {"o": owner, "t": task_id, "id": event_id, "p": raw, "j": job_id},
        )
        return job_id

    def schedule_dream(self, session, owner):
        state = self.service.store.state(session, owner)
        if state["writes"] - state["dream_writes"] < self.policy.dream_after_writes:
            return
        pending = session.execute(
            text("""SELECT 1 FROM memory_jobs WHERE owner=:o
            AND kind='dream' AND status IN ('pending','running') LIMIT 1"""),
            {"o": owner},
        ).first()
        if pending is None:
            self.enqueue(session, owner, JobKind.DREAM, {})

    def claim(self, owner=None):
        # BEGIN IMMEDIATE serializes claims across processes, including expired leases.
        with self.service.store.db.session_scope() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            now = time.time()
            params = {"now": now, "max": self.policy.max_attempts}
            scope = " AND owner=:o" if owner is not None else ""
            if owner is not None:
                params["o"] = owner
            session.execute(
                text(
                    """UPDATE memory_jobs SET status='failed',
                error='LeaseAttemptsExhausted',token=NULL WHERE status='running'
                AND lease_until<=:now AND attempts>=:max"""
                    + scope
                ),
                params,
            )
            row = (
                session.execute(
                    text(
                        """SELECT * FROM memory_jobs WHERE
                ((status='pending' AND available<=:now) OR
                (status='running' AND lease_until<=:now)) AND attempts<:max"""
                        + scope
                        + " ORDER BY available,id LIMIT 1"
                    ),
                    params,
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                session.commit()
                return None
            token = str(uuid4())
            session.execute(
                text("""UPDATE memory_jobs SET status='running',
                token=:token,lease_until=:until,attempts=attempts+1 WHERE owner=:o AND id=:id"""),
                {
                    "o": row["owner"],
                    "id": row["id"],
                    "token": token,
                    "until": now + self.policy.lease_seconds,
                },
            )
            claim = Claim(
                row["owner"],
                row["id"],
                row["task_id"],
                row["kind"],
                JSON_VALUE.validate_json(row["payload"]),
                token,
                row["attempts"] + 1,
                JSON_VALUE.validate_json(row["result"]) if row["result"] else None,
            )
            session.commit()
            return claim

    @staticmethod
    def assert_lease(session, claim):
        found = session.execute(
            text("""SELECT 1 FROM memory_jobs WHERE owner=:o
            AND id=:id AND status='running' AND token=:token AND lease_until>:now"""),
            {
                "o": claim.owner,
                "id": claim.id,
                "token": claim.token,
                "now": time.time(),
            },
        ).first()
        if found is None:
            raise ConflictError("maintenance lease expired or was replaced")

    def finish(self, session, claim, status, result=None, error=None):
        self.assert_lease(session, claim)
        session.execute(
            text("""UPDATE memory_jobs SET status=:s,result=:r,error=:e,
            token=NULL,lease_until=NULL WHERE owner=:o AND id=:id"""),
            {
                "o": claim.owner,
                "id": claim.id,
                "s": status,
                "r": JSON_VALUE.dump_json(result).decode()
                if result is not None
                else None,
                "e": error,
            },
        )
        if status == JobStatus.DONE:
            session.execute(
                text(
                    "INSERT OR IGNORE INTO memory_job_retention(owner,id,completed_at) VALUES (:o,:id,:t)"
                ),
                {"o": claim.owner, "id": claim.id, "t": time.time()},
            )

    def fail(self, claim, error):
        with (
            self.service._owner_lock(claim.owner),
            self.transaction(claim.owner) as session,
        ):
            try:
                self.assert_lease(session, claim)
            except ConflictError:
                return  # A newer worker owns this job; never overwrite its result.
            status = (
                JobStatus.FAILED
                if claim.attempts >= self.policy.max_attempts
                else JobStatus.PENDING
            )
            self.finish(session, claim, status, error=type(error).__name__)
            session.execute(
                text("UPDATE memory_jobs SET available=:a WHERE owner=:o AND id=:id"),
                {
                    "o": claim.owner,
                    "id": claim.id,
                    "a": time.time()
                    + min(3600, self.policy.retry_seconds * 2 ** (claim.attempts - 1)),
                },
            )

    @staticmethod
    def view(row):
        return JobView(
            id=row["id"],
            task_id=row["task_id"],
            kind=row["kind"],
            status=row["status"],
            attempts=row["attempts"],
            error=row["error"],
            result=JSON_VALUE.validate_json(row["result"]) if row["result"] else None,
        )
