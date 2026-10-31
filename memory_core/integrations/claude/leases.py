"""Cross-process owner/task exclusion with token fencing and crash expiry."""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import text

from .models import AgentRunConflict


@dataclass(frozen=True, slots=True)
class RunLease:
    memory: object
    owner: str
    task_id: str
    token: str

    def check(self, session):
        found = session.execute(
            text("""SELECT 1 FROM memory_agent_leases
            WHERE owner=:o AND task_id=:t AND token=:token AND expires>:now"""),
            {
                "o": self.owner,
                "t": self.task_id,
                "token": self.token,
                "now": time.time(),
            },
        ).first()
        if found is None:
            raise AgentRunConflict("task execution lease expired or was replaced")

    def validate(self):
        with (
            self.memory.service._lock,
            self.memory.queue.transaction(self.owner) as session,
        ):
            self.check(session)

    def release(self):
        with (
            self.memory.service._lock,
            self.memory.queue.transaction(self.owner) as session,
        ):
            session.execute(
                text(
                    "DELETE FROM memory_agent_leases WHERE owner=:o AND task_id=:t AND token=:token"
                ),
                {"o": self.owner, "t": self.task_id, "token": self.token},
            )


class RunLeases:
    def __init__(self, memory):
        self.memory = memory
        with memory.service.store.db.session_scope() as session:
            session.execute(
                text("""CREATE TABLE IF NOT EXISTS memory_agent_leases (
                owner TEXT NOT NULL, task_id TEXT NOT NULL, token TEXT NOT NULL,
                expires REAL NOT NULL, PRIMARY KEY(owner,task_id))""")
            )
            session.commit()

    def claim(self, owner, task_id, seconds):
        with self.memory.service._lock, self.memory.queue.transaction(owner) as session:
            row = session.execute(
                text(
                    "SELECT expires FROM memory_agent_leases WHERE owner=:o AND task_id=:t"
                ),
                {"o": owner, "t": task_id},
            ).first()
            if row and row[0] > time.time():
                raise AgentRunConflict(
                    "owner/task already running in another adapter or process"
                )
            lease = RunLease(self.memory, owner, task_id, str(uuid4()))
            session.execute(
                text("""INSERT INTO memory_agent_leases VALUES (:o,:t,:token,:expires)
                ON CONFLICT(owner,task_id) DO UPDATE SET token=excluded.token,expires=excluded.expires"""),
                {
                    "o": owner,
                    "t": task_id,
                    "token": lease.token,
                    "expires": time.time() + seconds,
                },
            )
            return lease
