"""Local SQLite durable state and atomic revisions using the existing Database connection.

Writes use BEGIN IMMEDIATE for reconciliation versions. Reads use a deferred
snapshot after one-time owner initialization.
"""

from __future__ import annotations

import zlib
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sqlalchemy import text

from memory_core.storage.codec import JSON_VALUE

from .metadata import DEFAULT_FIELDS
from .models import DreamPlan, FieldDefinition

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic import JsonValue
    from sqlalchemy.engine import RowMapping
    from sqlalchemy.orm import Session

    from memory_core.storage.database import Database

    from .models import Memory

DDL = (
    "CREATE TABLE IF NOT EXISTS memory_dream_archives (owner TEXT NOT NULL,id TEXT NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(owner,id))",
    """CREATE TABLE IF NOT EXISTS memory_archives (
        owner TEXT NOT NULL,id TEXT NOT NULL,generation INTEGER NOT NULL,
        payload BLOB NOT NULL,embedding BLOB,archived_at REAL NOT NULL,PRIMARY KEY(owner,id))""",
    "CREATE INDEX IF NOT EXISTS memory_archive_generation ON memory_archives(owner,generation)",
    """CREATE TABLE IF NOT EXISTS memory_vector_snapshots (
        owner TEXT PRIMARY KEY, identity TEXT NOT NULL, generation INTEGER NOT NULL,
        manifest TEXT NOT NULL, graph BLOB NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS memory_owners (
       owner TEXT PRIMARY KEY, generation INTEGER NOT NULL DEFAULT 0,
       writes INTEGER NOT NULL DEFAULT 0, dream_cursor INTEGER NOT NULL DEFAULT 0, dream_cursor_id TEXT NOT NULL DEFAULT '',
       dream_writes INTEGER NOT NULL DEFAULT 0, embedding_spec TEXT)""",
    """CREATE TABLE IF NOT EXISTS memories_v1 (
       owner TEXT NOT NULL, id TEXT NOT NULL, generation INTEGER NOT NULL,
       payload TEXT NOT NULL, embedding BLOB, PRIMARY KEY(owner,id))""",
    "CREATE INDEX IF NOT EXISTS memories_generation ON memories_v1(owner,generation)",
    "CREATE INDEX IF NOT EXISTS memories_scan_cursor ON memories_v1(owner,generation,id)",
    """CREATE TABLE IF NOT EXISTS memory_fields (
       owner TEXT NOT NULL, name TEXT NOT NULL, definition TEXT NOT NULL,
       PRIMARY KEY(owner,name))""",
    """CREATE TABLE IF NOT EXISTS memory_audit (
       seq INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL,
       generation INTEGER NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS memory_forgotten (
       owner TEXT NOT NULL, id TEXT NOT NULL, generation INTEGER NOT NULL,
       PRIMARY KEY(owner,id))""",
    """CREATE TABLE IF NOT EXISTS memory_dreams (
       owner TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
       PRIMARY KEY(owner,id))""",
)


class Store:
    def __init__(self, db: Database) -> None:
        self.db = db
        with db.session_scope() as session:
            for ddl in DDL:
                session.execute(text(ddl))
            session.commit()

    @contextmanager
    def transaction(self, owner: str) -> Iterator[Session]:
        if not owner:
            raise ValueError("authenticated owner must be nonempty")
        with self.db.session_scope() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            session.execute(
                text("INSERT OR IGNORE INTO memory_owners(owner) VALUES (:owner)"),
                {"owner": owner},
            )
            for field in DEFAULT_FIELDS:
                session.execute(
                    text(
                        "INSERT OR IGNORE INTO memory_fields VALUES (:owner,:name,:definition)"
                    ),
                    {
                        "owner": owner,
                        "name": field.name,
                        "definition": field.model_dump_json(),
                    },
                )
            yield session
            session.commit()

    @contextmanager
    def read_transaction(
        self, owner: str, *, embedding_spec: str | None = None
    ) -> Iterator[Session]:
        if not owner:
            raise ValueError("authenticated owner must be nonempty")
        with self.db.session_scope() as session:
            # Python's SQLite legacy transaction mode does not begin on SELECT.
            session.execute(text("BEGIN"))
            row = session.execute(
                text("SELECT embedding_spec FROM memory_owners WHERE owner=:o"),
                {"o": owner},
            ).first()
            if row is None or (embedding_spec is not None and row[0] is None):
                session.rollback()
                with self.transaction(owner) as initializing:
                    if embedding_spec is not None:
                        initializing.execute(
                            text(
                                "UPDATE memory_owners SET embedding_spec=:s WHERE owner=:o AND embedding_spec IS NULL"
                            ),
                            {"s": embedding_spec, "o": owner},
                        )
                session.execute(text("BEGIN"))
            yield session

    @staticmethod
    def state(session: Session, owner: str) -> RowMapping:
        return (
            session.execute(
                text("SELECT * FROM memory_owners WHERE owner=:owner"), {"owner": owner}
            )
            .mappings()
            .one()
        )

    @staticmethod
    def fields(session: Session, owner: str) -> list[FieldDefinition]:
        return [
            FieldDefinition.model_validate_json(row[0])
            for row in session.execute(
                text(
                    "SELECT definition FROM memory_fields WHERE owner=:owner ORDER BY name"
                ),
                {"owner": owner},
            )
        ]

    @staticmethod
    def bump(
        session: Session,
        owner: str,
        action: str,
        detail: JsonValue,
        *,
        write: bool = True,
    ) -> int:
        session.execute(
            text(
                "UPDATE memory_owners SET generation=generation+1,writes=writes+:write WHERE owner=:owner"
            ),
            {"owner": owner, "write": int(write)},
        )
        generation = Store.state(session, owner)["generation"]
        session.execute(
            text(
                "INSERT INTO memory_audit(owner,generation,action,detail) VALUES (:o,:g,:a,:d)"
            ),
            {
                "o": owner,
                "g": generation,
                "a": action,
                "d": JSON_VALUE.dump_json(detail).decode(),
            },
        )
        return generation

    def forget(self, session: Session, owner: str, memory_id: str) -> bool:
        """Delete every stored copy of a record. Other caches drop it on their next sync."""
        if not self.contains(session, owner, memory_id):
            return False
        generation = self.bump(session, owner, "forget", {"id": memory_id}, write=False)
        params = {"o": owner, "id": memory_id}
        session.execute(
            text("DELETE FROM memories_v1 WHERE owner=:o AND id=:id"), params
        )
        session.execute(
            text("DELETE FROM memory_archives WHERE owner=:o AND id=:id"), params
        )
        # Dream plans embed copies of the memories they group.
        for table, compressed in (
            ("memory_dreams", False),
            ("memory_dream_archives", True),
        ):
            for plan_id, payload in session.execute(
                text(f"SELECT id,payload FROM {table} WHERE owner=:o"), {"o": owner}
            ).all():
                raw = zlib.decompress(payload) if compressed else payload
                if _embeds(DreamPlan.model_validate_json(raw), memory_id):
                    session.execute(
                        text(f"DELETE FROM {table} WHERE owner=:o AND id=:id"),
                        {"o": owner, "id": plan_id},
                    )
        # A saved HNSW graph holds the record's vector; it is rebuilt on next load.
        session.execute(
            text("DELETE FROM memory_vector_snapshots WHERE owner=:o"), {"o": owner}
        )
        session.execute(
            text("INSERT OR REPLACE INTO memory_forgotten VALUES (:o,:id,:g)"),
            {**params, "g": generation},
        )
        return True

    @staticmethod
    def contains(session: Session, owner: str, memory_id: str) -> bool:
        return (
            session.execute(
                text(
                    "SELECT 1 FROM memories_v1 WHERE owner=:o AND id=:id UNION ALL SELECT 1 FROM memory_archives WHERE owner=:o AND id=:id LIMIT 1"
                ),
                {"o": owner, "id": memory_id},
            ).first()
            is not None
        )

    @staticmethod
    def put(
        session: Session,
        owner: str,
        memory: Memory,
        embedding: bytes | None,
        generation: int,
    ) -> None:
        session.execute(
            text("""INSERT INTO memories_v1(owner,id,generation,payload,embedding)
            VALUES (:o,:id,:g,:p,:e) ON CONFLICT(owner,id) DO UPDATE SET
            generation=excluded.generation,payload=excluded.payload,embedding=excluded.embedding"""),
            {
                "o": owner,
                "id": memory.id,
                "g": generation,
                "p": memory.model_dump_json(),
                "e": embedding,
            },
        )


def _embeds(plan: DreamPlan, memory_id: str) -> bool:
    return any(
        group.memory.id == memory_id
        or any(hit.memory.id == memory_id for hit in group.neighbors)
        for group in plan.groups
    ) or any(change.memory.id == memory_id for change in plan.proposal.changes)
