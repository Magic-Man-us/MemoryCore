"""Local SQLite durable state and atomic revisions using the existing Database connection.

All transactions use BEGIN IMMEDIATE: separate processes serialize before reading
reconciliation versions. Payloads are decoded only on hydration/change or selected reads.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

from sqlalchemy import text

from .metadata import DEFAULT_FIELDS
from .models import FieldDefinition

DDL = (
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
    """CREATE TABLE IF NOT EXISTS memory_dreams (
       owner TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
       PRIMARY KEY(owner,id))""",
    """CREATE TABLE IF NOT EXISTS memory_imports (
       owner TEXT NOT NULL, source TEXT NOT NULL, id TEXT NOT NULL,
       PRIMARY KEY(owner,source,id))""",
)


class Store:
    def __init__(self, db):
        if not db.is_sqlite:
            raise ValueError(
                "the extensible memory service currently requires local SQLite"
            )
        self.db = db
        with db.session_scope() as session:
            for ddl in DDL:
                session.execute(text(ddl))
            session.commit()

    @contextmanager
    def transaction(self, owner: str):
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

    @staticmethod
    def state(session, owner):
        return (
            session.execute(
                text("SELECT * FROM memory_owners WHERE owner=:owner"), {"owner": owner}
            )
            .mappings()
            .one()
        )

    @staticmethod
    def fields(session, owner):
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
    def bump(session, owner, action, detail, *, write=True):
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
            {"o": owner, "g": generation, "a": action, "d": json.dumps(detail)},
        )
        return generation

    @staticmethod
    def contains(session, owner, memory_id):
        return (
            session.execute(
                text("SELECT 1 FROM memories_v1 WHERE owner=:o AND id=:id"),
                {"o": owner, "id": memory_id},
            ).first()
            is not None
        )

    @staticmethod
    def put(session, owner, memory, embedding, generation):
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
