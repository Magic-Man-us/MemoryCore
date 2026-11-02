"""Local SQLite durable state and atomic revisions using the existing Database connection.

Writes use BEGIN IMMEDIATE for reconciliation versions. Reads use a deferred
snapshot after one-time owner initialization.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

from sqlalchemy import text

from .metadata import DEFAULT_FIELDS
from .models import FieldDefinition

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

    @contextmanager
    def read_transaction(self, owner: str, *, embedding_spec: str | None = None):
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
                text(
                    "SELECT 1 FROM memories_v1 WHERE owner=:o AND id=:id UNION ALL SELECT 1 FROM memory_archives WHERE owner=:o AND id=:id LIMIT 1"
                ),
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
