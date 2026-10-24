"""Assistant store: durable SMK assistant traces + embeddings in the shared SQL database.

The SMK index is RAM-only; this table is what lets learned patterns survive a
restart (``MemorySystem.hydrate_assistant``). Assistant traces are assistant-level,
not per-user, so hydration loads them all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, select

from memory_core.core.models import AssistantMemoryTrace

from .codec import decode_embedding, encode_embedding
from .database import AssistantTraceORM, Database

if TYPE_CHECKING:
    from collections.abc import Iterable


class AssistantStore:
    """SQL-backed storage for ``AssistantMemoryTrace`` records."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert_trace(
        self,
        trace: AssistantMemoryTrace,
        embedding: Iterable[float] | None = None,
    ) -> None:
        """Persist the full trace (as JSON) and its optional embedding."""
        row = AssistantTraceORM(
            trace_uid=trace.trace_uid,
            created_at=trace.created_at,
            smk_raw=trace.smk_raw,
            payload=trace.model_dump_json(),
            embedding=encode_embedding(embedding),
        )
        with self._db.session_scope() as session:
            session.merge(row)
            session.commit()

    def fetch_all(self) -> list[tuple[AssistantMemoryTrace, list[float] | None]]:
        """Every assistant trace with its embedding, oldest first."""
        stmt = select(AssistantTraceORM).order_by(AssistantTraceORM.created_at)
        with self._db.session_scope() as session:
            rows = session.execute(stmt).scalars().all()
            return [
                (
                    AssistantMemoryTrace.model_validate_json(row.payload),
                    decode_embedding(row.embedding),
                )
                for row in rows
            ]

    def delete_trace(self, trace_uid: str) -> bool:
        """Remove a trace permanently. Returns True when a row was deleted."""
        with self._db.session_scope() as session:
            result = session.execute(
                delete(AssistantTraceORM).where(AssistantTraceORM.trace_uid == trace_uid)
            )
            session.commit()
            return bool(result.rowcount)
