from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sqlalchemy.dialects.mysql import insert

from .database import LongTermTraceORM
from .settings import LongTermStoreSettings, build_engine_and_session

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.orm import Session

    from memory_core_py.core.models import MemoryTrace


class LongTermStore:
    """MariaDB-backed storage for consolidated, durable traces."""

    def __init__(
        self,
        *,
        settings: LongTermStoreSettings | None = None,
        **overrides: Any,
    ) -> None:
        base_settings = settings or LongTermStoreSettings()
        (
            self._settings,
            self._engine,
            self._session_factory,
        ) = build_engine_and_session(base_settings, overrides)

    def _session(self) -> Session:
        """Create a new SQLAlchemy session."""
        return self._session_factory()

    def upsert_trace(
        self,
        trace: MemoryTrace,
        embedding: Iterable[float] | None = None,
    ) -> None:
        """Persist the trace metadata and optional embedding vector."""
        tags_json = json.dumps(list(trace.tags))

        emb_bytes = None
        if embedding is not None:
            emb_bytes = json.dumps(list(embedding)).encode("utf-8")

        stmt = insert(LongTermTraceORM).values(
            trace_uid=trace.trace_uid,
            user_id=trace.user_id,
            summary=trace.summary,
            importance=trace.importance,
            created_at=trace.created_at,
            access_count=trace.access_count,
            tags=tags_json,
            embedding=emb_bytes,
        )

        stmt = stmt.on_duplicate_key_update(
            summary=stmt.inserted.summary,
            importance=stmt.inserted.importance,
            created_at=stmt.inserted.created_at,
            access_count=stmt.inserted.access_count,
            tags=stmt.inserted.tags,
            embedding=stmt.inserted.embedding,
        )

        session = self._session()
        try:
            session.execute(stmt)
            session.commit()
        finally:
            session.close()
