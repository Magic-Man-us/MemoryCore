"""The single SQL database behind all stores: engine, session factory, schema, ORM models.

``datetime`` is imported at runtime on purpose: SQLAlchemy resolves ``Mapped[...]``
annotations when the declarative classes are built, so type-only imports break the
whole storage layer at import time (see docs/review-junovera-integration.md, P0-1).

All datetimes are stored timezone-aware in UTC, and every comparison uses a
Python-supplied bound parameter (never ``func.now()``) so SQLite's string-typed
datetime comparisons stay consistent with what was written.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — runtime-required by Mapped[] resolution
from typing import Any

from sqlalchemy import LargeBinary, Text, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .settings import DatabaseSettings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class LongTermTraceORM(Base):
    """Durable, consolidated user memories (``ltm_traces``)."""

    __tablename__ = "ltm_traces"

    trace_uid: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(nullable=False, index=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    access_count: Mapped[int] = mapped_column(default=0, nullable=False)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)


class ShortTermTraceORM(Base):
    """Short-lived traces with explicit expiry (``stm_traces``)."""

    __tablename__ = "stm_traces"

    trace_uid: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[float] = mapped_column(nullable=False)
    access_count: Mapped[int] = mapped_column(default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    last_accessed: Mapped[datetime] = mapped_column(nullable=False)
    expires_at: Mapped[datetime] = mapped_column(nullable=False, index=True)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra: Mapped[str | None] = mapped_column(Text, nullable=True)


class WorkingMemoryEventORM(Base):
    """Rolling per-user session events with TTL expiry (``wm_events``)."""

    __tablename__ = "wm_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    expires_at: Mapped[datetime] = mapped_column(nullable=False, index=True)
    payload: Mapped[str] = mapped_column(Text, nullable=False)


class Database:
    """One engine + session factory shared by every store, with schema bootstrap."""

    def __init__(
        self,
        *,
        settings: DatabaseSettings | None = None,
        **overrides: Any,
    ) -> None:
        self.settings = settings or DatabaseSettings.from_overrides(overrides)
        engine_kwargs: dict[str, Any] = {}
        if self.settings.url.startswith("sqlite"):
            # Store calls run in worker threads (asyncio.to_thread); WAL keeps
            # concurrent reader/writer turns from blocking each other.
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        self.engine: Engine = create_engine(self.settings.url, **engine_kwargs)
        if self.settings.url.startswith("sqlite"):
            event.listen(self.engine, "connect", _sqlite_wal_pragma)
        self.session_factory: sessionmaker[Session] = sessionmaker(
            bind=self.engine, expire_on_commit=False
        )

    def create_schema(self) -> None:
        """Create all tables if missing. Idempotent."""
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        """A new session; callers own commit/close."""
        return self.session_factory()


def _sqlite_wal_pragma(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
    finally:
        cursor.close()
