"""The single SQL database behind all stores: engine, session factory, schema, ORM models.

``datetime`` is imported at runtime on purpose: SQLAlchemy resolves ``Mapped[...]``
annotations when the declarative classes are built, so type-only imports break the
whole storage layer at import time (see docs/review-junovera-integration.md, P0-1).

All datetimes are stored timezone-aware in UTC, and every comparison uses a
Python-supplied bound parameter (never ``func.now()``) so SQLite's string-typed
datetime comparisons stay consistent with what was written.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from datetime import datetime  # noqa: TC003 — runtime-required by Mapped[] resolution
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    LargeBinary,
    Text,
    create_engine,
    event,
    inspect,
    make_url,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from .settings import SQLITE_MEMORY, DatabaseSettings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy.engine import URL, Engine


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
    extra: Mapped[str | None] = mapped_column(Text, nullable=True)
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
    """Rolling per-user session events with TTL expiry (``wm_events``).

    ``trace_uid`` is set for events about a specific trace, so forgetting the trace
    can purge them too.
    """

    __tablename__ = "wm_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    expires_at: Mapped[datetime] = mapped_column(nullable=False, index=True)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    trace_uid: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)


class AssistantTraceORM(Base):
    """Assistant-level learning traces for the SMK index (``assistant_traces``).

    ``payload`` is the full ``AssistantMemoryTrace`` as JSON; ``smk_raw`` is the
    packed key, stored for inspection and ad-hoc SQL.
    """

    __tablename__ = "assistant_traces"

    trace_uid: Mapped[str] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    smk_raw: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)


def _is_sqlite_memory(url: URL) -> bool:
    database = url.database
    return (
        database in (None, "", SQLITE_MEMORY)
        or url.query.get("mode") == "memory"
    )


class Database:
    """One engine + session factory shared by every store, with schema bootstrap.

    SQLite specifics:

    - file databases get their parent directory created and ``~`` expanded, WAL
      journaling, and a 30 s busy timeout (a CLI and a daemon can share one file);
    - ``:memory:`` databases use a single shared connection (``StaticPool``) and
      serialize sessions, because store calls run in worker threads and the default
      pool would hand each thread its own empty database.
    """

    def __init__(
        self,
        *,
        settings: DatabaseSettings | None = None,
        **overrides: Any,
    ) -> None:
        self.settings = settings or DatabaseSettings.from_overrides(overrides)
        url = make_url(self.settings.resolved_url)
        engine_kwargs: dict[str, Any] = {}
        self._serial_lock: threading.RLock | None = None

        self.is_sqlite = url.get_backend_name() == "sqlite"
        self.is_memory = self.is_sqlite and _is_sqlite_memory(url)
        if self.is_sqlite:
            engine_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if self.is_memory:
                engine_kwargs["poolclass"] = StaticPool
                self._serial_lock = threading.RLock()
            else:
                path = Path(str(url.database)).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                url = url.set(database=str(path))

        self.url: URL = url
        self.engine: Engine = create_engine(url, **engine_kwargs)
        if self.is_sqlite and not self.is_memory:
            event.listen(self.engine, "connect", _sqlite_wal_pragma)
        self.session_factory: sessionmaker[Session] = sessionmaker(
            bind=self.engine, expire_on_commit=False
        )

    # Additive columns introduced after a table first shipped, applied by
    # create_schema so existing databases keep working without a migration
    # framework: (table, column, DDL type). ALTER TABLE ... ADD COLUMN is
    # portable across SQLite/PostgreSQL/MySQL for nullable columns.
    _MICRO_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
        ("ltm_traces", "extra", "TEXT"),
        ("wm_events", "trace_uid", "TEXT"),
    )

    def create_schema(self) -> None:
        """Create missing tables and add later-introduced columns. Idempotent."""
        with self._serialized():
            Base.metadata.create_all(self.engine)
            inspector = inspect(self.engine)
            tables = set(inspector.get_table_names())
            for table, column, ddl_type in self._MICRO_MIGRATIONS:
                if table not in tables:
                    continue
                existing = {col["name"] for col in inspector.get_columns(table)}
                if column not in existing:
                    with self.engine.begin() as connection:
                        connection.execute(
                            text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")
                        )

    def session(self) -> Session:
        """A new session; callers own commit/close. Prefer :meth:`session_scope`."""
        return self.session_factory()

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """A session that is rolled back on error and always closed.

        Commits stay explicit. For in-memory SQLite the scope also holds the
        connection lock, so concurrent worker threads never interleave
        transactions on the one shared connection.
        """
        with self._serialized():
            session = self.session_factory()
            try:
                yield session
            except BaseException:
                session.rollback()
                raise
            finally:
                session.close()

    def dispose(self) -> None:
        """Close pooled connections (an in-memory database is discarded)."""
        self.engine.dispose()

    def _serialized(self) -> threading.RLock | nullcontext[None]:
        return self._serial_lock if self._serial_lock is not None else nullcontext()


def _sqlite_wal_pragma(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
    finally:
        cursor.close()
