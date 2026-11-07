"""The SQLite engine and session factory behind every store."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy.engine import Engine

SQLITE_MEMORY = ":memory:"
BUSY_TIMEOUT_SECONDS = 30


class Database:
    """One SQLite database shared by every store.

    A file path gets ``~`` expanded, its parent directory created, WAL journaling, and a
    busy timeout, so a CLI and a daemon can share one file. ``":memory:"`` uses a single
    shared connection (``StaticPool``) and serializes sessions: store calls run in worker
    threads, and the default pool would hand each thread its own empty database.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        connect_args: dict[str, Any] = {
            "check_same_thread": False,
            "timeout": BUSY_TIMEOUT_SECONDS,
        }
        self.is_memory = os.fspath(path) == SQLITE_MEMORY
        self._serial_lock: threading.RLock | None = None
        if self.is_memory:
            self._serial_lock = threading.RLock()
            self.engine: Engine = create_engine(
                f"sqlite:///{SQLITE_MEMORY}",
                connect_args=connect_args,
                poolclass=StaticPool,
            )
        else:
            file = Path(path).expanduser()
            file.parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(
                f"sqlite:///{file.as_posix()}", connect_args=connect_args
            )
            event.listen(self.engine, "connect", _sqlite_wal_pragma)
        self.session_factory: sessionmaker[Session] = sessionmaker(
            bind=self.engine, expire_on_commit=False
        )

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
