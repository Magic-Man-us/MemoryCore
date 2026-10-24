"""Typed settings for the SQL storage backend.

One database, one URL. A local SQLite file by default (a personal-assistant install
needs no services), but any SQLAlchemy URL works (PostgreSQL via the ``postgres``
extra). ``sqlite_path`` is the shortcut for "use a local SQLite file here": when set,
it wins over ``url``. ``":memory:"`` gives a process-local in-memory database.

Explicit keyword arguments always work without environment variables: overrides are
passed into construction (init kwargs beat env in pydantic-settings), never applied
after an env-only construction that could fail on missing variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DB_URL = "sqlite:///memory_core.db"
SQLITE_MEMORY = ":memory:"

DATABASE_CONNECTION_KEYS = {"url", "sqlite_path"}


def filter_overrides(overrides: dict[str, Any] | None, allowed: set[str]) -> dict[str, Any]:
    """Keep only known override keys with non-None values."""
    if not overrides:
        return {}
    return {k: v for k, v in overrides.items() if k in allowed and v is not None}


def sqlite_url(path: str | Path) -> str:
    """SQLAlchemy URL for a local SQLite file (``~`` expanded) or ``":memory:"``."""
    if str(path) == SQLITE_MEMORY:
        return f"sqlite:///{SQLITE_MEMORY}"
    return f"sqlite:///{Path(path).expanduser().as_posix()}"


class DatabaseSettings(BaseSettings):
    """Connection settings for the single SQL database backing all stores."""

    url: str = Field(
        default=DEFAULT_DB_URL,
        description="SQLAlchemy database URL. Defaults to a local SQLite file.",
    )
    sqlite_path: str | None = Field(
        default=None,
        description=(
            "Local SQLite file path (or ':memory:'). When set, overrides `url`; "
            "parent directories are created on connect."
        ),
    )

    model_config = SettingsConfigDict(env_prefix="MEMORY_DB_", extra="ignore")

    @property
    def resolved_url(self) -> str:
        """The URL actually connected to: ``sqlite_path`` when set, else ``url``."""
        if self.sqlite_path is not None:
            return sqlite_url(self.sqlite_path)
        return self.url

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any] | None = None) -> DatabaseSettings:
        """Construct from env with explicit overrides taking precedence.

        An explicit ``url`` with no ``sqlite_path`` clears any env-provided
        ``MEMORY_DB_SQLITE_PATH``, so the most specific *explicit* choice wins.
        """
        kwargs = filter_overrides(overrides, DATABASE_CONNECTION_KEYS)
        if "url" in kwargs and "sqlite_path" not in kwargs:
            kwargs["sqlite_path"] = None
        return cls(**kwargs)
