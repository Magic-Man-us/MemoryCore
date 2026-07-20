"""Typed settings for the SQL storage backend.

One database, one URL. SQLite by default — a personal-assistant install needs no
services — but any SQLAlchemy URL works (PostgreSQL via the ``postgres`` extra).
Explicit keyword arguments always work without environment variables: overrides are
passed into construction (init kwargs beat env in pydantic-settings), never applied
after an env-only construction that could fail on missing variables.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DB_URL = "sqlite:///memory_core.db"

DATABASE_CONNECTION_KEYS = {"url"}


def filter_overrides(overrides: dict[str, Any] | None, allowed: set[str]) -> dict[str, Any]:
    """Keep only known override keys with non-None values."""
    if not overrides:
        return {}
    return {k: v for k, v in overrides.items() if k in allowed and v is not None}


class DatabaseSettings(BaseSettings):
    """Connection settings for the single SQL database backing all stores."""

    url: str = Field(
        default=DEFAULT_DB_URL,
        description="SQLAlchemy database URL. Defaults to a local SQLite file.",
    )

    model_config = SettingsConfigDict(env_prefix="MEMORY_DB_", extra="ignore")

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any] | None = None) -> DatabaseSettings:
        """Construct from env with explicit overrides taking precedence."""
        return cls(**filter_overrides(overrides, DATABASE_CONNECTION_KEYS))
