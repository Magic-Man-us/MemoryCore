"""Typed settings objects for storage backends."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Self

from pydantic import Discriminator, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.engine import Engine

MYSQL_CONNECTION_KEYS = {"host", "user", "password", "database", "port"}
REDIS_CONNECTION_KEYS = {"host", "port", "database", "password", "ttl_seconds", "url"}


def _filter_overrides(
    overrides: Mapping[str, Any] | None,
    allowed_keys: set[str],
) -> dict[str, Any]:
    """Allow only known override keys and drop None values."""
    if not overrides:
        return {}

    return {
        key: value
        for key, value in overrides.items()
        if key in allowed_keys and value is not None
    }


class MemoryType(str, Enum):
    """Memory storage type enumeration."""

    STM = "stm"
    LTM = "ltm"
    REDIS = "redis"


class MySQLStoreSettings(BaseSettings):
    """Connection settings for MySQL-backed stores."""

    memory_type: MemoryType
    host: str
    user: str
    password: str
    database: str
    port: int = Field(default=3306, ge=1)

    model_config = SettingsConfigDict(extra="ignore")

    @computed_field
    @property
    def dsn(self) -> str:
        """Return the SQLAlchemy DSN including the mysqlconnector driver."""
        return (
            f"mysql+mysqlconnector://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    def apply_overrides(self, overrides: Mapping[str, Any] | None) -> Self:
        """Return a settings copy with sanitized overrides applied."""
        updates = _filter_overrides(overrides, MYSQL_CONNECTION_KEYS)
        return self if not updates else self.model_copy(update=updates)


class LongTermStoreSettings(MySQLStoreSettings):
    """Typed DSN for the long-term store, sourced from env or kwargs."""

    memory_type: MemoryType = Field(default=MemoryType.LTM)
    model_config = SettingsConfigDict(env_prefix="LTM_DB_", extra="ignore")


class ShortTermStoreSettings(MySQLStoreSettings):
    """Connection settings for the STM store, pulled from env when desired."""

    memory_type: MemoryType = Field(default=MemoryType.STM)
    model_config = SettingsConfigDict(env_prefix="STM_DB_", extra="ignore")


class RedisWorkingMemorySettings(BaseSettings):
    """Settings for Redis working memory pulled from env or kwargs."""

    memory_type: MemoryType = Field(default=MemoryType.REDIS)
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1)
    database: int = Field(default=0, ge=0)
    password: str | None = None
    ttl_seconds: int = Field(default=900, ge=1)
    url: str | None = Field(
        default=None,
        description="Optional full Redis URL override; overrides host/port/database.",
    )

    model_config = SettingsConfigDict(env_prefix="REDIS_DB_", extra="ignore")

    @computed_field
    @property
    def dsn(self) -> str:
        """Return a Redis URL, respecting a supplied URL override when present."""
        if self.url:
            return self.url

        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"

    def apply_overrides(self, overrides: Mapping[str, Any] | None) -> Self:
        """Return a settings copy with sanitized overrides applied."""
        updates = _filter_overrides(overrides, REDIS_CONNECTION_KEYS)
        return self if not updates else self.model_copy(update=updates)


def build_engine_and_session(
    settings: MySQLStoreSettings,
    overrides: Mapping[str, Any] | None = None,
    **engine_kwargs: Any,
) -> tuple[MySQLStoreSettings, Engine, sessionmaker[Session]]:
    """Instantiate a SQLAlchemy engine and session factory from settings."""
    resolved_settings = settings.apply_overrides(overrides)
    engine = create_engine(resolved_settings.dsn, **engine_kwargs)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    return resolved_settings, engine, session_factory


MemorySettings = Annotated[
    LongTermStoreSettings | ShortTermStoreSettings | RedisWorkingMemorySettings,
    Discriminator("memory_type"),
]

__all__ = [
    "LongTermStoreSettings",
    "MemorySettings",
    "MemoryType",
    "MySQLStoreSettings",
    "RedisWorkingMemorySettings",
    "ShortTermStoreSettings",
    "build_engine_and_session",
]
