"""Configuration helpers to wire up the MemorySystem from environment or kwargs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from memory_core.core.system import MemorySystem
from memory_core.indexing import AssistantMemoryIndex, RustMemoryIndex
from memory_core.storage import (
    Database,
    DatabaseSettings,
    LongTermStore,
    ShortTermStore,
    SqlWorkingMemory,
)

if TYPE_CHECKING:
    from memory_core.core.interfaces import Embedder


class MemoryCoreSettings(BaseSettings):
    """Top-level configuration for constructing a MemorySystem.

    Every field has a default, so ``MemoryCoreSettings()`` always succeeds — no
    environment variables are required to construct a local SQLite-backed system.
    """

    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    wm_ttl_seconds: int = Field(default=3600, ge=1)
    wm_max_events: int = Field(default=500, ge=1)
    stm_ttl_seconds: int = Field(default=900, ge=1)
    enable_stm: bool = False
    enable_assistant_index: bool = False
    assistant_index_dim: int | None = Field(
        default=None,
        ge=1,
        description="Vector dimension for the assistant index; required when enabled.",
    )

    model_config = SettingsConfigDict(env_prefix="MEMORY_CORE_", extra="ignore")


def build_memory_system(
    settings: MemoryCoreSettings | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    embedder: Embedder | None = None,
    create_schema: bool = True,
) -> MemorySystem:
    """Create a fully-wired MemorySystem from settings and optional overrides.

    Args:
        settings: Optional pre-built settings instance (uses environment by default).
        overrides: Optional overrides; keys may include:
            - ``db``: dict of DatabaseSettings overrides (e.g. ``{"url": "sqlite:///m.db"}``)
            - ``wm_ttl_seconds`` / ``wm_max_events``: working-memory policy
            - ``stm_ttl_seconds``: integer TTL for STM entries
            - ``enable_stm`` / ``enable_assistant_index``: feature toggles
            - ``assistant_index_dim``: required when enabling the assistant index
        embedder: Optional Embedder; when set, remember/recall accept raw text.
        create_schema: Create missing tables on the target database (idempotent).

    Returns:
        An initialized MemorySystem wired to one SQL database (SQLite by default).
    """
    overrides = overrides or {}
    base_settings = settings or MemoryCoreSettings()

    db_settings = base_settings.db
    db_overrides = overrides.get("db")
    if db_overrides:
        cleaned = {k: v for k, v in dict(db_overrides).items() if v is not None}
        db_settings = DatabaseSettings(**{**db_settings.model_dump(), **cleaned})
    wm_ttl_seconds = overrides.get("wm_ttl_seconds", base_settings.wm_ttl_seconds)
    wm_max_events = overrides.get("wm_max_events", base_settings.wm_max_events)
    stm_ttl_seconds = overrides.get("stm_ttl_seconds", base_settings.stm_ttl_seconds)
    enable_stm = overrides.get("enable_stm", base_settings.enable_stm)
    enable_assistant_index = overrides.get(
        "enable_assistant_index", base_settings.enable_assistant_index
    )
    assistant_index_dim = overrides.get(
        "assistant_index_dim", base_settings.assistant_index_dim
    )

    database = Database(settings=db_settings)
    if create_schema:
        database.create_schema()

    ltm_store = LongTermStore(database)
    stm_store = ShortTermStore(database) if enable_stm else None
    working_mem = SqlWorkingMemory(
        database, ttl_seconds=wm_ttl_seconds, max_events=wm_max_events
    )
    memory_index = RustMemoryIndex()

    assistant_index = None
    if enable_assistant_index:
        if assistant_index_dim is None:
            msg = "assistant_index_dim must be provided when enable_assistant_index is True"
            raise ValueError(msg)
        assistant_index = AssistantMemoryIndex(dim=assistant_index_dim)

    return MemorySystem(
        memory_index=memory_index,
        working_mem=working_mem,
        ltm_store=ltm_store,
        embedder=embedder,
        assistant_index=assistant_index,
        stm_store=stm_store,
        stm_ttl_seconds=stm_ttl_seconds,
    )
