"""Configuration helpers to wire up the MemorySystem from environment or kwargs."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from memory_core_py.core.system import MemorySystem
from memory_core_py.indexing import AssistantMemoryIndex, RustMemoryIndex
from memory_core_py.storage import (
    LongTermStore,
    LongTermStoreSettings,
    RedisWorkingMemory,
    RedisWorkingMemorySettings,
    ShortTermStore,
    ShortTermStoreSettings,
)


class MemoryCoreSettings(BaseSettings):
    """Top-level configuration for constructing a MemorySystem."""

    stm_ttl_seconds: int = Field(default=900, ge=1)
    enable_stm: bool = True
    enable_assistant_index: bool = False
    assistant_index_dim: int | None = Field(
        default=None,
        ge=1,
        description="Vector dimension for the assistant index; required when enabled.",
    )
    ltm: LongTermStoreSettings = Field(default_factory=lambda: LongTermStoreSettings())
    stm: ShortTermStoreSettings = Field(default_factory=lambda: ShortTermStoreSettings())
    working_mem: RedisWorkingMemorySettings = Field(
        default_factory=lambda: RedisWorkingMemorySettings()
    )

    model_config = SettingsConfigDict(env_prefix="MEMORY_CORE_", extra="ignore")


def build_memory_system(
    settings: MemoryCoreSettings | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> MemorySystem:
    """Create a fully-wired MemorySystem from settings and optional overrides.

    Args:
        settings: Optional pre-built settings instance (uses environment by default).
        overrides: Optional per-component overrides; keys may include:
            - ``ltm``: dict for LongTermStoreSettings overrides
            - ``stm``: dict for ShortTermStoreSettings overrides
            - ``working_mem``: dict for RedisWorkingMemorySettings overrides
            - ``stm_ttl_seconds``: integer TTL for STM entries
            - ``enable_stm`` / ``enable_assistant_index``: feature toggles
            - ``assistant_index_dim``: required when enabling assistant index

    Returns:
        An initialized MemorySystem instance wired with storage and indexes.
    """
    overrides = overrides or {}
    base_settings = settings or MemoryCoreSettings()

    ltm_settings = _maybe_update_settings(base_settings.ltm, overrides.get("ltm"))
    stm_settings = _maybe_update_settings(base_settings.stm, overrides.get("stm"))
    wm_settings = _maybe_update_settings(
        base_settings.working_mem,
        overrides.get("working_mem"),
    )

    enable_stm = overrides.get("enable_stm", base_settings.enable_stm)
    stm_ttl_seconds = overrides.get(
        "stm_ttl_seconds",
        base_settings.stm_ttl_seconds,
    )
    enable_assistant_index = overrides.get(
        "enable_assistant_index",
        base_settings.enable_assistant_index,
    )
    assistant_index_dim = overrides.get(
        "assistant_index_dim",
        base_settings.assistant_index_dim,
    )

    ltm_store = LongTermStore(settings=ltm_settings)
    stm_store = ShortTermStore(settings=stm_settings) if enable_stm else None
    working_mem = RedisWorkingMemory(settings=wm_settings)
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
        assistant_index=assistant_index,
        stm_store=stm_store,
        stm_ttl_seconds=stm_ttl_seconds,
    )


TSettings = TypeVar("TSettings", bound=BaseSettings)


def _maybe_update_settings(
    settings_obj: TSettings,
    updates: dict[str, Any] | None,
) -> TSettings:
    """Safely update a pydantic BaseSettings object with provided values."""
    if not updates:
        return settings_obj
    return settings_obj.model_copy(update=updates)
