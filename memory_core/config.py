"""Configuration helpers to wire up the MemorySystem from environment or kwargs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from memory_core.core.models import ScoringWeights
from memory_core.core.system import MemorySystem
from memory_core.indexing import RustMemoryIndex
from memory_core.storage import (
    Database,
    DatabaseSettings,
    LongTermStore,
    ShortTermStore,
    SqlWorkingMemory,
)

if TYPE_CHECKING:
    from memory_core.core.interfaces import Embedder
    from memory_core.memory.interfaces import DreamPlanner, Reconciler
    from memory_core.memory.models import EmbeddingSpec, RetrievalPolicy


class MemoryCoreSettings(BaseSettings):
    """Top-level configuration for constructing a MemorySystem.

    Every field has a default, so ``MemoryCoreSettings()`` always succeeds: no
    environment variables are required to construct a local SQLite-backed system.
    Nested fields read from ``MEMORY_CORE_<FIELD>__<SUBFIELD>`` (e.g.
    ``MEMORY_CORE_SCORING__RECENCY=0.1``); the database also reads ``MEMORY_DB_*``.
    """

    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    wm_ttl_seconds: int = Field(default=3600, ge=1)
    wm_max_events: int = Field(default=500, ge=1)
    stm_ttl_seconds: int = Field(default=900, ge=1)
    enable_stm: bool = False
    scoring: ScoringWeights = Field(default_factory=ScoringWeights)

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_CORE_", env_nested_delimiter="__", extra="ignore"
    )

    def require_complete(self) -> MemoryCoreSettings:
        """Return the validated settings (kept for source compatibility)."""
        return self


_NESTED_MODELS: dict[str, type[BaseModel]] = {
    "db": DatabaseSettings,
    "scoring": ScoringWeights,
}


def resolve_settings(
    settings: MemoryCoreSettings | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    sqlite_path: str | os.PathLike[str] | None = None,
) -> MemoryCoreSettings:
    """Merge explicit overrides onto settings (env by default) and validate the result.

    Unknown keys raise instead of being silently ignored. Nested ``db``/``scoring``
    overrides merge field-by-field; a ``None`` value means "keep the base value".
    ``sqlite_path`` is shorthand for ``overrides={"db": {"sqlite_path": ...}}``.
    """
    base = settings or MemoryCoreSettings()
    overrides = dict(overrides or {})

    unknown = set(overrides) - set(MemoryCoreSettings.model_fields)
    if unknown:
        raise ValueError(f"unknown override keys: {sorted(unknown)}")

    if sqlite_path is not None:
        db_overrides = dict(overrides.get("db") or {})
        db_overrides["sqlite_path"] = os.fspath(sqlite_path)
        overrides["db"] = db_overrides

    data = base.model_dump()
    for key, value in overrides.items():
        model = _NESTED_MODELS.get(key)
        if model is None:
            data[key] = value
            continue
        if value is None:
            continue
        nested = {k: v for k, v in dict(value).items() if v is not None}
        unknown_nested = set(nested) - set(model.model_fields)
        if unknown_nested:
            raise ValueError(f"unknown {key} override keys: {sorted(unknown_nested)}")
        if key == "db" and "url" in nested and "sqlite_path" not in nested:
            # an explicit URL beats an env-provided sqlite_path
            nested["sqlite_path"] = None
        data[key] = {**data[key], **nested}

    return MemoryCoreSettings.model_validate(data).require_complete()


def build_memory_system(
    settings: MemoryCoreSettings | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    sqlite_path: str | os.PathLike[str] | None = None,
    embedder: Embedder | None = None,
    create_schema: bool = True,
    embedding_spec: EmbeddingSpec | None = None,
    reconciler: Reconciler | None = None,
    dream_planner: DreamPlanner | None = None,
    dream_after_writes: int = 100,
    retrieval_policy: RetrievalPolicy | None = None,
    max_cached_owners: int = 16,
) -> MemorySystem:
    """Create a fully-wired MemorySystem from settings and optional overrides.

    Args:
        settings: Optional pre-built settings instance (uses environment by default).
        overrides: Optional overrides; keys are ``MemoryCoreSettings`` fields:
            - ``db``: DatabaseSettings fields (``url`` or ``sqlite_path``)
            - ``wm_ttl_seconds`` / ``wm_max_events``: working-memory policy
            - ``stm_ttl_seconds``: integer TTL for STM entries
            - ``enable_stm``: enable the legacy short-term layer
            - ``scoring``: ScoringWeights fields (``recency``, ``access``, ...)
        sqlite_path: Use a local SQLite file at this path (``~`` expanded, parent
            directories created), or ``":memory:"``. Wins over any URL.
        embedder: Optional Embedder; when set, remember/recall accept raw text.
        create_schema: Create missing tables on the target database (idempotent).

    Returns:
        An initialized MemorySystem wired to one SQL database (SQLite by default).
        Typed operations hydrate automatically when embedding_spec is supplied.
        The legacy user API still uses explicit ``hydrate(user_id)``.
    """
    resolved = resolve_settings(settings, overrides=overrides, sqlite_path=sqlite_path)

    database = Database(settings=resolved.db)
    if create_schema:
        database.create_schema()

    system = MemorySystem(
        memory_index=RustMemoryIndex(weights=resolved.scoring),
        working_mem=SqlWorkingMemory(
            database,
            ttl_seconds=resolved.wm_ttl_seconds,
            max_events=resolved.wm_max_events,
        ),
        ltm_store=LongTermStore(database),
        embedder=embedder,
        stm_store=ShortTermStore(database) if resolved.enable_stm else None,
        stm_ttl_seconds=resolved.stm_ttl_seconds,
    )

    if embedding_spec is not None:
        from memory_core.memory.models import EmbeddingSpec
        from memory_core.memory.service import MemoryService

        system.service = MemoryService(
            database,
            embedding_spec=EmbeddingSpec.model_validate(embedding_spec),
            embedder=embedder,
            reconciler=reconciler,
            dream_planner=dream_planner,
            dream_after_writes=dream_after_writes,
            retrieval_policy=retrieval_policy,
            max_cached_owners=max_cached_owners,
        )
    return system
