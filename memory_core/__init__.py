"""Local typed agent memory with SQLite persistence and Rust vector search."""

from .config import MemoryCoreSettings, build_memory_system, resolve_settings
from .core import (
    Embedder,
    MemoryCandidate,
    MemoryIndex,
    MemorySystem,
    MemoryTrace,
    PurgeResult,
    RecallResult,
    ScoringWeights,
)
from .indexing import (
    RustMemoryIndex,
)
from .storage import (
    Database,
    DatabaseSettings,
    LongTermStore,
    ShortTermStore,
    SqlWorkingMemory,
    sqlite_url,
)

__all__ = [
    "Database",
    "DatabaseSettings",
    "Embedder",
    "LongTermStore",
    "MemoryCandidate",
    "MemoryCoreSettings",
    "MemoryIndex",
    "MemorySystem",
    "MemoryTrace",
    "PurgeResult",
    "RecallResult",
    "RustMemoryIndex",
    "ScoringWeights",
    "ShortTermStore",
    "SqlWorkingMemory",
    "build_memory_system",
    "resolve_settings",
    "sqlite_url",
]

from .memory import REQUEST_ADAPTER, RESPONSE_ADAPTER, EmbeddingSpec, MemoryService

__all__ += ["REQUEST_ADAPTER", "RESPONSE_ADAPTER", "EmbeddingSpec", "MemoryService"]
