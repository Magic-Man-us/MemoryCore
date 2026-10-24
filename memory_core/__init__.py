"""Memory Core: a multi-layered memory system with Rust-backed vector indexing.

One SQL database (a local SQLite file by default) persists long-term, short-term,
working, and assistant memory; all runtime queries hit the in-RAM Rust indexes
(``memory_core._native``).
"""

from .config import MemoryCoreSettings, build_memory_system, resolve_settings
from .core import (
    AssistantMemoryTrace,
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
    AssistantMemoryIndex,
    RustMemoryIndex,
)
from .storage import (
    AssistantStore,
    Database,
    DatabaseSettings,
    LongTermStore,
    ShortTermStore,
    SqlWorkingMemory,
    sqlite_url,
)
from .types import (
    Level2Bits,
    MemoryKind,
    ToolFlag,
    TopicBucket,
    build_smk_features,
)

__all__ = [
    "AssistantMemoryIndex",
    "AssistantMemoryTrace",
    "AssistantStore",
    "Database",
    "DatabaseSettings",
    "Embedder",
    "Level2Bits",
    "LongTermStore",
    "MemoryCandidate",
    "MemoryCoreSettings",
    "MemoryIndex",
    "MemoryKind",
    "MemorySystem",
    "MemoryTrace",
    "PurgeResult",
    "RecallResult",
    "RustMemoryIndex",
    "ScoringWeights",
    "ShortTermStore",
    "SqlWorkingMemory",
    "ToolFlag",
    "TopicBucket",
    "build_memory_system",
    "build_smk_features",
    "resolve_settings",
    "sqlite_url",
]
