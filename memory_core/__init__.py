"""Memory Core — a multi-layered memory system with Rust-backed vector indexing.

One SQL database (SQLite by default) persists long-term, short-term, and working
memory; all runtime queries hit the in-RAM Rust index (``memory_core._native``).
"""

from .config import MemoryCoreSettings, build_memory_system
from .core import (
    AssistantMemoryTrace,
    Embedder,
    MemoryCandidate,
    MemoryIndex,
    MemorySystem,
    MemoryTrace,
    RecallResult,
)
from .indexing import (
    AssistantMemoryIndex,
    RustMemoryIndex,
)
from .storage import (
    Database,
    DatabaseSettings,
    LongTermStore,
    ShortTermStore,
    SqlWorkingMemory,
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
    "RecallResult",
    "RustMemoryIndex",
    "ShortTermStore",
    "SqlWorkingMemory",
    "ToolFlag",
    "TopicBucket",
    "build_memory_system",
    "build_smk_features",
]
