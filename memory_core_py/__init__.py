"""Memory Core - A multi-layered memory system with vector indexing.

This package provides a comprehensive memory system with:
- Core abstractions (interfaces, models, system orchestrator)
- Storage layer (LTM, STM, working memory)
- Indexing layer (Rust-backed vector search, SMK assistant index)
- Type definitions (SMK types and features)
"""

# Core abstractions
from .config import MemoryCoreSettings, build_memory_system
from .core import (
    MemoryCandidate,
    MemoryIndex,
    MemorySystem,
    MemoryTrace,
)

# Indexing layer
from .indexing import (
    AssistantMemoryIndex,
    RustMemoryIndex,
)

# Storage layer
from .storage import (
    LongTermStore,
    RedisWorkingMemory,
    ShortTermStore,
)

# Type definitions
from .types import (
    Level2Bits,
    MemoryKind,
    ToolFlag,
    TopicBucket,
    build_smk_features,
)

__all__ = [
    # Indexing
    "AssistantMemoryIndex",
    # Types
    "Level2Bits",
    # Storage
    "LongTermStore",
    # Core
    "MemoryCandidate",
    # Config
    "MemoryCoreSettings",
    "MemoryIndex",
    "MemoryKind",
    "MemorySystem",
    "MemoryTrace",
    "RedisWorkingMemory",
    "RustMemoryIndex",
    "ShortTermStore",
    "ToolFlag",
    "TopicBucket",
    "build_memory_system",
    "build_smk_features",
]
