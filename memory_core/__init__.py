"""Local typed agent memory with SQLite persistence and Rust vector search."""

from ._native import PyVectorIndex as RustVectorIndex
from .indexing.semantic import IndexPolicy
from .memory import REQUEST_ADAPTER, RESPONSE_ADAPTER, EmbeddingSpec, MemoryService
from .memory.interfaces import Embedder
from .runtime import MemoryCore, RetentionPolicy, RetentionResult
from .storage import Database

__all__ = [
    "REQUEST_ADAPTER",
    "RESPONSE_ADAPTER",
    "Database",
    "Embedder",
    "EmbeddingSpec",
    "IndexPolicy",
    "MemoryCore",
    "MemoryService",
    "RetentionPolicy",
    "RetentionResult",
    "RustVectorIndex",
]
