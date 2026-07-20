"""Core abstractions: protocols, trace/candidate models, and the MemorySystem orchestrator."""

from .interfaces import Embedder, MemoryIndex
from .models import (
    AssistantMemoryTrace,
    MemoryCandidate,
    MemoryTrace,
    RecallResult,
)
from .system import MemorySystem

__all__ = [
    "AssistantMemoryTrace",
    "Embedder",
    "MemoryCandidate",
    "MemoryIndex",
    "MemorySystem",
    "MemoryTrace",
    "RecallResult",
]
