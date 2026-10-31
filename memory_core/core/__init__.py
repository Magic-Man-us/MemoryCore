"""Core abstractions: protocols, trace/candidate models, and the MemorySystem orchestrator."""

from .interfaces import Embedder, MemoryIndex
from .models import (
    MemoryCandidate,
    MemoryTrace,
    PurgeResult,
    RecallResult,
    ScoringWeights,
)
from .system import MemorySystem

__all__ = [
    "Embedder",
    "MemoryCandidate",
    "MemoryIndex",
    "MemorySystem",
    "MemoryTrace",
    "PurgeResult",
    "RecallResult",
    "ScoringWeights",
]
