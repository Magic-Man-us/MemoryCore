"""Type definitions and feature extraction for SMK memory system.

This module contains type definitions, enums, and feature extraction utilities
for the SMK (Semantic Memory Kernel) assistant memory system.
"""

from .smk_features import build_smk_features
from .smk_types import Level2Bits, MemoryKind, ToolFlag, TopicBucket

__all__ = [
    "Level2Bits",
    "MemoryKind",
    "ToolFlag",
    "TopicBucket",
    "build_smk_features",
]
