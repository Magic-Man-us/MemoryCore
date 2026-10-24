"""SMK enums. Python is the single source of truth for topic and tool meanings.

The Rust index treats topic as an opaque u8 (1..=255; 0 is reserved) and tools as a
16-bit mask, so growing either taxonomy is a Python-only change. ``MemoryKind`` and
``Level2Bits`` are fixed by the key layout and validated on both sides.
"""

from __future__ import annotations

from enum import IntEnum, IntFlag


class TopicBucket(IntEnum):
    """Topic of an assistant memory (SMK bits 0-7). Add members freely, values 1..=255."""

    RUST_PYTHON_TOOLCHAIN = 1
    MEMORY_ARCHITECTURE = 2
    LOCAL_ENVIRONMENT = 3
    DB_SCHEMA = 4


class MemoryKind(IntEnum):
    """What kind of lesson a memory is (SMK bits 8-10)."""

    INSIGHT = 0
    PATTERN = 1
    ANTI_PATTERN = 2
    PRINCIPLE = 3
    WORKFLOW = 4


class Level2Bits(IntEnum):
    """Two-bit quantized level for difficulty, generality, and importance."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    EXTREME = 3


class ToolFlag(IntFlag):
    """Tools a memory involves (SMK bits 11-26). Up to 16 flags."""

    RS = 1 << 0
    PY = 1 << 1
    UV = 1 << 2
    MATURIN = 1 << 3
    CFN = 1 << 4
