
from __future__ import annotations

from enum import IntEnum, IntFlag


class TopicBucket(IntEnum):
    RUST_PYTHON_TOOLCHAIN = 1
    MEMORY_ARCHITECTURE = 2
    LOCAL_ENVIRONMENT = 3
    DB_SCHEMA = 4


class MemoryKind(IntEnum):
    INSIGHT = 0
    PATTERN = 1
    ANTI_PATTERN = 2
    PRINCIPLE = 3
    WORKFLOW = 4


class Level2Bits(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    EXTREME = 3


class ToolFlag(IntFlag):
    RS = 1 << 0
    PY = 1 << 1
    UV = 1 << 2
    MATURIN = 1 << 3
    CFN = 1 << 4

