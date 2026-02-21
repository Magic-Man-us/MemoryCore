from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .smk_types import Level2Bits, MemoryKind, TopicBucket

if TYPE_CHECKING:
    from memory_core_py.core.models import AssistantMemoryTrace

# Quantization thresholds
LOW_THRESHOLD: float = 0.25
MEDIUM_THRESHOLD: float = 0.5
HIGH_THRESHOLD: float = 0.75


def _quantize_level(x: float) -> Level2Bits:
    """Quantize a float in [0, 1] into a Level2Bits bucket."""
    if x < LOW_THRESHOLD:
        return Level2Bits.LOW
    if x < MEDIUM_THRESHOLD:
        return Level2Bits.MEDIUM
    if x < HIGH_THRESHOLD:
        return Level2Bits.HIGH
    return Level2Bits.EXTREME


@dataclass
class SmkFeatures:
    topic: TopicBucket
    kind: MemoryKind
    tool_mask: int
    difficulty: Level2Bits
    generality: Level2Bits
    importance: Level2Bits


def build_smk_features(trace: AssistantMemoryTrace) -> SmkFeatures:
    tool_mask = 0
    for t in trace.tools:
        tool_mask |= int(t)

    # pick which scalar drives "difficulty"
    difficulty = _quantize_level(trace.before_state_confusion)
    generality = _quantize_level(trace.generality)
    importance = _quantize_level(trace.importance)

    return SmkFeatures(
        topic=trace.context_topic,
        kind=trace.kind,
        tool_mask=tool_mask,
        difficulty=difficulty,
        generality=generality,
        importance=importance,
    )
