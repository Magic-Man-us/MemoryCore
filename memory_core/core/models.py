"""Legacy user-memory boundary models; typed 0.4 records live in memory.models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MemoryTrace(BaseModel):
    """User-facing long-term memory record persisted across sessions."""

    model_config = ConfigDict(frozen=False)

    trace_uid: str
    user_id: str
    content: str
    summary: str
    importance: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    access_count: int = 0
    tags: set[str] = Field(default_factory=set)
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_rust_args(self) -> tuple[str, str, str, float, int, int, list[str]]:
        """Serialize the trace into the argument order PyMemoryEngine expects.

        Returns:
            tuple: (trace_uid, user_id, summary, importance, created_at_ts, access_count, tags)
        """
        # Keeping this as an explicit tuple is clearer and safer than relying on
        # model_dump / as_tuple ordering, which can accidentally change.
        return (
            self.trace_uid,
            self.user_id,
            self.summary,
            float(self.importance),
            int(self.created_at.timestamp()),
            int(self.access_count),
            list(self.tags),
        )


class MemoryCandidate(BaseModel):
    """Search result candidate produced by the Rust memory index."""

    model_config = ConfigDict(frozen=True)

    trace_uid: str
    score: float
    summary: str
    tags: list[str]
    created_at: datetime
    importance: float = 0.0
    access_count: int = 0


class ScoringWeights(BaseModel):
    """Weights for the conversation-memory candidate score.

    ``score = semantic * cosine + importance * importance
    + recency * 0.5 ** (age / half_life) + access * a / (1 + a)``, with
    ``a = ln(1 + access_count)``. The defaults are the original
    ``0.7 * cosine + 0.3 * importance``: recency and access are opt-in.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    semantic: float = Field(default=0.7, ge=0.0, allow_inf_nan=False)
    importance: float = Field(default=0.3, ge=0.0, allow_inf_nan=False)
    recency: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    access: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    recency_half_life_days: float = Field(default=30.0, gt=0.0, allow_inf_nan=False)


class PurgeResult(BaseModel):
    """Rows removed by ``MemorySystem.purge_expired``."""

    stm_traces: int = 0
    wm_events: int = 0


class RecallResult(BaseModel):
    """Everything one recall returns, across the layers."""

    model_config = ConfigDict(frozen=False)

    ltm_candidates: list[MemoryCandidate] = Field(default_factory=list)
    wm_events: list[dict[str, Any]] = Field(default_factory=list)
    stm_traces: list[MemoryTrace] = Field(default_factory=list)
