"""SQLAlchemy ORM models for memory storage tables."""

from __future__ import annotations

# NOTE: this must be a runtime import, not TYPE_CHECKING — SQLAlchemy resolves
# ``Mapped[datetime]`` annotations when the class is declared, and hiding the
# import breaks the whole package at import time (MappedAnnotationError).
from datetime import datetime  # noqa: TC003

from sqlalchemy import LargeBinary, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""



class LongTermTraceORM(Base):
    """ORM model for ltm_traces table."""

    __tablename__ = "ltm_traces"

    trace_uid: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(nullable=False, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    access_count: Mapped[int] = mapped_column(default=0, nullable=False)
    tags: Mapped[str] = mapped_column(Text, nullable=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)


class ShortTermTraceORM(Base):
    """ORM model for stm_traces table."""

    __tablename__ = "stm_traces"

    trace_uid: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[float] = mapped_column(nullable=False)
    access_count: Mapped[int] = mapped_column(default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    last_accessed: Mapped[datetime] = mapped_column(nullable=False)
    expires_at: Mapped[datetime] = mapped_column(nullable=False, index=True)
    tags: Mapped[str] = mapped_column(Text, nullable=True)
    extra: Mapped[str] = mapped_column(Text, nullable=True)
