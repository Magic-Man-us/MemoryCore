"""Shared fixtures: a temp SQLite database per test."""

from __future__ import annotations

import pytest

from memory_core.storage import Database, DatabaseSettings


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(settings=DatabaseSettings(url=f"sqlite:///{tmp_path}/test.db"))
    database.create_schema()
    return database


class FakeEmbedder:
    """Deterministic embedder: 8-dim vector derived from the text bytes."""

    dim = 8

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            data = text.encode()
            out.append([((data[i % max(len(data), 1)] if data else 0) % 97) / 97.0 + 0.01 for i in range(self.dim)])
        return out


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()
