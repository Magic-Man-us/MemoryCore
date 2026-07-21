"""build_memory_system wiring: override precedence and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from memory_core import build_memory_system


async def test_build_with_db_override(tmp_path):
    system = build_memory_system(
        overrides={"db": {"url": f"sqlite:///{tmp_path}/override.db"}}
    )
    await system.remember(
        user_id="alice", summary="s", importance=0.5, tags=[], embedding=[0.1, 0.2]
    )
    assert (tmp_path / "override.db").exists()


def test_none_db_override_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_DB_URL", f"sqlite:///{tmp_path}/from_env.db")
    system = build_memory_system(overrides={"db": {"url": None}})
    assert system.ltm_store._db.settings.url == f"sqlite:///{tmp_path}/from_env.db"


def test_invalid_db_override_raises(tmp_path):
    with pytest.raises(ValidationError):
        build_memory_system(overrides={"db": {"url": 123}})


def test_stm_toggle(tmp_path):
    url = f"sqlite:///{tmp_path}/t.db"
    without = build_memory_system(overrides={"db": {"url": url}})
    assert without.stm_store is None
    with_stm = build_memory_system(overrides={"db": {"url": url}, "enable_stm": True})
    assert with_stm.stm_store is not None
