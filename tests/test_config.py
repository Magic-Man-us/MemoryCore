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


def test_unknown_override_keys_raise(tmp_path):
    with pytest.raises(ValueError, match="unknown override keys"):
        build_memory_system(
            sqlite_path=tmp_path / "t.db", overrides={"enable_sm": True}
        )
    with pytest.raises(ValueError, match="unknown db override keys"):
        build_memory_system(overrides={"db": {"uri": "sqlite://"}})


def test_overrides_are_validated(tmp_path):
    """Overrides used to bypass validation entirely (a negative TTL went through)."""
    with pytest.raises(ValidationError):
        build_memory_system(
            sqlite_path=tmp_path / "t.db", overrides={"wm_ttl_seconds": -5}
        )


def test_nested_env_configures_scoring(monkeypatch):
    from memory_core import MemoryCoreSettings

    monkeypatch.setenv("MEMORY_CORE_SCORING__RECENCY", "0.2")
    assert MemoryCoreSettings().scoring.recency == 0.2


def test_settings_instance_plus_overrides(tmp_path):
    from memory_core import MemoryCoreSettings, resolve_settings

    base = MemoryCoreSettings(enable_stm=True)
    resolved = resolve_settings(
        base, overrides={"scoring": {"recency": 0.1}}, sqlite_path=tmp_path / "t.db"
    )
    assert resolved.enable_stm is True
    assert resolved.scoring.recency == 0.1
    assert resolved.scoring.semantic == 0.7
    assert resolved.db.sqlite_path == str(tmp_path / "t.db")
