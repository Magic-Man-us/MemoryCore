"""Regression tests for the P0 settings-construction bug."""

from __future__ import annotations

from memory_core import MemoryCoreSettings
from memory_core.storage import DatabaseSettings
from memory_core.storage.settings import DEFAULT_DB_URL


def test_database_settings_no_env_required(monkeypatch):
    """P0-4: construction must succeed with no environment at all."""
    monkeypatch.delenv("MEMORY_DB_URL", raising=False)
    assert DatabaseSettings().url == DEFAULT_DB_URL


def test_database_settings_overrides_win_over_env(monkeypatch):
    monkeypatch.setenv("MEMORY_DB_URL", "sqlite:///from_env.db")
    assert DatabaseSettings().url == "sqlite:///from_env.db"
    assert (
        DatabaseSettings.from_overrides({"url": "sqlite:///explicit.db"}).url
        == "sqlite:///explicit.db"
    )
    # None values in overrides fall back to env
    assert DatabaseSettings.from_overrides({"url": None}).url == "sqlite:///from_env.db"


def test_memory_core_settings_no_env_required(monkeypatch):
    for var in ("MEMORY_DB_URL", "MEMORY_CORE_ENABLE_STM"):
        monkeypatch.delenv(var, raising=False)
    settings = MemoryCoreSettings()
    assert settings.db.url == DEFAULT_DB_URL
    assert settings.enable_stm is False
