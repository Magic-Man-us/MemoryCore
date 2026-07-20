"""Settings objects: DSN construction and override sanitization."""

from __future__ import annotations

from memory_core_py.storage.settings import (
    MemoryType,
    MySQLStoreSettings,
    RedisWorkingMemorySettings,
)


def mysql_settings(**overrides) -> MySQLStoreSettings:
    defaults = dict(
        memory_type=MemoryType.LTM,
        host="db.local",
        user="mc",
        password="s3cret",
        database="memory",
    )
    defaults.update(overrides)
    return MySQLStoreSettings(**defaults)


class TestMySQLSettings:
    def test_dsn_uses_mysqlconnector_driver(self):
        settings = mysql_settings(port=3307)
        assert settings.dsn == "mysql+mysqlconnector://mc:s3cret@db.local:3307/memory"

    def test_apply_overrides_accepts_known_keys(self):
        updated = mysql_settings().apply_overrides({"host": "other", "port": 3308})
        assert updated.host == "other"
        assert updated.port == 3308

    def test_apply_overrides_drops_unknown_and_none(self):
        base = mysql_settings()
        updated = base.apply_overrides({"host": None, "nonsense": "x"})
        assert updated is base  # nothing valid to apply -> same instance


class TestRedisSettings:
    def test_dsn_from_parts_with_password(self):
        settings = RedisWorkingMemorySettings(
            host="cache.local", port=6380, database=2, password="pw"
        )
        assert settings.dsn == "redis://:pw@cache.local:6380/2"

    def test_url_override_wins(self):
        settings = RedisWorkingMemorySettings(url="redis://elsewhere:1234/9")
        assert settings.dsn == "redis://elsewhere:1234/9"

    def test_defaults(self):
        settings = RedisWorkingMemorySettings()
        assert settings.dsn == "redis://localhost:6379/0"
        assert settings.ttl_seconds == 900
