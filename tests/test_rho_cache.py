"""Tests for the RHO content-hash disk cache."""
from __future__ import annotations

from pathlib import Path

from agent_evolve.core.rho.cache import JsonDiskCache


def test_put_then_get_hits(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path)
    cache.put("key-a", {"difficulty": 7.5})

    assert cache.get("key-a") == {"difficulty": 7.5}
    assert cache.hits == 1
    assert cache.misses == 0


def test_unknown_key_misses(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path)

    assert cache.get("nope") is None
    assert cache.misses == 1


def test_different_key_does_not_collide(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path)
    cache.put("hash-1", {"v": 1})

    assert cache.get("hash-2") is None


def test_none_root_disables_caching_and_writes_nothing(tmp_path: Path) -> None:
    cache = JsonDiskCache(None)
    cache.put("key-a", {"v": 1})

    assert cache.get("key-a") is None
    assert cache.misses == 1
    assert list(tmp_path.iterdir()) == []


def test_survives_a_new_instance_on_the_same_root(tmp_path: Path) -> None:
    JsonDiskCache(tmp_path).put("key-a", {"v": 1})

    assert JsonDiskCache(tmp_path).get("key-a") == {"v": 1}


def test_corrupt_entry_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path)
    cache.put("key-a", {"v": 1})
    # Corrupt the stored entry.
    for path in tmp_path.iterdir():
        path.write_text("{not json", encoding="utf-8")

    assert cache.get("key-a") is None
    assert cache.misses == 1
