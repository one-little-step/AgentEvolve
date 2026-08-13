"""Recursive redaction and JSON research storage tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_evolve.core.errors import PersistenceSafetyError
from agent_evolve.core.storage import (
    JSONFileStorage,
    RedactedValue,
    StorageBackend,
    _safe_component,
    sanitize_for_persistence,
)


def test_storage_writes_and_reads_one_redacted_record(tmp_path: Path) -> None:
    store = JSONFileStorage(tmp_path)
    store.write_record("attempts", "attempt-1", {"summary": "safe"})
    assert store.read_record("attempts", "attempt-1") == {"summary": "safe"}


def test_storage_rejects_nested_expected_answer(tmp_path: Path) -> None:
    store = JSONFileStorage(tmp_path)
    with pytest.raises(PersistenceSafetyError):
        store.write_record("attempts", "attempt-1", {"nested": {"expected_answer": "x"}})


def test_json_storage_rejects_active_parallel_execution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parallel_execution"):
        JSONFileStorage(tmp_path, parallel_execution=True)


def test_storage_write_returns_redacted_value(tmp_path: Path) -> None:
    store = JSONFileStorage(tmp_path)
    result = store.write_record("attempts", "attempt-1", {"summary": "safe"})
    assert isinstance(result, RedactedValue)
    assert result.value == {"summary": "safe"}
    assert result.rule_hits == ()
    assert result.truncations == 0


def test_storage_read_missing_record_returns_none(tmp_path: Path) -> None:
    store = JSONFileStorage(tmp_path)
    assert store.read_record("attempts", "missing") is None


def test_storage_list_records_sorted(tmp_path: Path) -> None:
    store = JSONFileStorage(tmp_path)
    store.write_record("attempts", "b", {"v": 2})
    store.write_record("attempts", "a", {"v": 1})
    store.write_record("attempts", "c", {"v": 3})
    assert store.list_records("attempts") == ({"v": 1}, {"v": 2}, {"v": 3})


def test_storage_list_records_empty_type(tmp_path: Path) -> None:
    store = JSONFileStorage(tmp_path)
    assert store.list_records("nope") == ()


def test_storage_atomic_replacement_overwrites(tmp_path: Path) -> None:
    store = JSONFileStorage(tmp_path)
    store.write_record("attempts", "a", {"v": 1})
    store.write_record("attempts", "a", {"v": 2})
    assert store.read_record("attempts", "a") == {"v": 2}
    assert store.list_records("attempts") == ({"v": 2},)


def test_storage_write_is_deterministic_json(tmp_path: Path) -> None:
    store = JSONFileStorage(tmp_path)
    store.write_record("attempts", "a", {"b": 1, "a": {"d": 2, "c": 3}})
    raw = (tmp_path / "attempts" / "a.json").read_text(encoding="utf-8")
    assert raw == '{"a":{"c":3,"d":2},"b":1}'


def test_storage_close_is_noop(tmp_path: Path) -> None:
    store = JSONFileStorage(tmp_path)
    assert store.close() is None


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b", "..\\evil", "../evil"])
def test_safe_component_rejects_unsafe_components(bad: str) -> None:
    with pytest.raises(ValueError):
        _safe_component(bad)


def test_safe_component_accepts_opaque_component() -> None:
    assert _safe_component("attempt-1") == "attempt-1"


@pytest.mark.parametrize(
    ("record_type", "record_id"),
    [("../evil", "a"), ("a", "../evil"), ("a/b", "a"), ("a", "a\\b")],
)
def test_storage_write_rejects_traversal(
    tmp_path: Path, record_type: str, record_id: str
) -> None:
    store = JSONFileStorage(tmp_path)
    with pytest.raises(ValueError):
        store.write_record(record_type, record_id, {"v": 1})


@pytest.mark.parametrize(
    ("record_type", "record_id"),
    [("../evil", "a"), ("a", "../evil"), ("a/b", "a"), ("a", "a\\b")],
)
def test_storage_read_rejects_traversal(
    tmp_path: Path, record_type: str, record_id: str
) -> None:
    store = JSONFileStorage(tmp_path)
    with pytest.raises(ValueError):
        store.read_record(record_type, record_id)


@pytest.mark.parametrize("record_type", ["../evil", "a/b", "a\\b"])
def test_storage_list_rejects_unsafe_record_type(tmp_path: Path, record_type: str) -> None:
    store = JSONFileStorage(tmp_path)
    with pytest.raises(ValueError):
        store.list_records(record_type)


@pytest.mark.parametrize("field", ["expected_answer", "password", "api_key", "secret"])
def test_sanitize_rejects_denylisted_top_level_field(field: str) -> None:
    with pytest.raises(PersistenceSafetyError):
        sanitize_for_persistence({field: "value"})


@pytest.mark.parametrize(
    "field",
    [
        "Expected_Answer",
        "PASSWORD",
        "Api_Key",
        "credential",
        "credentials",
        "label",
        "labels",
        "regex",
        "raw_prompt",
        "raw_response",
        "raw_trace",
        "token",
    ],
)
def test_sanitize_rejects_denylisted_field_case_insensitive(field: str) -> None:
    with pytest.raises(PersistenceSafetyError):
        sanitize_for_persistence({field: "value"})


def test_sanitize_rejects_nested_denylisted_field() -> None:
    with pytest.raises(PersistenceSafetyError):
        sanitize_for_persistence({"nested": {"deep": [{"expected_answer": "x"}]}})


@pytest.mark.parametrize(
    "secret",
    [
        "AKIAIOSFODNN7EXAMPLE",
        "sk-abcdefghijklmnopqrstuvwx",
        "ghp_abcdefghijklmnopqrstuvwxyz123456",
        "-----BEGIN RSA PRIVATE KEY-----",
        "Bearer abcdefghijklmnopqrstuvwxyz123456",
    ],
)
def test_sanitize_rejects_secret_value_patterns(secret: str) -> None:
    with pytest.raises(PersistenceSafetyError):
        sanitize_for_persistence({"summary": secret})


def test_sanitize_rejects_secret_nested_in_sequence() -> None:
    with pytest.raises(PersistenceSafetyError):
        sanitize_for_persistence({"notes": ["ok", "AKIAIOSFODNN7EXAMPLE"]})


def test_sanitize_passes_clean_nested_structure() -> None:
    payload = {
        "summary": "safe text",
        "nested": {"list": [1, 2, 3], "mapping": {"k": True, "n": None}},
    }
    result = sanitize_for_persistence(payload)
    assert result.value == payload
    assert result.rule_hits == ()
    assert result.truncations == 0


def test_sanitize_truncates_long_strings() -> None:
    result = sanitize_for_persistence({"summary": "x" * 100}, max_string_length=10)
    assert result.value == {"summary": "x" * 10}
    assert result.truncations == 1
    assert "string_bounded" in result.rule_hits


def test_sanitize_counts_each_truncation() -> None:
    result = sanitize_for_persistence(
        {"a": "y" * 100, "b": ["z" * 100, "w" * 100]}, max_string_length=10
    )
    assert result.truncations == 3
    assert result.rule_hits == ("string_bounded",)


def test_sanitize_does_not_mutate_input() -> None:
    payload = {"nested": {"a": [1, 2], "b": "text"}}
    sanitize_for_persistence(payload)
    assert payload == {"nested": {"a": [1, 2], "b": "text"}}


def test_storage_backend_protocol_is_structural(tmp_path: Path) -> None:
    store = JSONFileStorage(tmp_path)
    assert isinstance(store, StorageBackend)
