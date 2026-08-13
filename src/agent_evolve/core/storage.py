"""Path-contained JSON research storage with a recursive redaction gateway.

Per docs/architecture/storage-and-transactions.md:118-133, every write passes
through a recursive sanitizer that walks mappings, sequences, nested records,
and strings. It rejects prohibited categories by field name and by obvious
secret value pattern, and truncates long strings to bounded summaries. If a
safe representation cannot be obtained, the write fails closed with
:class:`PersistenceSafetyError`.

This JSON implementation deliberately defers SQLite WAL pragmas, atomic
multi-record barriers, idempotency keys, interrupted-run recovery,
content-addressed blob staging, orphan cleanup, and durable leases to Phase 5.
Per-record temp-write-then-rename atomicity is the complete persistence
guarantee for the single-threaded research path.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

from agent_evolve.core.errors import PersistenceSafetyError

# ---------------------------------------------------------------------- #
# Denylist and secret pattern tables
# ---------------------------------------------------------------------- #

_DENYLIST_FIELDS = frozenset(
    {
        "expected_answer",
        "expected_answers",
        "expected_label",
        "expected_labels",
        "expected_regex",
        "expected",
        "label",
        "labels",
        "regex",
        "regexes",
        "secret",
        "secrets",
        "token",
        "tokens",
        "password",
        "passwords",
        "api_key",
        "api_keys",
        "apikey",
        "credential",
        "credentials",
        "raw_prompt",
        "raw_prompts",
        "raw_response",
        "raw_responses",
        "raw_trace",
        "raw_traces",
        "raw_trace_body",
    }
)

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}\b"),
)

MAX_STRING_LENGTH = 2000


# ---------------------------------------------------------------------- #
# Redacted value
# ---------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RedactedValue:
    """A sanitized payload plus the redaction summary produced for it."""

    value: object
    rule_hits: tuple[str, ...] = field(default_factory=tuple)
    truncations: int = 0


# ---------------------------------------------------------------------- #
# Recursive sanitizer
# ---------------------------------------------------------------------- #


def _looks_like_secret(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SECRET_PATTERNS)


def _sanitize(value: object, hits: set[str], max_len: int, truncations: list[int]) -> object:
    if isinstance(value, Mapping):
        result: dict[object, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PersistenceSafetyError(
                    f"refusing to persist non-string mapping key of type "
                    f"{type(key).__name__}"
                )
            if key.lower() in _DENYLIST_FIELDS or key.lower().startswith("expected_"):
                raise PersistenceSafetyError(
                    f"refusing to persist denied field {key!r} "
                    "(credential, expected answer, evaluator internal, label, "
                    "regex, or raw model material)"
                )
            result[key] = _sanitize(item, hits, max_len, truncations)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, hits, max_len, truncations) for item in value]
    if isinstance(value, str):
        if _looks_like_secret(value):
            raise PersistenceSafetyError(
                "refusing to persist a value that matches a secret pattern"
            )
        if len(value) > max_len:
            truncations[0] += 1
            hits.add("string_bounded")
            return value[:max_len]
        return value
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    raise PersistenceSafetyError(
        f"refusing to persist non-JSON-serializable value of type {type(value).__name__}"
    )


def sanitize_for_persistence(
    payload: Mapping[str, object],
    *,
    max_string_length: int = MAX_STRING_LENGTH,
) -> RedactedValue:
    """Recursively sanitize ``payload`` and return a :class:`RedactedValue`.

    Raises :class:`PersistenceSafetyError` when prohibited material (denylisted
    field names, obvious secret values, or non-serializable types) is found,
    failing closed rather than silently persisting an unsafe representation.
    """
    hits: set[str] = set()
    truncations = [0]
    value = _sanitize(payload, hits, max_string_length, truncations)
    return RedactedValue(
        value=value,
        rule_hits=tuple(sorted(hits)),
        truncations=truncations[0],
    )


# ---------------------------------------------------------------------- #
# Storage backend
# ---------------------------------------------------------------------- #


@runtime_checkable
class StorageBackend(Protocol):
    def write_record(
        self, record_type: str, record_id: str, payload: Mapping[str, object]
    ) -> RedactedValue: ...
    def read_record(
        self, record_type: str, record_id: str
    ) -> Mapping[str, object] | None: ...
    def list_records(self, record_type: str) -> tuple[Mapping[str, object], ...]: ...
    def close(self) -> None: ...


def _safe_component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("record type and ID must be safe opaque path components")
    return value


class JSONFileStorage:
    """Single-threaded JSON file backend with per-record atomic writes."""

    def __init__(self, root: Path, parallel_execution: bool = False) -> None:
        if parallel_execution:
            raise ValueError(
                "JSONFileStorage does not support parallel_execution; the JSON "
                "backend is single-threaded only"
            )
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _record_path(self, record_type: str, record_id: str) -> Path:
        return self._root / _safe_component(record_type) / f"{_safe_component(record_id)}.json"

    def write_record(
        self, record_type: str, record_id: str, payload: Mapping[str, object]
    ) -> RedactedValue:
        redacted = sanitize_for_persistence(payload)
        data = json.dumps(redacted.value, sort_keys=True, separators=(",", ":"))
        path = self._record_path(record_type, record_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(path)
        return redacted

    def read_record(
        self, record_type: str, record_id: str
    ) -> Mapping[str, object] | None:
        path = self._record_path(record_type, record_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_records(self, record_type: str) -> tuple[Mapping[str, object], ...]:
        record_dir = self._root / _safe_component(record_type)
        if not record_dir.is_dir():
            return ()
        records: list[Mapping[str, object]] = []
        for path in sorted(record_dir.glob("*.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
        return tuple(records)

    def close(self) -> None:
        return None
