"""Append-only, agent-scoped edit experiment memory with ablation modes."""
from __future__ import annotations

import json
import os
import tempfile
import hashlib
import re
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any


_PROHIBITED_TERMS = ("api_key", "token", "secret", "expected", "evaluator", "regex", "label")
_PROHIBITED_ASSIGNMENT = re.compile(
    r"\b(?:api_key|token|secret|expected|evaluator|regex|label)\b\s*(?:=|:)\s*(?:\"[^\"]*\"|'[^']*'|\S+)",
    re.IGNORECASE,
)


def redact_history_value(value: Any) -> Any:
    """Remove prohibited fields and inline key/value material from history data."""
    if isinstance(value, dict):
        return {
            str(key): redact_history_value(item)
            for key, item in value.items()
            if not any(term in str(key).lower() for term in _PROHIBITED_TERMS)
        }
    if isinstance(value, list):
        return [redact_history_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_history_value(item) for item in value)
    if isinstance(value, str):
        return _PROHIBITED_ASSIGNMENT.sub("[REDACTED]", value)
    return value


def _redact_record(record: EditHistoryRecord) -> EditHistoryRecord:
    payload = redact_history_value(asdict(record))
    return EditHistoryRecord(**payload)


@dataclass(frozen=True, slots=True)
class EditHistoryRecord:
    record_id: str
    lineage_id: str
    module: str
    text: str
    outcome: str


@dataclass(frozen=True, slots=True)
class HistoryRetrieval:
    mode: str
    records: tuple[EditHistoryRecord, ...]
    fallback_reason: str | None = None


class EditHistoryStore:
    def __init__(self, root: Path, agent_name: str, *, retrieval_enabled: bool, semantic_enabled: bool, embedder: Any | None = None) -> None:
        self.path = Path(root) / "history" / agent_name / "records.jsonl"
        self.retrieval_enabled = retrieval_enabled
        self.semantic_enabled = semantic_enabled
        self.embedder = embedder

    @property
    def manifest_path(self) -> Path:
        return self.path.parent / "manifest.json"

    def _embedding_path(self, record: EditHistoryRecord) -> Path:
        safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in record.record_id)
        return self.path.parent / "embeddings" / f"{safe_id}.json"

    def append(self, record: EditHistoryRecord) -> None:
        records = [*self._records(), _redact_record(record)]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".records.", suffix=".tmp", dir=self.path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(asdict(item), sort_keys=True) + "\n")
        Path(temporary).replace(self.path)
        self._write_manifest()

    def retrieve(self, query: str, *, lineage_id: str, module: str, minimum_records: int) -> HistoryRetrieval:
        if not self.retrieval_enabled:
            return HistoryRetrieval("off", ())
        query = str(redact_history_value(query))
        records = self._records()
        local = [record for record in records if record.lineage_id == lineage_id and record.module == module]
        selected = list(local)
        if len(selected) < minimum_records:
            selected.extend(record for record in records if record.module == module and record not in selected)
        if len(selected) < minimum_records:
            selected.extend(record for record in records if record not in selected)
        terms = set(query.lower().split())
        lexical = lambda record: len(terms & set(record.text.lower().split()))
        if self.semantic_enabled and self.embedder is not None:
            embedder = self.embedder
            try:
                query_vector = embedder.embed_query(query)
                def semantic(record: EditHistoryRecord) -> float:
                    vector = self._embedding(record, embedder)
                    denominator = sqrt(sum(x * x for x in query_vector)) * sqrt(sum(x * x for x in vector))
                    return sum(x * y for x, y in zip(query_vector, vector)) / denominator if denominator else 0.0
                selected.sort(key=lambda record: (-semantic(record), -lexical(record), record.record_id))
                return HistoryRetrieval("semantic", tuple(selected))
            except Exception as exc:  # noqa: BLE001 - ablations must remain runnable offline
                selected.sort(key=lambda record: (-lexical(record), record.record_id))
                return HistoryRetrieval("lexical", tuple(selected), str(exc))
        selected.sort(key=lambda record: (-lexical(record), record.record_id))
        return HistoryRetrieval("lexical", tuple(selected))

    def _embedding(self, record: EditHistoryRecord, embedder: Any) -> list[float]:
        text_hash = hashlib.sha256(record.text.encode()).hexdigest()
        path = self._embedding_path(record)
        model = str(getattr(embedder, "model", getattr(embedder, "model_name", type(embedder).__name__)))
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            vector = cached["vector"]
            if (
                cached["schema_version"] == "1"
                and cached["text_sha256"] == text_hash
                and cached["embedding_model"] == model
                and isinstance(vector, list)
                and vector
                and all(isinstance(item, (int, float)) for item in vector)
            ):
                return [float(item) for item in vector]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        vector = [float(item) for item in embedder.embed_document(record.text)]
        if not vector:
            raise ValueError("empty embedding vector")
        self._atomic_write_json(path, {
            "schema_version": "1", "record_id": record.record_id,
            "text_sha256": text_hash, "embedding_model": model,
            "dimension": len(vector), "vector": vector,
        })
        self._write_manifest()
        return vector

    def _write_manifest(self) -> None:
        payload: dict[str, object] = {"schema_version": "1", "record_count": len(self._records())}
        if self.embedder is not None:
            payload["embedding_model"] = str(getattr(self.embedder, "model", getattr(self.embedder, "model_name", type(self.embedder).__name__)))
        self._atomic_write_json(self.manifest_path, payload)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        Path(temporary).replace(path)

    def _records(self) -> list[EditHistoryRecord]:
        if not self.path.exists():
            return []
        return [_redact_record(EditHistoryRecord(**json.loads(line))) for line in self.path.read_text(encoding="utf-8").splitlines() if line]
