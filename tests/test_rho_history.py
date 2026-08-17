"""Tests for RHO historical-corpus loading.

A stale-format trace has no tool_call events and actor_id=None throughout; it
cannot support group diagnosis, so it must be rejected rather than silently
treated as history.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent_evolve.core.rho.history import (
    HistoricalRecord,
    load_history,
)


def _write_trace(root: Path, run_id: str, *, current_format: bool) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    if current_format:
        events = [
            {
                "event_id": "e1",
                "kind": "graph_node_start",
                "actor_id": "call_model",
                "parent_event_id": None,
                "payload": {"node": "call_model"},
            },
            {
                "event_id": "e2",
                "kind": "llm_call_end",
                "actor_id": "call_model",
                "parent_event_id": "e1",
                "payload": {"text": "I will search."},
            },
        ]
    else:
        events = [
            {
                "event_id": f"e{i}",
                "kind": "stream_event",
                "actor_id": None,
                "parent_event_id": None,
                "payload": {},
            }
            for i in range(8)
        ]
    payload = {
        "task_id": "gaia-1",
        "input_text": "what is 2+2",
        "events": events,
        "final_output": "4",
        "tool_observations": [],
        "harness_version": "vanilla",
    }
    path = run_dir / "causal-trace.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_current_format_trace(tmp_path: Path) -> None:
    _write_trace(tmp_path, "run-1", current_format=True)

    report = load_history(tmp_path)

    assert len(report.records) == 1
    assert report.is_cold_start is False
    record = report.records[0]
    assert isinstance(record, HistoricalRecord)
    assert record.task_id == "gaia-1"
    assert record.final_output == "4"
    assert record.harness_version == "vanilla"
    assert record.content_hash.startswith("sha256:")


def test_rejects_stale_format_trace_with_a_reason(tmp_path: Path) -> None:
    _write_trace(tmp_path, "run-1", current_format=False)

    report = load_history(tmp_path)

    assert report.records == ()
    assert len(report.rejected) == 1
    path, reason = report.rejected[0]
    assert "run-1" in path
    assert "actor_id" in reason


def test_missing_root_is_a_cold_start_not_an_error(tmp_path: Path) -> None:
    report = load_history(tmp_path / "does-not-exist")

    assert report.records == ()
    assert report.is_cold_start is True


def test_empty_root_is_a_cold_start(tmp_path: Path) -> None:
    report = load_history(tmp_path)

    assert report.is_cold_start is True


def test_content_hash_changes_when_trace_changes(tmp_path: Path) -> None:
    path = _write_trace(tmp_path, "run-1", current_format=True)
    first = load_history(tmp_path).records[0].content_hash

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["final_output"] = "5"
    path.write_text(json.dumps(payload), encoding="utf-8")
    second = load_history(tmp_path).records[0].content_hash

    assert first != second
