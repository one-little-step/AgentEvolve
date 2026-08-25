"""Phase 3 — TapeIndex over recorded traces (offline, no network).

Fixture mirrors the PRODUCTION serialized trace shape observed in reference
trace 3306905e (2026-08-25): events.jsonl lines with top-level keys
actor_id/event_id/kind/parent_event_id/payload/sequence/timestamp and refs
nested inside `payload`; blobs stored as payloads/<sha256>.json, verbatim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent_evolve.core.tape import (
    ToolClass,
    TapeIntegrityError,
    TapeIndex,
    ToolTapeClassifier,
)


def _blob(payloads: Path, raw: bytes) -> str:
    ref = hashlib.sha256(raw).hexdigest()
    (payloads / f"{ref}.json").write_bytes(raw)
    return ref


def _event(kind: str, seq: int, payload: dict, parent: str | None = None) -> dict:
    return {
        "actor_id": None,
        "event_id": f"graph:{seq}",
        "kind": kind,
        "parent_event_id": parent,
        "payload": payload,
        "sequence": seq,
        "timestamp": "2026-08-25T14:10:45Z",
    }


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    """Two-step synthetic trace: node -> llm boundary -> tool call."""
    payloads = tmp_path / "payloads"
    payloads.mkdir()

    args_raw = json.dumps({"query": "stable-query-text", "scope": "agent"}).encode()
    out_raw = json.dumps({"results": ["alpha-stable", "beta-stable"]}).encode()
    msgs_raw = b'[{"role": "user", "content": "fixture-prompt"}]'
    resp_raw = b'{"content": "fixture-response"}'
    before_a = b'{"task": {"input": "fixture-task"}}'
    after_a = b'{"task": {"input": "fixture-task"}, "scratch": 1}'

    refs = {
        "args": _blob(payloads, args_raw),
        "out": _blob(payloads, out_raw),
        "msgs": _blob(payloads, msgs_raw),
        "resp": _blob(payloads, resp_raw),
        "before": _blob(payloads, before_a),
        "after": _blob(payloads, after_a),
    }

    run = "run-fixture"
    events = [
        _event("graph_node_start", 0, {
            "node": "planner", "step": 0, "run_id": run,
            "parent_run_id": None, "state_before_ref": refs["before"],
        }),
        _event("llm_call_start", 1, {"run_id": run, "parent_run_id": None,
                                     "messages_ref": refs["msgs"]}),
        _event("llm_call_end", 2, {"run_id": run, "parent_run_id": None,
                                   "response_ref": refs["resp"]}),
        _event("graph_node_end", 3, {"node": "planner", "run_id": run,
                                     "parent_run_id": None,
                                     "state_after_ref": refs["after"]}),
        _event("graph_node_start", 4, {
            "node": "sandbox", "step": 1, "run_id": run,
            "parent_run_id": None, "state_before_ref": refs["after"],
        }),
        _event("graph_tool_start", 5, {
            "run_id": "tool-run", "parent_run_id": run,
            "tool_name": "knowledge_search_knowledge",
            "args_ref": refs["args"],
        }, parent="graph:4"),
        _event("graph_tool_end", 6, {
            "run_id": "tool-run", "parent_run_id": run,
            "output_ref": refs["out"],
        }, parent="graph:4"),
        _event("graph_node_end", 7, {"node": "sandbox", "run_id": run,
                                     "parent_run_id": None,
                                     "state_after_ref": refs["out"]}),
    ]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return tmp_path


class TestRefResolution:
    def test_all_refs_resolve_to_verbatim_bytes(self, trace_dir: Path) -> None:
        idx = TapeIndex.load(trace_dir)
        idx.verify_all_refs()  # must not raise

    def test_resolve_returns_exact_bytes(self, trace_dir: Path) -> None:
        idx = TapeIndex.load(trace_dir)
        entry = idx.tool_entries[0]
        assert json.loads(idx.resolve(entry.args_ref))["query"] == "stable-query-text"
        assert json.loads(idx.resolve(entry.output_ref))["results"] == [
            "alpha-stable", "beta-stable"]

    def test_missing_blob_named_honestly(self, trace_dir: Path) -> None:
        victim = sorted((trace_dir / "payloads").glob("*.json"))[0]
        victim.unlink()
        idx = TapeIndex.load(trace_dir)
        with pytest.raises(TapeIntegrityError) as excinfo:
            idx.verify_all_refs()
        assert victim.stem in str(excinfo.value)

    def test_tampered_blob_fails_on_resolve(self, trace_dir: Path) -> None:
        idx = TapeIndex.load(trace_dir)
        ref = idx.llm_boundaries[0].response_ref
        blob = trace_dir / "payloads" / f"{ref}.json"
        blob.write_bytes(b'{"content": "TAMPERED"}')
        with pytest.raises(TapeIntegrityError):
            idx.resolve(ref)


class TestIndexes:
    def test_llm_boundaries_in_sequence_order(self, trace_dir: Path) -> None:
        idx = TapeIndex.load(trace_dir)
        assert [b.sequence for b in idx.llm_boundaries] == [1]
        assert idx.llm_boundaries[0].messages_ref
        assert idx.llm_boundaries[0].response_ref

    def test_tool_entry_carries_name_and_both_refs(self, trace_dir: Path) -> None:
        idx = TapeIndex.load(trace_dir)
        assert len(idx.tool_entries) == 1
        entry = idx.tool_entries[0]
        assert entry.tool_name == "knowledge_search_knowledge"
        assert entry.args_ref and entry.output_ref
        assert entry.sequence == 5

    def test_node_steps_recorded_for_resume_addressing(self, trace_dir: Path) -> None:
        idx = TapeIndex.load(trace_dir)
        starts = [(n.node, n.step, n.state_before_ref) for n in idx.node_starts]
        assert starts[0][0] == "planner" and starts[0][1] == 0
        assert starts[1][0] == "sandbox" and starts[1][1] == 1


class TestClassification:
    def test_registered_pattern_classifies(self) -> None:
        clf = ToolTapeClassifier()
        clf.register("knowledge_*", ToolClass.STATEFUL_LOCAL)
        assert clf.classify("knowledge_search_knowledge") == (
            ToolClass.STATEFUL_LOCAL, None)

    def test_unknown_tool_is_conservatively_unrecordable(self) -> None:
        clf = ToolTapeClassifier()
        cls, reason = clf.classify("calculator")
        assert cls is ToolClass.UNRECORDABLE
        assert "unclassified" in reason

    def test_none_tool_name_handled(self) -> None:
        clf = ToolTapeClassifier()
        cls, reason = clf.classify(None)
        assert cls is ToolClass.UNRECORDABLE

    def test_first_matching_registration_wins(self) -> None:
        clf = ToolTapeClassifier()
        clf.register("web_*", ToolClass.EXTERNAL)
        clf.register("web_search*", ToolClass.EXTERNAL)
        clf.register("*_search_*", ToolClass.STATEFUL_LOCAL)
        # knowledge_search_knowledge matches *_search_* only
        assert clf.classify("knowledge_search_knowledge")[0] is ToolClass.STATEFUL_LOCAL


class TestStrictDryRun:
    def test_dry_run_classifies_every_tool_call(self, trace_dir: Path) -> None:
        idx = TapeIndex.load(trace_dir)
        clf = ToolTapeClassifier()
        clf.register("knowledge_*", ToolClass.STATEFUL_LOCAL)
        report = idx.dry_classify(clf)
        assert report.total_calls == 1
        assert report.counts[ToolClass.STATEFUL_LOCAL] == 1
        assert not report.unclassified_names  # nothing fell through unregistered

    def test_dry_run_surfaces_unregistered_honestly(self, trace_dir: Path) -> None:
        idx = TapeIndex.load(trace_dir)
        report = idx.dry_classify(ToolTapeClassifier())  # empty registry
        assert report.unclassified_names == ["knowledge_search_knowledge"]
