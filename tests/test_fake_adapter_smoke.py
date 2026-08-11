"""Smoke test exercising the full EvolutionAdapter contract against FakeAdapter.

This test does NOT use any LLM, any CUGA API, or any Gaia-specific artifact
shape. It is purely a contract composition check.

Scope of the test
-----------------
* Adapter passes ``validate_adapter``.
* Base inventory is non-empty and each descriptor is well-formed.
* Materializing a candidate produces an isolated workspace whose artifact
  contents match the parent at materialization time.
* Applying a structured edit changes the candidate's artifact without leaking
  into the parent version (snapshot/lease isolation).
* Running a rollout returns a trace whose ``final_output`` reflects the
  candidate's current artifact contents.
* Replay is correctly reported as unsupported; calling
  ``replay_from_checkpoint`` raises (the core must fall back to full rollout).
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Make the project importable when running tests from the repo root via uv.
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # for `examples.fake_adapter`

from agent_evolve.adapters.base import validate_adapter  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    ArtifactEdit,
    CheckpointDescriptor,
    EvolutionTask,
)
from examples.fake_adapter import FakeAdapter  # noqa: E402


def test_fake_adapter_passes_contract_validation():
    adapter = FakeAdapter()
    validate_adapter(adapter)
    assert adapter.adapter_name == "fake"


def test_base_inventory_is_well_formed():
    adapter = FakeAdapter()
    descs = adapter.artifact_inventory("base-v0")
    assert len(descs) >= 1
    for d in descs:
        assert d.artifact_id
        assert d.kind
        assert d.format == "text/plain"
        assert d.version_hash.startswith("sha256:")
        assert d.readable is True
        assert d.writable is True
        assert d.merge_strategy  # opaque string, just non-empty


def test_materialize_isolates_from_parent():
    adapter = FakeAdapter()
    parent_version = "base-v0"
    parent_before = adapter.read_artifacts(parent_version, ("skills/retrieval",))

    ws = adapter.materialize_candidate(parent_version, "attempt-A")
    # Mutate the candidate workspace.
    adapter.apply_structured_edits(
        ws,
        (ArtifactEdit(
            artifact_id="skills/retrieval",
            operation="replace",
            payload={"content": "mutated-content"},
        ),),
    )

    # Parent must be unaffected.
    parent_after = adapter.read_artifacts(parent_version, ("skills/retrieval",))
    assert parent_after == parent_before, "materialize leaked edits to parent"


def test_two_siblings_do_not_interfere():
    adapter = FakeAdapter()
    parent = "base-v0"

    ws_a = adapter.materialize_candidate(parent, "attempt-A")
    ws_b = adapter.materialize_candidate(parent, "attempt-B")

    adapter.apply_structured_edits(
        ws_a,
        (ArtifactEdit(
            artifact_id="skills/retrieval",
            operation="replace",
            payload={"content": "from-A"},
        ),),
    )
    adapter.apply_structured_edits(
        ws_b,
        (ArtifactEdit(
            artifact_id="skills/retrieval",
            operation="replace",
            payload={"content": "from-B"},
        ),),
    )

    a_contents = adapter.read_artifacts(ws_a.version, ("skills/retrieval",))
    b_contents = adapter.read_artifacts(ws_b.version, ("skills/retrieval",))
    assert a_contents["skills/retrieval"] == "from-A"
    assert b_contents["skills/retrieval"] == "from-B"


def test_rollout_reflects_edit_and_trace_carries_provenance():
    adapter = FakeAdapter()
    ws = adapter.materialize_candidate("base-v0", "attempt-001")
    marker = "graphrag-retrieval"
    adapter.apply_structured_edits(
        ws,
        (ArtifactEdit(
            artifact_id="skills/retrieval",
            operation="replace",
            payload={"content": f"retrieve(query): use {marker} for top_k docs"},
        ),),
    )

    task = EvolutionTask(
        task_id="task-001",
        input_text="Find the API spec and call it.",
        expected_contract={"expected_substring": marker},
    )
    result = adapter.run_full_rollout(ws, task, "rollout-001")
    trace = adapter.capture_trace(result)

    assert trace.candidate_id == ws.version
    assert trace.task_id == task.task_id
    assert trace.status == "success"
    assert marker in trace.final_output
    assert len(trace.events) >= 2
    # Events form a chain: each (non-root) event has a parent_event_id.
    root_events = [e for e in trace.events if e.parent_event_id is None]
    assert len(root_events) == 1


def test_replay_is_unsupported_and_raises():
    adapter = FakeAdapter()
    assert adapter.supports_counterfactual_replay() is False

    ws = adapter.materialize_candidate("base-v0", "attempt-001")
    task = EvolutionTask(task_id="task-001", input_text="x")
    result = adapter.run_full_rollout(ws, task, "rollout-001")
    trace = adapter.capture_trace(result)

    assert adapter.discover_checkpoints(trace) == ()

    # The contract requires replay_from_checkpoint to raise when replay is
    # unsupported, so the core falls back to a full rollout.
    fake_checkpoint = CheckpointDescriptor(
        checkpoint_id="cp-1",
        trace_id=trace.trace_id,
        event_id=trace.events[0].event_id,
        state_hash="sha256:0",
        replay_scope=(),
    )
    with pytest.raises(RuntimeError):
        adapter.replay_from_checkpoint(fake_checkpoint, ws, task, "rollout-replay")


def test_unknown_artifact_in_edit_raises_loudly():
    adapter = FakeAdapter()
    ws = adapter.materialize_candidate("base-v0", "attempt-001")
    with pytest.raises(KeyError):
        adapter.apply_structured_edits(
            ws,
            (ArtifactEdit(
                artifact_id="skills/does-not-exist",
                operation="replace",
                payload={"content": "x"},
            ),),
        )


def test_unknown_edit_operation_raises():
    adapter = FakeAdapter()
    ws = adapter.materialize_candidate("base-v0", "attempt-001")
    with pytest.raises(ValueError):
        adapter.apply_structured_edits(
            ws,
            (ArtifactEdit(
                artifact_id="skills/retrieval",
                operation="delete-everything",  # unsupported
                payload={},
            ),),
        )
