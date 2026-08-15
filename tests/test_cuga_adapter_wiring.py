"""Phase 8 regressions for the CUGA adapter boundary.

These tests pin the three loop-breaking wiring bugs:

1. ``run_full_rollout`` must deliver the candidate's edited artifacts to the
   CUGA wrapper as harness keys, otherwise evolution measures model noise.
2. ``capture_trace`` must map ``actor_id``/``parent_event_id`` from the rich
   persisted trace, otherwise causal blame is permanently empty.
3. ``artifact_inventory`` must reflect a registered candidate's artifacts,
   otherwise the editor has nothing to select.

They also pin two safety properties: unmappable artifact ids fail loudly
rather than being silently dropped, and trace mapping never dereferences
payload blobs (which may hold raw prompts/state).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_evolve.adapters.cuga_adapter import CugaAdapter
from agent_evolve.core.analyzer import FakeAnalyzerJudge
from agent_evolve.core.contracts import ArtifactEdit, EvolutionTask


class RecordingWrapper:
    """Wrapper double that records exactly what harness config it received."""

    def __init__(
        self,
        artifacts: dict[str, str] | None = None,
        causal_trace_path: Path | None = None,
        final_output: str = "four",
    ) -> None:
        self.artifacts = dict(artifacts or {})
        self.calls: list[tuple[str, dict[str, object]]] = []
        self._causal_trace_path = causal_trace_path
        self._final_output = final_output

    def run_task(self, task_id: str, harness_config):
        self.calls.append((task_id, dict(harness_config)))
        result: dict[str, object] = {
            "task_id": task_id,
            "status": "success",
            "final_output": self._final_output,
            "events": [{"event_id": f"{task_id}:started", "kind": "run_started"}],
        }
        if self._causal_trace_path is not None:
            result["causal_trace_path"] = str(self._causal_trace_path)
        return result

    def get_artifacts(self) -> dict[str, str]:
        return dict(self.artifacts)

    def update_artifact(self, artifact_id: str, content: str) -> None:
        self.artifacts[artifact_id] = content

    @property
    def last_harness(self) -> dict[str, object]:
        return self.calls[-1][1]


def _write_rich_trace(
    directory: Path,
    *,
    task_id: str = "task-1",
    final_output: str = "four",
    status: str = "success",
) -> Path:
    """Write a minimal rich trace mirroring the live CUGA trace layout.

    Payload values are content-addressed refs only, matching the verified
    live format where the longest inline payload string is a 64-char hash.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "payloads").mkdir(exist_ok=True)
    secret_blob = {"expected_answer": "SECRET-DO-NOT-LEAK"}
    ref = "a" * 64
    (directory / "payloads" / f"{ref}.json").write_text(
        json.dumps(secret_blob), encoding="utf-8"
    )
    events = [
        {
            "event_id": "graph:0",
            "kind": "graph_node_start",
            "actor_id": "prepare",
            "parent_event_id": None,
            "timestamp": "2026-08-15T04:22:00Z",
            "sequence": 0,
            "payload": {"node": "prepare", "state_before_ref": ref},
        },
        {
            "event_id": "graph:1",
            "kind": "llm_call_start",
            "actor_id": None,
            "parent_event_id": "graph:0",
            "timestamp": "2026-08-15T04:22:01Z",
            "sequence": 1,
            "payload": {"messages_ref": ref},
        },
        {
            "event_id": "graph:2",
            "kind": "graph_node_start",
            "actor_id": "sandbox",
            "parent_event_id": "graph:0",
            "timestamp": "2026-08-15T04:22:02Z",
            "sequence": 2,
            "payload": {"node": "sandbox", "state_before_ref": ref},
        },
        {
            "event_id": "graph:3",
            "kind": "graph_node_end",
            "actor_id": "sandbox",
            "parent_event_id": "graph:0",
            "timestamp": "2026-08-15T04:22:03Z",
            "sequence": 3,
            "payload": {"node": "sandbox", "routed_to": "final", "state_after_ref": ref},
        },
    ]
    (directory / "causal-trace.json").write_text(
        json.dumps(
            {
                "run_id": "run-abc",
                "task_id": task_id,
                "status": status,
                "final_output": final_output,
                "events": events,
                "tool_observations": [],
                "capabilities": {},
            }
        ),
        encoding="utf-8",
    )
    return directory


# --------------------------------------------------------------------- #
# Fix 1: edits must reach the CUGA agent
# --------------------------------------------------------------------- #


def test_run_full_rollout_delivers_edited_skill_to_wrapper_harness():
    wrapper = RecordingWrapper()
    adapter = CugaAdapter(wrapper)
    adapter.register_candidate("base-1", {"skills/retrieval": "Use the catalog."})

    workspace = adapter.materialize_candidate("base-1", "attempt-1")
    adapter.apply_structured_edits(
        workspace,
        [
            ArtifactEdit(
                artifact_id="skills/retrieval",
                operation="replace",
                payload={"content": "Always cite the source id."},
            )
        ],
    )
    adapter.run_full_rollout(
        workspace, EvolutionTask(task_id="task-1", input_text="compute"), "rollout-1"
    )

    harness = wrapper.last_harness
    assert harness["input"] == "compute"
    assert harness["skills"] == {"retrieval": "Always cite the source id."}


def test_run_full_rollout_groups_policies_memory_and_instructions():
    wrapper = RecordingWrapper()
    adapter = CugaAdapter(wrapper)
    adapter.register_candidate(
        "base-1",
        {
            "skills/retrieval": "skill body",
            "policies/format": "policy body",
            "memory/facts": "memory body",
            "instructions": "be terse",
        },
    )
    workspace = adapter.materialize_candidate("base-1", "attempt-1")

    adapter.run_full_rollout(
        workspace, EvolutionTask(task_id="task-1", input_text="go"), "rollout-1"
    )

    harness = wrapper.last_harness
    assert harness["skills"] == {"retrieval": "skill body"}
    assert harness["policies"] == {"format": "policy body"}
    assert harness["memory"] == {"facts": "memory body"}
    assert harness["instructions"] == "be terse"


def test_register_candidate_rejects_unmappable_artifact_id():
    """A bad seed must fail at registration, not degrade into a no-op rollout."""
    adapter = CugaAdapter(RecordingWrapper())

    with pytest.raises(ValueError, match="mystery/thing"):
        adapter.register_candidate("base-1", {"mystery/thing": "body"})


def test_run_full_rollout_rejects_unmappable_artifact_from_wrapper():
    """The rollout path must also guard, since artifacts can bypass registration.

    Silently dropping an unmappable artifact would let the loop report a
    successful edit that never reached the agent.
    """
    wrapper = RecordingWrapper(artifacts={"mystery/thing": "body"})
    adapter = CugaAdapter(wrapper)
    workspace = adapter.materialize_candidate("base-1", "attempt-1")

    with pytest.raises(ValueError, match="mystery/thing"):
        adapter.run_full_rollout(
            workspace, EvolutionTask(task_id="task-1", input_text="go"), "rollout-1"
        )


def test_sibling_candidates_receive_independent_harnesses():
    """RHO seeds N candidates; each must run with its own artifacts."""
    wrapper = RecordingWrapper()
    adapter = CugaAdapter(wrapper)
    adapter.register_candidate("base-1", {"skills/s": "base body"})

    first = adapter.materialize_candidate("base-1", "attempt-1")
    second = adapter.materialize_candidate("base-1", "attempt-2")
    adapter.apply_structured_edits(
        first,
        [ArtifactEdit(artifact_id="skills/s", operation="replace", payload={"content": "first"})],
    )
    adapter.apply_structured_edits(
        second,
        [ArtifactEdit(artifact_id="skills/s", operation="replace", payload={"content": "second"})],
    )

    task = EvolutionTask(task_id="task-1", input_text="go")
    adapter.run_full_rollout(first, task, "rollout-1")
    assert wrapper.last_harness["skills"] == {"s": "first"}
    adapter.run_full_rollout(second, task, "rollout-2")
    assert wrapper.last_harness["skills"] == {"s": "second"}
    # The parent must remain untouched by either child's edits.
    assert adapter.read_artifacts("base-1", ("skills/s",)) == {"skills/s": "base body"}


# --------------------------------------------------------------------- #
# Fix 2: the rich DAG must reach the analyzer
# --------------------------------------------------------------------- #


def test_capture_trace_maps_actor_and_parent_from_rich_trace(tmp_path):
    trace_dir = _write_rich_trace(tmp_path / "trace")
    wrapper = RecordingWrapper(causal_trace_path=trace_dir)
    adapter = CugaAdapter(wrapper)
    adapter.register_candidate("base-1", {"skills/s": "body"})
    workspace = adapter.materialize_candidate("base-1", "attempt-1")

    result = adapter.run_full_rollout(
        workspace, EvolutionTask(task_id="task-1", input_text="go"), "rollout-1"
    )
    trace = adapter.capture_trace(result)

    assert len(trace.events) == 4
    by_id = {e.event_id: e for e in trace.events}
    assert by_id["graph:0"].actor_id == "prepare"
    assert by_id["graph:2"].actor_id == "sandbox"
    assert by_id["graph:2"].parent_event_id == "graph:0"
    assert by_id["graph:0"].parent_event_id is None
    assert {e.actor_id for e in trace.events if e.actor_id} == {"prepare", "sandbox"}


def test_capture_trace_never_dereferences_payload_blobs(tmp_path):
    """Refs must stay refs. Blob contents may hold raw prompts/state."""
    trace_dir = _write_rich_trace(tmp_path / "trace")
    wrapper = RecordingWrapper(causal_trace_path=trace_dir)
    adapter = CugaAdapter(wrapper)
    adapter.register_candidate("base-1", {"skills/s": "body"})
    workspace = adapter.materialize_candidate("base-1", "attempt-1")

    result = adapter.run_full_rollout(
        workspace, EvolutionTask(task_id="task-1", input_text="go"), "rollout-1"
    )
    trace = adapter.capture_trace(result)

    serialized = json.dumps([dict(e.payload) for e in trace.events])
    assert "SECRET-DO-NOT-LEAK" not in serialized
    assert "expected_answer" not in serialized
    assert "a" * 64 in serialized  # the ref itself survives


def test_capture_trace_falls_back_to_thin_events_without_rich_trace():
    wrapper = RecordingWrapper()  # no causal_trace_path
    adapter = CugaAdapter(wrapper)
    adapter.register_candidate("base-1", {"skills/s": "body"})
    workspace = adapter.materialize_candidate("base-1", "attempt-1")

    result = adapter.run_full_rollout(
        workspace, EvolutionTask(task_id="task-1", input_text="go"), "rollout-1"
    )
    trace = adapter.capture_trace(result)

    assert trace.task_id == "task-1"
    assert trace.events[0].event_id == "task-1:started"


# --------------------------------------------------------------------- #
# Fix 3: inventory must be non-empty
# --------------------------------------------------------------------- #


def test_artifact_inventory_reflects_registered_candidate():
    wrapper = RecordingWrapper()
    adapter = CugaAdapter(wrapper)
    adapter.register_candidate(
        "base-1", {"skills/retrieval": "body", "instructions": "be terse"}
    )

    inventory = adapter.artifact_inventory("base-1")

    assert {d.artifact_id for d in inventory} == {"skills/retrieval", "instructions"}
    assert all(d.writable for d in inventory)
    assert all(d.version_hash.startswith("sha256:") for d in inventory)


def test_artifact_inventory_tracks_edits_per_candidate():
    wrapper = RecordingWrapper()
    adapter = CugaAdapter(wrapper)
    adapter.register_candidate("base-1", {"skills/s": "before"})
    workspace = adapter.materialize_candidate("base-1", "attempt-1")

    before = {d.artifact_id: d.version_hash for d in adapter.artifact_inventory("base-1")}
    adapter.apply_structured_edits(
        workspace,
        [ArtifactEdit(artifact_id="skills/s", operation="replace", payload={"content": "after"})],
    )
    after = {
        d.artifact_id: d.version_hash for d in adapter.artifact_inventory(workspace.version)
    }

    assert before["skills/s"] != after["skills/s"]


# --------------------------------------------------------------------- #
# Integration: the full loop must not be inert
# --------------------------------------------------------------------- #


def test_edit_reaches_agent_and_blame_graph_has_real_node_actors(tmp_path):
    """seed -> edit -> rollout -> trace -> analyze -> non-empty blame graph."""
    trace_dir = _write_rich_trace(tmp_path / "trace", final_output="wrong answer")
    wrapper = RecordingWrapper(causal_trace_path=trace_dir, final_output="wrong answer")
    adapter = CugaAdapter(wrapper)
    adapter.register_candidate("base-1", {"skills/retrieval": "vague guidance"})

    workspace = adapter.materialize_candidate("base-1", "attempt-1")
    adapter.apply_structured_edits(
        workspace,
        [
            ArtifactEdit(
                artifact_id="skills/retrieval",
                operation="replace",
                payload={"content": "cite the source id"},
            )
        ],
    )
    task = EvolutionTask(
        task_id="task-1",
        input_text="go",
        expected_contract={"expected_substring": "four"},
    )
    result = adapter.run_full_rollout(workspace, task, "rollout-1")
    trace = adapter.capture_trace(result)
    analysis = FakeAnalyzerJudge().analyze(task, trace)

    # The edit actually reached the agent.
    assert wrapper.last_harness["skills"] == {"retrieval": "cite the source id"}
    # Blame is attributed to real CUGA graph nodes, not a synthetic placeholder.
    actors = {n.actor_id for n in analysis.blame_graph.nodes}
    assert actors == {"prepare", "sandbox"}
    assert "unknown" not in actors
    assert analysis.score == 0.0


# --------------------------------------------------------------------- #
# Regression against the real live-CUGA reference trace
# --------------------------------------------------------------------- #

_LIVE_TRACE = Path("data/traces/5d434903-bc26-4dc4-9229-8d886d2c6781")


@pytest.mark.skipif(
    not (_LIVE_TRACE / "causal-trace.json").is_file(),
    reason="live reference trace not present",
)
def test_live_reference_trace_yields_real_graph_actors_without_payload_leak():
    """Pin the mapping against a real 56-event CUGA trace, not a fixture.

    Guards two properties simultaneously: the full DAG reaches the analyzer
    with real CUGA node actors, and no payload blob body is inlined into core
    trace events (blobs may hold raw prompts, state, or expected answers).
    """
    wrapper = RecordingWrapper(causal_trace_path=_LIVE_TRACE, final_output="wrong")
    adapter = CugaAdapter(wrapper)
    adapter.register_candidate("base-1", {"skills/s": "body"})
    workspace = adapter.materialize_candidate("base-1", "attempt-1")

    task = EvolutionTask(
        task_id="complete-graph-demo",
        input_text="go",
        expected_contract={"expected_substring": "NEVER_MATCHES"},
    )
    result = adapter.run_full_rollout(workspace, task, "rollout-1")
    trace = adapter.capture_trace(result)

    source = json.loads((_LIVE_TRACE / "causal-trace.json").read_text())
    assert len(trace.events) == len(source["events"]) == 56
    assert sum(1 for e in trace.events if e.parent_event_id) == 52

    actors = {e.actor_id for e in trace.events if e.actor_id}
    assert "sandbox" in actors and "prepare" in actors
    assert len(actors) >= 5

    nodes = FakeAnalyzerJudge().analyze(task, trace).blame_graph.nodes
    assert len(nodes) >= 5
    assert not any(n.actor_id == "unknown" for n in nodes)
    assert sum(n.blame for n in nodes) == pytest.approx(1.0)

    # Refs stay bare hashes; only tool observations may exceed hash length.
    ref_keys = {"state_before_ref", "state_after_ref", "messages_ref", "response_ref"}
    for event in trace.events:
        for key, value in dict(event.payload).items():
            text = value if isinstance(value, str) else json.dumps(value)
            if key in ref_keys:
                assert len(text) == 64, f"{key} on {event.event_id} is not a bare hash"
            elif key != "tool_call":
                assert len(text) <= 64, f"oversized payload {key} on {event.event_id}"
