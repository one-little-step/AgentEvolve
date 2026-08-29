"""S4-9: structured surface absence is first-class evidence.

Three live runs 2026-08-27 (``terminal_output/live-test-judge2/``) hit a wall:
every analyzed failure had ``blamed_actors[].artifacts == []`` because the
analyzer's evidence carries only ``tool_call`` payloads, and the dominant L1
failure shape is "the agent never exercised any surface" (no skill loaded, no
policy consulted, no memory fetched).  Empty artifact use *is* evidence -- a
guide who was paid and never showed up is himself the finding -- and it must be
able to justify CREATING a brand-new artifact, not only editing an existing one.

This file pins the absence-evidence chain:

1. ``surface_activity``: a per-trace summary of which artifact ids each surface
   actually exercised, derived from ``tool_call`` payloads (ids + counts only,
   never contents, contamination guard still applied), present even when the
   summary is EMPTY -- explicit absence, never a silently missing key.
2. ``CausalFinding.absent_surfaces`` / ``CausalAnalysis.absent_surfaces``: the
   analyzer's way to say "this failed because nothing was in play"; forwarded
   losslessly by :func:`analysis_from_finding`.
3. ``build_issue``: a finding that names ``absent_surfaces`` with empty
   attributed artifacts is actionable -- it must not be dropped -- and the
   editor's issue contract carries the absence so ``stage_create`` on a fresh
   artifact id is a first-class answer to it.

The create path itself (``stage_create`` / ``apply_edits`` create-on-unknown-id)
already exists and is not re-tested here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.analysis import RolloutGroupReport  # noqa: E402
from agent_evolve.core.blame import (  # noqa: E402
    BlameGraph,
    BlameNode,
    CausalAnalysis,
    CausalFinding,
    analysis_from_finding,
)
from agent_evolve.core.contracts import (  # noqa: E402
    ArtifactDescriptor,
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)
from agent_evolve.core.evidence import rollout_group_report  # noqa: E402
from agent_evolve.core.issues import build_issue  # noqa: E402


def _task(**overrides) -> EvolutionTask:
    kwargs = {
        "task_id": "task-1",
        "input_text": "what is the capital of the country described",
        "expected_contract": {"expected_substring": "lisbon"},
    }
    kwargs.update(overrides)
    return EvolutionTask(**kwargs)


def _trace(trace_id="trace-1", events=(), final_output="paris") -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        candidate_id="cand-1",
        task_id="task-1",
        events=tuple(events),
        final_output=final_output,
        status="success",
    )


def _event(event_id="e1", kind="tool_call", actor_id="sandbox", payload=None):
    return TraceEvent(
        event_id=event_id,
        kind=kind,
        actor_id=actor_id,
        parent_event_id=None,
        payload=payload if payload is not None else {},
    )


def _skill_load_call(skill_id: str) -> TraceEvent:
    """A CUGA-shaped load_skill tool_call payload (mirrors cuga 0.3.1)."""
    return _event(
        payload={
            "tool_call": {
                "name": "load_skill",
                "app_name": "skills",
                "arguments": {"name": skill_id},
                "result": "skill body",
                "error": None,
            }
        }
    )


def _descriptor(artifact_id: str, writable: bool = True) -> ArtifactDescriptor:
    return ArtifactDescriptor(
        artifact_id=artifact_id,
        kind="skills",
        format="markdown",
        version_hash="sha256:00" * 5,
        readable=True,
        writable=writable,
        merge_strategy="replace",
    )


def _finding(
    *,
    absent_surfaces: tuple[str, ...] = (),
    artifacts: tuple[str, ...] = (),
    evidence_refs: tuple[str, ...] = ("ev",),
    status: str = "observed",
) -> CausalFinding:
    nodes = (
        (BlameNode(actor_id="call_model", blame=0.9, artifacts=artifacts),)
        if artifacts
        else (BlameNode(actor_id="call_model", blame=0.9),)
    )
    return CausalFinding(
        verdict_id="v-1",
        candidate_id="cand-1",
        task_id="task-1",
        trace_id="trace-1",
        valence=1,
        status=status,
        mechanism_description="call_model never loaded any guidance before answering",
        mechanism_cluster_id="c0",
        severity=0.8,
        confidence=0.7,
        blame_graph=BlameGraph(nodes=nodes),
        evidence_refs=evidence_refs,
        absent_surfaces=absent_surfaces,
        rationale="surface summary shows zero loads",
    )


# ---------------------------------------------------------------------- #
# 1. surface_activity in the sanitized evidence
# ---------------------------------------------------------------------- #
def test_surface_activity_present_even_when_empty():
    """Explicit absence: the key exists with empty members, never missing."""
    report = rollout_group_report(_task(), _trace())
    evidence = report.sanitized_evidence[0]
    assert evidence["surface_activity"] == {"skills": [], "policies": [], "memory": []}


def test_surface_activity_counts_skill_loads_from_tool_call_payloads():
    report = rollout_group_report(_task(), _trace(events=(_skill_load_call("web-research"),)))
    evidence = report.sanitized_evidence[0]
    assert evidence["surface_activity"]["skills"] == ["web-research"]


def test_surface_activity_ignores_non_load_tools():
    other = _event(
        payload={
            "tool_call": {
                "name": "web_search",
                "app_name": "web",
                "arguments": {"q": "capital"},
                "result": "stuff",
                "error": None,
            }
        }
    )
    report = rollout_group_report(_task(), _trace(events=(other,)))
    evidence = report.sanitized_evidence[0]
    assert evidence["surface_activity"] == {"skills": [], "policies": [], "memory": []}


def test_surface_activity_redacts_load_ids_matching_the_answer_key():
    """A load id that carries answer-key material must be withheld, not leaked."""
    skill = _skill_load_call("lisbon")
    report = rollout_group_report(_task(), _trace(events=(skill,)))
    evidence = report.sanitized_evidence[0]
    assert evidence["surface_activity"]["skills"] == []
    # The id is withheld twice over: the payload carrying it is redacted AND
    # the surface summary withholds the load. Both channels count one.
    assert evidence["redaction_count"] == 2


def test_surface_activity_dedupes_and_sorts_ids():
    calls = tuple(
        _skill_load_call(s) for s in ("z-skill", "a-skill", "a-skill", "m-skill")
    )
    report = rollout_group_report(_task(), _trace(events=calls))
    evidence = report.sanitized_evidence[0]
    assert evidence["surface_activity"]["skills"] == ["a-skill", "m-skill", "z-skill"]


def test_surface_activity_reads_event_beyond_the_trim_window():
    """Surface loads past max_events (e.g. event 60 of 70) still count.

    The events list in the evidence is trimmed, but the surface summary is
    computed over the FULL trace: a load that happened late still proves the
    surface was exercised.
    """
    late = _skill_load_call("late-skill")
    filler = tuple(
        _event(event_id=f"f{i}", kind="graph_node_start", actor_id="prepare")
        for i in range(60)
    )
    report = rollout_group_report(_task(), _trace(events=(*filler, late)), max_events_per_trace=50)
    evidence = report.sanitized_evidence[0]
    assert evidence["events_truncated"] is True
    assert evidence["surface_activity"]["skills"] == ["late-skill"]


# ---------------------------------------------------------------------- #
# 2. absence on the finding / analysis contract
# ---------------------------------------------------------------------- #
def test_finding_accepts_absent_surfaces():
    f = _finding(absent_surfaces=("skills", "memory"))
    assert f.absent_surfaces == ("skills", "memory")


def test_finding_absent_surfaces_must_be_known_surface_names():
    with pytest.raises(Exception):
        _finding(absent_surfaces=("vibes",))


def test_finding_observed_requires_evidence_for_absence_too():
    """An observed finding claiming absence must still be trace-backed overall."""
    with pytest.raises(Exception):
        _finding(absent_surfaces=("skills",), evidence_refs=())


def test_analysis_from_finding_forwards_absent_surfaces():
    analysis = analysis_from_finding(_finding(absent_surfaces=("skills",)), score=0.0)
    assert analysis.absent_surfaces == ("skills",)


def test_absent_surfaces_survive_into_causal_analysis_mechanism_string():
    """The analysis keeps the signal without inventing prose (no fabricated data)."""
    analysis = analysis_from_finding(_finding(absent_surfaces=("policies",)), score=0.0)
    assert analysis.absent_surfaces == ("policies",)
    assert "policies" not in analysis.mechanism


# ---------------------------------------------------------------------- #
# 3. build_issue: absence is actionable
# ---------------------------------------------------------------------- #
def test_absence_finding_without_artifacts_is_actionable():
    """THE S4-9 regression: absence + empty artifacts must not be dropped."""
    inventory = [_descriptor("skills/existing")]
    issue = build_issue(_finding(absent_surfaces=("skills",)), inventory)
    assert issue is not None


def test_absence_issue_attributes_declared_unloaded_artifacts():
    inventory = [_descriptor("skills/existing")]
    issue = build_issue(_finding(absent_surfaces=("skills",)), inventory)
    assert issue is not None
    assert issue.writable_artifact_ids == ("skills/existing",)


def test_absence_issue_ignores_artifacts_of_unnamed_surfaces():
    """Absent on 'memory' must not authorize editing a 'skills' artifact."""
    inventory = [_descriptor("skills/existing"), _descriptor("memory/kept")]
    issue = build_issue(_finding(absent_surfaces=("memory",)), inventory)
    assert issue is not None
    assert issue.writable_artifact_ids == ("memory/kept",)


def test_absence_issue_carries_the_absence_signal_for_the_editor():
    inventory = [_descriptor("skills/existing")]
    issue = build_issue(_finding(absent_surfaces=("skills",)), inventory)
    assert issue is not None
    assert issue.absent_surfaces == ("skills",)


def test_absence_with_no_declared_artifacts_still_yields_none():
    """No writable member on the absent surface: nothing to edit, still no issue."""
    inventory = [_descriptor("skills/existing")]
    issue = build_issue(_finding(absent_surfaces=("memory",)), inventory)
    assert issue is None


def test_non_absence_finding_with_empty_artifacts_still_rejected():
    """The S4-9 fix must not open the floodgates: no absence, no attribution, no issue."""
    inventory = [_descriptor("skills/existing")]
    issue = build_issue(_finding(), inventory)
    assert issue is None
