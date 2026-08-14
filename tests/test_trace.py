"""Validation and conversion tests for the persisted causal trace schema.

These tests exercise only the agent-neutral models in
``agent_evolve.core.trace``; no CUGA, LangChain, or adapter import is allowed.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_evolve.core.contracts import ExecutionTrace, TraceEvent
from agent_evolve.core.trace import (
    CausalEvent,
    CausalTrace,
    FacilityCapability,
    StateSnapshot,
    ToolObservation,
    canonical_json,
)


def test_causal_trace_maps_only_minimal_adapter_fields():
    trace = CausalTrace(
        run_id="run-1",
        task_id="task-1",
        thread_id="run-1",
        thread_id_source="wrapper_generated",
        harness_version="h1",
        status="success",
        final_output="answer",
        events=(),
        checkpoints=(),
        tool_observations=(),
        capabilities={"graph_history": FacilityCapability(status="unavailable_no_checkpointer")},
    )

    minimal = trace.to_execution_trace(candidate_id="candidate-1", trace_id="rollout-1")

    assert isinstance(minimal, ExecutionTrace)
    assert minimal.trace_id == "rollout-1"
    assert minimal.candidate_id == "candidate-1"
    assert minimal.task_id == "task-1"
    assert minimal.final_output == "answer"
    assert minimal.status == "success"
    assert minimal.checkpoint_ids == ()
    assert minimal.events == ()


def test_to_execution_trace_maps_events_and_replay_safe_snapshots():
    trace = CausalTrace(
        run_id="run-1",
        task_id="task-1",
        status="success",
        final_output="done",
        events=(
            CausalEvent(event_id="e1", sequence=0, kind="run_started", payload={"a": 1}),
            CausalEvent(event_id="e2", sequence=1, kind="run_completed"),
        ),
        checkpoints=(
            StateSnapshot(sequence=0, checkpoint_id="ck-1", replay_safe=True),
            StateSnapshot(sequence=1, checkpoint_id="ck-2", replay_safe=False),
        ),
    )

    minimal = trace.to_execution_trace(candidate_id="c", trace_id="t")

    assert minimal.events == (
        TraceEvent(event_id="e1", kind="run_started", actor_id=None, parent_event_id=None, payload={"a": 1}),
        TraceEvent(event_id="e2", kind="run_completed", actor_id=None, parent_event_id=None, payload={}),
    )
    assert minimal.checkpoint_ids == ("ck-1",)


def test_tool_observation_rejects_replay_eligible_truncation():
    with pytest.raises(ValidationError, match="truncated"):
        ToolObservation(
            sequence=0,
            tool_name="lookup",
            canonical_arguments='{"q":"x"}',
            result="partial",
            truncated=True,
            original_bytes=2_000_000,
            retained_bytes=1_048_576,
            content_digest="sha256:abc",
            replay_eligible=True,
        )


def test_tool_observation_rejects_replay_eligible_withheld():
    with pytest.raises(ValidationError, match="withheld"):
        ToolObservation(
            sequence=0,
            tool_name="save_note",
            canonical_arguments='{"note":"x"}',
            withheld_reason="high_risk_tool",
            replay_eligible=True,
        )


def test_tool_observation_rejects_replay_eligible_error():
    with pytest.raises(ValidationError, match="error"):
        ToolObservation(
            sequence=0,
            tool_name="lookup",
            canonical_arguments="{}",
            error="boom",
            replay_eligible=True,
        )


def test_tool_observation_accepts_clean_replay_eligible():
    observation = ToolObservation(
        sequence=0,
        tool_name="lookup",
        canonical_arguments='{"q":"x"}',
        result={"value": "x"},
        original_bytes=10,
        retained_bytes=10,
        content_digest="sha256:abc",
        replay_eligible=True,
    )
    assert observation.replay_eligible is True
    assert observation.truncated is False


def test_facility_capability_is_immutable_and_forbids_extra():
    capability = FacilityCapability(status="captured")

    assert capability.status == "captured"
    assert capability.reason is None

    with pytest.raises(ValidationError):
        FacilityCapability(status="captured", bogus="x")


def test_canonical_json_sorts_mapping_keys_and_preserves_list_order():
    assert canonical_json({"b": [2, 1], "a": {"y": 2, "x": 1}}) == '{"a":{"x":1,"y":2},"b":[2,1]}'


def test_canonical_json_rejects_non_finite_float():
    with pytest.raises(ValueError, match="finite"):
        canonical_json({"value": float("nan")})

    with pytest.raises(ValueError, match="finite"):
        canonical_json({"value": float("inf")})


def test_canonical_json_is_deterministic_across_insertion_order():
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_canonical_json_rejects_unsupported_type():
    with pytest.raises(ValueError):
        canonical_json({"value": object()})
