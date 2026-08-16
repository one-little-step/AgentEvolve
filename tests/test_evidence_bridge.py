"""(task, traces) -> RolloutGroupReport bridge.

The orchestrator holds `(EvolutionTask, ExecutionTrace)`; the report-based
`AnalyzerJudge` protocol consumes a `RolloutGroupReport`. This bridge converts
between them, and -- the load-bearing part -- strips the answer key on the way.

The contamination boundary matters more here than for the editor: an LLM
analyzer that can see `expected_substring` will "diagnose" by reading the
answer key rather than by causal reasoning, and it will look convincing while
doing it.
"""
from __future__ import annotations

import pytest

from agent_evolve.core.analysis import RolloutGroupReport
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace, TraceEvent
from agent_evolve.core.evidence import (
    contamination_terms_from,
    rollout_group_report,
)


def _task(**overrides) -> EvolutionTask:
    kwargs = {
        "task_id": "task-1",
        "input_text": "what is 2+2",
        "expected_contract": {"expected_substring": "four"},
    }
    kwargs.update(overrides)
    return EvolutionTask(**kwargs)


def _trace(trace_id: str = "trace-1", events=(), final_output="four") -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        candidate_id="cand-1",
        task_id="task-1",
        events=tuple(events),
        final_output=final_output,
        status="success",
    )


def _event(event_id="e1", kind="tool_call", actor_id="planner", payload=None):
    return TraceEvent(
        event_id=event_id,
        kind=kind,
        actor_id=actor_id,
        parent_event_id=None,
        payload=payload if payload is not None else {},
    )


# ---------------------------------------------------------------------- #
# Shape
# ---------------------------------------------------------------------- #
def test_bridge_produces_a_report_for_a_group_of_traces():
    traces = [_trace("trace-1"), _trace("trace-2"), _trace("trace-3")]

    report = rollout_group_report(_task(), traces)

    assert isinstance(report, RolloutGroupReport)
    assert report.candidate_id == "cand-1"
    assert report.task_id == "task-1"
    assert report.trace_refs == ("trace-1", "trace-2", "trace-3")


def test_single_trace_is_accepted_as_a_group_of_one():
    """The orchestrator's current path has one trace; it must not need a list."""
    report = rollout_group_report(_task(), _trace("trace-9"))

    assert report.trace_refs == ("trace-9",)


def test_rollout_ids_default_to_trace_ids_but_can_be_supplied():
    traces = [_trace("trace-1"), _trace("trace-2")]

    default = rollout_group_report(_task(), traces)
    assert default.rollout_ids == ("trace-1", "trace-2")

    explicit = rollout_group_report(
        _task(), traces, rollout_ids=("roll-a", "roll-b")
    )
    assert explicit.rollout_ids == ("roll-a", "roll-b")


def test_mismatched_rollout_ids_length_is_rejected():
    with pytest.raises(ValueError, match="rollout_ids"):
        rollout_group_report(_task(), [_trace("t1")], rollout_ids=("a", "b"))


def test_empty_trace_group_is_rejected():
    with pytest.raises(ValueError, match="at least one trace"):
        rollout_group_report(_task(), [])


def test_traces_from_different_candidates_are_rejected():
    """A rollout GROUP is one candidate on one task; mixing breaks variance math."""
    a = _trace("t1")
    b = ExecutionTrace(
        trace_id="t2",
        candidate_id="cand-OTHER",
        task_id="task-1",
        events=(),
        final_output="four",
        status="success",
    )
    with pytest.raises(ValueError, match="candidate"):
        rollout_group_report(_task(), [a, b])


def test_traces_from_different_tasks_are_rejected():
    a = _trace("t1")
    b = ExecutionTrace(
        trace_id="t2",
        candidate_id="cand-1",
        task_id="task-OTHER",
        events=(),
        final_output="four",
        status="success",
    )
    with pytest.raises(ValueError, match="task"):
        rollout_group_report(_task(), [a, b])


# ---------------------------------------------------------------------- #
# The answer key must never reach the analyzer
# ---------------------------------------------------------------------- #
def test_final_output_never_appears_in_the_report():
    trace = _trace(final_output="the answer is four")

    report = rollout_group_report(_task(), [trace])

    assert "the answer is four" not in repr(report)


def test_expected_contract_values_never_appear_in_the_report():
    task = _task(expected_contract={"expected_substring": "SENTINEL_ANSWER"})
    trace = _trace(
        events=[_event(payload={"tool": "search", "result": "SENTINEL_ANSWER here"})]
    )

    report = rollout_group_report(task, [trace])

    assert "SENTINEL_ANSWER" not in repr(report)


def test_nested_expected_contract_shapes_are_still_guarded():
    """The guard must see contract values at any depth, not just flat mappings."""
    task = _task(
        expected_contract={
            "grader": {"expected_any": ["DEEP_SENTINEL", "other"]},
        }
    )
    trace = _trace(
        events=[_event(payload={"tool": "search", "result": "DEEP_SENTINEL"})]
    )

    report = rollout_group_report(task, [trace])

    assert "DEEP_SENTINEL" not in repr(report)


def test_contaminated_payload_is_reported_as_redacted_not_silently_dropped():
    """An analyzer must be able to tell 'no evidence' from 'evidence withheld'."""
    task = _task(expected_contract={"expected_substring": "SENTINEL"})
    trace = _trace(events=[_event(payload={"result": "SENTINEL"})])

    report = rollout_group_report(task, [trace])

    events = report.sanitized_evidence[0]["events"]
    assert events[0]["payload_redacted"] is True
    assert events[0]["payload"] == {}


def test_blob_refs_are_stripped_from_payloads():
    """Blob bodies carry raw prompts and AgentState; refs must not be forwarded."""
    trace = _trace(
        events=[
            _event(payload={"tool": "search", "messages_ref": "blob:abc", "response_ref": "blob:def"})
        ]
    )

    report = rollout_group_report(_task(), [trace])

    payload = report.sanitized_evidence[0]["events"][0]["payload"]
    assert "messages_ref" not in payload
    assert "response_ref" not in payload
    assert payload["tool"] == "search"


def test_a_task_with_no_expected_contract_yields_no_terms_and_keeps_evidence():
    """No contract is a legitimate case (unlabeled task), not a guard failure."""
    task = _task(expected_contract={})
    trace = _trace(events=[_event(payload={"tool": "search", "result": "anything"})])

    report = rollout_group_report(task, [trace])

    payload = report.sanitized_evidence[0]["events"][0]["payload"]
    assert payload["result"] == "anything"


# ---------------------------------------------------------------------- #
# Evidence the analyzer legitimately needs
# ---------------------------------------------------------------------- #
def test_task_input_text_is_exposed():
    """Safe by construction: the agent under test already saw it."""
    report = rollout_group_report(_task(input_text="what is 2+2"), [_trace()])

    assert report.sanitized_evidence[0]["task_input"] == "what is 2+2"


def test_event_metadata_and_actors_are_exposed():
    trace = _trace(
        events=[
            _event("e1", "tool_call", "planner"),
            _event("e2", "llm_call_start", "coder"),
        ]
    )

    report = rollout_group_report(_task(), [trace])

    evidence = report.sanitized_evidence[0]
    assert evidence["trace_id"] == "trace-1"
    assert evidence["status"] == "success"
    assert evidence["actors"] == ("planner", "coder")
    kinds = [e["kind"] for e in evidence["events"]]
    assert kinds == ["tool_call", "llm_call_start"]


def test_non_tool_call_payloads_are_withheld_but_event_metadata_survives():
    """Metadata locates the faulty node; non-tool payloads carry prompt bodies."""
    trace = _trace(events=[_event("e1", "llm_call_start", "coder", {"prompt": "secret"})])

    report = rollout_group_report(_task(), [trace])

    event = report.sanitized_evidence[0]["events"][0]
    assert event["kind"] == "llm_call_start"
    assert event["actor_id"] == "coder"
    assert event["payload"] == {}
    assert "secret" not in repr(report)


def test_each_trace_in_the_group_gets_its_own_evidence_entry():
    traces = [
        _trace("trace-1", events=[_event("e1")]),
        _trace("trace-2", events=[_event("e2"), _event("e3")]),
    ]

    report = rollout_group_report(_task(), traces)

    assert len(report.sanitized_evidence) == 2
    assert report.sanitized_evidence[0]["trace_id"] == "trace-1"
    assert len(report.sanitized_evidence[1]["events"]) == 2


def test_event_limit_bounds_prompt_size_per_trace():
    trace = _trace(events=[_event(f"e{i}") for i in range(200)])

    report = rollout_group_report(_task(), [trace], max_events_per_trace=10)

    assert len(report.sanitized_evidence[0]["events"]) == 10
    assert report.sanitized_evidence[0]["events_truncated"] is True


def test_untruncated_evidence_is_marked_as_such():
    trace = _trace(events=[_event(f"e{i}") for i in range(3)])

    report = rollout_group_report(_task(), [trace], max_events_per_trace=10)

    assert report.sanitized_evidence[0]["events_truncated"] is False


def test_redaction_count_is_surfaced_for_observability():
    task = _task(expected_contract={"expected_substring": "SENTINEL"})
    trace = _trace(
        events=[
            _event("e1", payload={"result": "SENTINEL"}),
            _event("e2", payload={"result": "clean"}),
        ]
    )

    report = rollout_group_report(task, [trace])

    assert report.sanitized_evidence[0]["redaction_count"] == 1


# ---------------------------------------------------------------------- #
# Shared guard primitives
# ---------------------------------------------------------------------- #
def test_contamination_terms_walks_nested_contract_structures():
    task = _task(
        expected_contract={
            "expected_substring": "alpha",
            "expected_any": ["beta", "gamma"],
            "grader": {"expected": "delta"},
        }
    )

    terms = contamination_terms_from(task)

    assert set(terms) >= {"alpha", "beta", "gamma", "delta"}


def test_contamination_terms_skips_terms_too_short_to_scan_safely():
    """Short terms match incidental text and would redact real evidence."""
    task = _task(expected_contract={"expected_substring": "a"})

    assert contamination_terms_from(task) == ()


def test_adapter_and_core_share_one_guard_implementation():
    """Two copies of a security guard is one copy that will drift."""
    from agent_evolve.adapters import cuga_editor_evidence

    assert (
        cuga_editor_evidence.contamination_terms_from is contamination_terms_from
    )


def test_core_evidence_imports_no_agent_implementation():
    """Check real import statements, not prose: docstrings may name adapters."""
    import ast

    import agent_evolve.core.evidence as mod

    tree = ast.parse(open(str(mod.__file__), encoding="utf-8").read())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")

    assert imported, "expected at least one import to be checked"
    for name in imported:
        assert not name.startswith("cuga"), f"core imported {name}"
        assert "adapters" not in name, f"core imported {name}"
