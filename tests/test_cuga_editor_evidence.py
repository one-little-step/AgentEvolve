"""Editor evidence boundary and contamination guard (spec §8).

The guard CONSUMES expected_contract to build its term list but never SHOWS it.
Key-name denylisting alone is insufficient: sanitize_payload matches keys such
as 'expected_answer', not an expected answer appearing as free text inside a
tool result string.
"""
from __future__ import annotations

from agent_evolve.adapters.cuga_editor_evidence import (
    EvidenceView,
    contamination_terms_from,
)
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace, TraceEvent

_SECRET = "token-a"


def _task() -> EvolutionTask:
    return EvolutionTask(
        task_id="task-a",
        input_text="produce the A capability",
        expected_contract={"expected_substring": _SECRET},
    )


def _analysis() -> CausalAnalysis:
    return CausalAnalysis(
        mechanism="skill never loaded",
        severity=0.9,
        score=0.0,
        blame_graph=BlameGraph(
            nodes=(
                BlameNode(actor_id="call_model", blame=0.7, artifacts=("skills/retrieval",)),
                BlameNode(actor_id="prepare", blame=0.3, artifacts=()),
            )
        ),
    )


def _trace(events: tuple[TraceEvent, ...] | None = None) -> ExecutionTrace:
    if events is None:
        events = (
            TraceEvent(
                event_id="graph:1",
                kind="llm_call",
                actor_id="call_model",
                parent_event_id="graph:0",
                payload={"messages_ref": "a" * 64, "sequence": 1},
            ),
            TraceEvent(
                event_id="graph:2",
                kind="tool_call",
                actor_id="sandbox",
                parent_event_id="graph:1",
                payload={"name": "run_command", "result": "exit 0"},
            ),
        )
    return ExecutionTrace(
        trace_id="trace-1",
        candidate_id="cand-1",
        task_id="task-a",
        events=events,
        final_output=f"the answer is {_SECRET}",
        status="completed",
    )


def _view(trace: ExecutionTrace | None = None) -> EvidenceView:
    task = _task()
    return EvidenceView(
        analysis=_analysis(),
        trace=trace if trace is not None else _trace(),
        task=task,
        contamination_terms=contamination_terms_from(task),
    )


# ------------------------------------------------------------------ #
# term extraction
# ------------------------------------------------------------------ #
def test_contamination_terms_extracts_string_values() -> None:
    assert contamination_terms_from(_task()) == (_SECRET,)


def test_contamination_terms_ignores_short_and_nonstring_values() -> None:
    task = EvolutionTask(
        task_id="t",
        input_text="i",
        expected_contract={"expected_substring": "ab", "threshold": 0.5},
    )
    # 2-char terms are too short to scan safely (false positives everywhere).
    assert contamination_terms_from(task) == ()


def test_contamination_terms_finds_strings_nested_in_containers() -> None:
    """A flat .values() scan yields no terms for these shapes, and a guard with
    no terms passes every payload through -- so the extractor must recurse."""
    nested = EvolutionTask(
        task_id="t",
        input_text="i",
        expected_contract={
            "expected_any": ["token-alpha", "token-beta"],
            "grader": {"inner": {"expected": "token-gamma"}},
            "as_tuple": ("token-delta",),
        },
    )
    assert set(contamination_terms_from(nested)) == {
        "token-alpha",
        "token-beta",
        "token-gamma",
        "token-delta",
    }


def test_contamination_terms_are_deduplicated_and_ordered() -> None:
    task = EvolutionTask(
        task_id="t",
        input_text="i",
        expected_contract={"a": "token-alpha", "b": ["token-alpha", "token-beta"]},
    )
    assert contamination_terms_from(task) == ("token-alpha", "token-beta")


def test_guard_redacts_a_payload_matching_a_nested_contract_term() -> None:
    """End-to-end: the nested term must actually drive redaction."""
    task = EvolutionTask(
        task_id="task-a",
        input_text="produce the A capability",
        expected_contract={"grader": {"expected_any": ["deep-secret"]}},
    )
    dirty = (
        TraceEvent(
            event_id="graph:9",
            kind="tool_call",
            actor_id="sandbox",
            parent_event_id=None,
            payload={"result": "the value is deep-secret"},
        ),
    )
    view = EvidenceView(
        analysis=_analysis(),
        trace=_trace(dirty),
        task=task,
        contamination_terms=contamination_terms_from(task),
    )
    events = view.events()
    assert events[0]["payload"] == {}
    assert events[0]["payload_redacted"] is True


# ------------------------------------------------------------------ #
# what the editor may see
# ------------------------------------------------------------------ #
def test_mechanism_exposes_description_and_severity() -> None:
    assert _view().mechanism() == {
        "mechanism": "skill never loaded",
        "severity": 0.9,
    }


def test_blamed_actors_are_sorted_by_blame_descending() -> None:
    actors = _view().blamed_actors()
    assert [a["actor_id"] for a in actors] == ["call_model", "prepare"]
    assert actors[0]["artifacts"] == ("skills/retrieval",)


def test_task_input_exposes_input_text() -> None:
    assert _view().task_input() == "produce the A capability"


def test_actors_lists_distinct_trace_actors() -> None:
    assert _view().actors() == ("call_model", "sandbox")


# ------------------------------------------------------------------ #
# what the editor may NOT see
# ------------------------------------------------------------------ #
def test_events_strip_ref_payload_keys() -> None:
    events = _view().events()
    llm = next(e for e in events if e["kind"] == "llm_call")
    assert "messages_ref" not in llm["payload"]


def test_events_keep_tool_call_payloads() -> None:
    events = _view().events(kind="tool_call")
    assert events[0]["payload"]["name"] == "run_command"


def test_contamination_guard_drops_payload_containing_expected_value() -> None:
    dirty = (
        TraceEvent(
            event_id="graph:3",
            kind="tool_call",
            actor_id="sandbox",
            parent_event_id=None,
            payload={"name": "run_command", "result": f"found {_SECRET} here"},
        ),
    )
    view = _view(_trace(dirty))
    events = view.events()
    assert events[0]["payload"] == {}
    assert events[0]["payload_redacted"] is True
    assert view.redaction_count == 1


def test_no_view_output_contains_the_expected_value() -> None:
    """Full leak audit across every exposed surface."""
    view = _view()
    blob = repr(
        (
            view.mechanism(),
            view.blamed_actors(),
            view.task_input(),
            view.actors(),
            view.events(limit=100),
        )
    )
    assert _SECRET not in blob


def test_no_view_output_contains_the_final_output() -> None:
    view = _view()
    blob = repr((view.mechanism(), view.events(limit=100), view.task_input()))
    assert "the answer is" not in blob


# ------------------------------------------------------------------ #
# filtering and bounding
# ------------------------------------------------------------------ #
def test_events_filter_by_kind() -> None:
    events = _view().events(kind="tool_call")
    assert [e["kind"] for e in events] == ["tool_call"]


def test_events_filter_by_actor() -> None:
    events = _view().events(actor_id="call_model")
    assert [e["actor_id"] for e in events] == ["call_model"]


def test_events_respect_the_limit() -> None:
    assert len(_view().events(limit=1)) == 1


def test_events_preserve_dag_fields() -> None:
    llm = _view().events(kind="llm_call")[0]
    assert llm["event_id"] == "graph:1"
    assert llm["parent_event_id"] == "graph:0"


# ------------------------------------------------------------------ #
# S4-9: measured absence reaches the editor view
# ------------------------------------------------------------------ #
def test_absent_surfaces_empty_by_default() -> None:
    assert _view().absent_surfaces() == ()


def test_absent_surfaces_forwarded_from_analysis() -> None:
    task = _task()
    analysis = CausalAnalysis(
        mechanism="no guidance was ever loaded to steer the run",
        severity=0.9,
        score=0.0,
        blame_graph=BlameGraph(nodes=()),
        absent_surfaces=("skills", "memory"),
    )
    view = EvidenceView(
        analysis=analysis,
        trace=_trace(),
        task=task,
        contamination_terms=contamination_terms_from(task),
    )
    assert view.absent_surfaces() == ("skills", "memory")
