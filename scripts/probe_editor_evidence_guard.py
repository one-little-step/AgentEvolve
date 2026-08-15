"""Adversarial probe: does the evidence contamination guard actually fail closed?

Not part of the suite; run manually to validate Task 3 beyond its own tests.
"""
from __future__ import annotations

from agent_evolve.adapters.cuga_editor_evidence import (
    EvidenceView,
    contamination_terms_from,
)
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace, TraceEvent

SECRET = "token-a"
TASK = EvolutionTask(
    task_id="t", input_text="i", expected_contract={"expected_substring": SECRET}
)
ANALYSIS = CausalAnalysis(
    mechanism="m",
    severity=0.5,
    score=0.0,
    blame_graph=BlameGraph(nodes=(BlameNode(actor_id="a", blame=1.0, artifacts=()),)),
)


def view(events, terms=None):
    trace = ExecutionTrace(
        trace_id="x",
        candidate_id="c",
        task_id="t",
        events=events,
        final_output=f"ans {SECRET}",
        status="completed",
    )
    return EvidenceView(
        analysis=ANALYSIS,
        trace=trace,
        task=TASK,
        contamination_terms=contamination_terms_from(TASK) if terms is None else terms,
    )


def main() -> None:
    deep = (
        TraceEvent(
            event_id="1",
            kind="tool_call",
            actor_id="s",
            parent_event_id=None,
            payload={"name": "run", "result": {"nested": {"deep": [f"got {SECRET}"]}}},
        ),
    )
    v = view(deep)
    ev = v.events()
    print("1 nested-leak redacted:", ev[0]["payload"] == {} and ev[0]["payload_redacted"])

    blob = repr(
        (v.mechanism(), v.blamed_actors(), v.task_input(), v.actors(), v.events(limit=100))
    )
    print("2 no secret on any surface:", SECRET not in blob)

    llm = (
        TraceEvent(
            event_id="2",
            kind="llm_call",
            actor_id="m",
            parent_event_id=None,
            payload={"messages_ref": "a" * 64, "raw_prompt": f"has {SECRET}"},
        ),
    )
    e2 = view(llm).events()
    print("3 llm payload emptied:", e2[0]["payload"] == {})

    clean = (
        TraceEvent(
            event_id="3",
            kind="tool_call",
            actor_id="s",
            parent_event_id=None,
            payload={"name": "run_command", "result": "exit 0"},
        ),
    )
    print("4 clean payload preserved:", view(clean).events()[0]["payload"]["name"] == "run_command")

    # Fails OPEN when no terms are supplied -- the caller MUST pass terms.
    open_view = view(clean, terms=())
    print("5 empty-terms passes through (fails open):", open_view.events()[0]["payload"] != {})

    dirty_open = view(
        (
            TraceEvent(
                event_id="4",
                kind="tool_call",
                actor_id="s",
                parent_event_id=None,
                payload={"result": f"leak {SECRET}"},
            ),
        ),
        terms=(),
    )
    leaked = SECRET in repr(dirty_open.events())
    print("6 empty-terms LEAKS a contaminated payload:", leaked)


if __name__ == "__main__":
    main()
