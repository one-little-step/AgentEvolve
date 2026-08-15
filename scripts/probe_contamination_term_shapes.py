"""Probe: which expected_contract shapes produce NO contamination terms?

A contract value the extractor cannot see becomes a term it cannot scan for,
so any answer-shaped text in that value can pass the guard.
"""
from __future__ import annotations

from agent_evolve.adapters.cuga_editor_evidence import (
    EvidenceView,
    contamination_terms_from,
)
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace, TraceEvent

ANALYSIS = CausalAnalysis(
    mechanism="m",
    severity=0.5,
    score=0.0,
    blame_graph=BlameGraph(nodes=(BlameNode(actor_id="a", blame=1.0, artifacts=()),)),
)

SHAPES = {
    "plain string": {"expected_substring": "token-alpha"},
    "list of strings": {"expected_any": ["token-alpha", "token-beta"]},
    "nested dict": {"grader": {"expected": "token-alpha"}},
    "tuple": {"expected_any": ("token-alpha",)},
    "int-keyed answer": {"expected_value": 42},
    "short string": {"expected_substring": "ab"},
}


def leaks(contract: dict) -> tuple[tuple[str, ...], bool]:
    task = EvolutionTask(task_id="t", input_text="i", expected_contract=contract)
    terms = contamination_terms_from(task)
    event = TraceEvent(
        event_id="1",
        kind="tool_call",
        actor_id="s",
        parent_event_id=None,
        payload={"result": "the value is token-alpha"},
    )
    view = EvidenceView(
        analysis=ANALYSIS,
        trace=ExecutionTrace(
            trace_id="x",
            candidate_id="c",
            task_id="t",
            events=(event,),
            final_output="",
            status="completed",
        ),
        task=task,
        contamination_terms=terms,
    )
    return terms, "token-alpha" in repr(view.events())


def main() -> None:
    for label, contract in SHAPES.items():
        terms, leaked = leaks(contract)
        print(f"{label:20s} terms={terms!r:40s} LEAKS={leaked}")


if __name__ == "__main__":
    main()
