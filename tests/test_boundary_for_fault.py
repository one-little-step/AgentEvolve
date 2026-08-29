"""W2: ``boundary_for_fault`` -- map a blame graph to a resume boundary.

Gap 3 (``docs/plans/editor-tools-live-wiring.md``): resume selection was manual
("boundary 3 of 4" chosen by reading the trace). The loop has the blame graph
instead. This pure function walks ``NodeStart``s, matches the blamed actor's
failing cycle, and returns the number of LLM boundaries to tape before going
live -- the same ``--resume N`` semantics RQ5 settled (boundaries ``< N`` taped,
``>= N`` live).

Contract:
* ``int`` = resume boundary (boundaries with ``sequence < N`` are taped).
* ``None`` = fall through to full validation: no blame, no matching node, fault
  before the first boundary (nothing to tape), or fault at/after the last
  boundary (tail ~= whole run, replay pointless).

Edge cases pinned here:
* multiple blamed nodes -> max-blame actor wins (ties broken by actor_id);
* a node that re-runs across cycles -> the LAST occurrence is the failing cycle
  (faults localize near the end), chosen by greatest ``sequence``;
* ``parent_event_id`` is carried on ``NodeStart`` so a caller CAN disambiguate
  the same actor at different nesting levels; the mapper itself uses last-
  occurrence, which is correct for the real trace shape where every node nests
  under a ``CugaLiteSubgraph`` root (resuming at the root would always yield 0).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis  # noqa: E402
from agent_evolve.core.tape import TapeIndex, boundary_for_fault  # noqa: E402


def _analysis(*nodes: BlameNode) -> CausalAnalysis:
    return CausalAnalysis(
        mechanism="m", severity=0.7, score=0.0,
        blame_graph=BlameGraph(nodes=tuple(nodes)),
    )


def _node(seq: int, node: str, parent: str | None = None) -> dict:
    return {
        "event_id": f"graph:{seq}",
        "kind": "graph_node_start",
        "parent_event_id": parent,
        "sequence": seq,
        "payload": {"node": node, "step": seq, "state_before_ref": f"ref-{seq}"},
    }


def _llm(seq: int, run_id: str) -> list[dict]:
    return [
        {
            "event_id": f"graph:{seq}",
            "kind": "llm_call_start",
            "parent_event_id": None,
            "sequence": seq,
            "payload": {"run_id": run_id, "messages_ref": f"m-{run_id}"},
        },
        {
            "event_id": f"graph:{seq + 1}",
            "kind": "llm_call_end",
            "parent_event_id": None,
            "sequence": seq + 1,
            "payload": {"run_id": run_id, "response_ref": f"r-{run_id}"},
        },
    ]


def _index(events: list[dict], tmp_path: Path) -> TapeIndex:
    (tmp_path / "payloads").mkdir(exist_ok=True)
    return TapeIndex(events=events, payloads_dir=tmp_path / "payloads")


def test_returns_boundary_count_before_blamed_node(tmp_path) -> None:
    events = [
        _node(0, "prepare"),
        *_llm(1, "a"),          # boundary 0
        _node(3, "sandbox"),
        *_llm(4, "b"),          # boundary 1
        _node(6, "finalize"),
        *_llm(7, "c"),          # boundary 2
    ]
    idx = _index(events, tmp_path)

    n = boundary_for_fault(idx, _analysis(BlameNode(actor_id="sandbox", blame=0.9)))
    assert n == 1  # sandbox at seq 3; boundaries < 3 = {seq 1}


def test_max_blame_actor_wins_with_deterministic_tiebreak(tmp_path) -> None:
    events = [
        _node(0, "prepare"),
        *_llm(1, "a"),          # boundary 0
        _node(3, "sandbox"),
        *_llm(4, "b"),          # boundary 1
        _node(6, "retriever"),
        *_llm(7, "c"),          # boundary 2
        _node(9, "finalize"),
    ]
    idx = _index(events, tmp_path)

    n = boundary_for_fault(
        idx,
        _analysis(
            BlameNode(actor_id="sandbox", blame=0.2),
            BlameNode(actor_id="retriever", blame=0.9),
        ),
    )
    assert n == 2  # retriever at seq 6; boundaries < 6 = {seq1, seq4}


def test_last_occurrence_of_a_rerunning_node_is_the_failing_cycle(tmp_path) -> None:
    events = [
        _node(0, "sandbox"),
        *_llm(1, "a"),          # boundary 0
        _node(3, "prepare"),
        *_llm(4, "b"),          # boundary 1
        _node(6, "sandbox"),    # the SAME actor runs again, later -> the fault
        *_llm(7, "c"),          # boundary 2
    ]
    idx = _index(events, tmp_path)

    n = boundary_for_fault(idx, _analysis(BlameNode(actor_id="sandbox", blame=0.9)))
    assert n == 2  # last sandbox occurrence at seq 6


def test_fault_before_first_boundary_falls_through(tmp_path) -> None:
    events = [
        _node(0, "prepare"),
        _node(1, "sandbox"),          # blamed, but before any LLM boundary
        *_llm(2, "a"),                # first boundary
        *_llm(4, "b"),
    ]
    idx = _index(events, tmp_path)

    n = boundary_for_fault(idx, _analysis(BlameNode(actor_id="sandbox", blame=0.9)))
    assert n is None


def test_fault_at_last_boundary_falls_through(tmp_path) -> None:
    events = [
        *_llm(0, "a"),
        _node(2, "prepare"),
        *_llm(3, "b"),
        _node(5, "finalize"),         # blamed, after the last boundary
    ]
    idx = _index(events, tmp_path)

    n = boundary_for_fault(idx, _analysis(BlameNode(actor_id="finalize", blame=0.9)))
    assert n is None


def test_no_blame_nodes_falls_through(tmp_path) -> None:
    idx = _index([*_llm(0, "a"), _node(2, "prepare")], tmp_path)
    n = boundary_for_fault(idx, _analysis())
    assert n is None


def test_unmatched_actor_falls_through(tmp_path) -> None:
    idx = _index([*_llm(0, "a"), _node(2, "prepare")], tmp_path)
    n = boundary_for_fault(idx, _analysis(BlameNode(actor_id="ghost", blame=0.9)))
    assert n is None


def test_parent_event_id_is_carried_on_node_starts(tmp_path) -> None:
    events = [
        _node(0, "root"),
        _node(1, "child", parent="graph:0"),
        *_llm(2, "a"),
    ]
    idx = _index(events, tmp_path)

    starts = idx.node_starts
    assert starts[0].parent_event_id is None
    assert starts[1].parent_event_id == "graph:0"
