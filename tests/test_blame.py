"""Tests for the causal blame graph data model."""
from __future__ import annotations

import pytest

from agent_evolve.core.blame import (
    BlameEdge,
    BlameGraph,
    BlameNode,
    CausalAnalysis,
    empty_analysis,
    merge_analyses,
)


def _node(actor_id: str, blame: float, artifacts=()) -> BlameNode:
    return BlameNode(actor_id=actor_id, blame=blame, artifacts=tuple(artifacts))


def test_blame_node_validates_blame_range():
    with pytest.raises(ValueError):
        _node("a", -0.1)
    with pytest.raises(ValueError):
        _node("a", 1.5)


def test_blame_node_requires_actor_id():
    with pytest.raises(ValueError):
        BlameNode(actor_id="", blame=0.5)


def test_blame_graph_rejects_duplicate_actor_ids():
    with pytest.raises(ValueError):
        BlameGraph(nodes=(_node("a", 0.5), _node("a", 0.3)))


def test_blame_graph_rejects_edge_to_unknown_actor():
    with pytest.raises(ValueError):
        BlameGraph(
            nodes=(_node("a", 0.5),),
            edges=(BlameEdge(from_actor="a", to_actor="ghost", mechanism="x"),),
        )


def test_blame_graph_total_blame_is_sum():
    g = BlameGraph(nodes=(_node("a", 0.4), _node("b", 0.6)))
    assert g.total_blame() == pytest.approx(1.0)


def test_top_blame_artifacts_orders_by_blame_then_name():
    g = BlameGraph(
        nodes=(
            _node("retriever", 0.8, artifacts=["skills/r1", "skills/r2"]),
            _node("executor", 0.2, artifacts=["policies/e1"]),
        )
    )
    assert g.top_blame_artifacts(2) == ("skills/r1", "skills/r2")
    assert g.top_blame_artifacts(3) == ("skills/r1", "skills/r2", "policies/e1")


def test_top_blame_artifacts_breaks_ties_deterministically():
    g = BlameGraph(
        nodes=(
            _node("a", 0.5, artifacts=["z", "a"]),
            _node("b", 0.5, artifacts=["m"]),
        )
    )
    # All same blame, so artifact_id ascending.
    assert g.top_blame_artifacts(3) == ("a", "m", "z")


def test_causal_analysis_validates_severity_and_score():
    g = BlameGraph(nodes=(_node("a", 0.5),))
    with pytest.raises(ValueError):
        CausalAnalysis(mechanism="m", severity=1.2, score=0.5, blame_graph=g)
    with pytest.raises(ValueError):
        CausalAnalysis(mechanism="m", severity=0.5, score=-0.1, blame_graph=g)


def test_causal_analysis_requires_mechanism():
    g = BlameGraph(nodes=(_node("a", 0.5),))
    with pytest.raises(ValueError):
        CausalAnalysis(mechanism="", severity=0.5, score=0.5, blame_graph=g)


def test_causal_analysis_artifact_ids_dedupes():
    g = BlameGraph(
        nodes=(
            _node("a", 0.5, artifacts=["x", "y"]),
            _node("b", 0.3, artifacts=["y", "z"]),
        )
    )
    a = CausalAnalysis(mechanism="m", severity=0.5, score=0.5, blame_graph=g)
    assert a.artifact_ids == ("x", "y", "z")
    assert a.actor_ids == ("a", "b")


def test_empty_analysis_has_no_blame_and_perfect_score():
    a = empty_analysis()
    assert a.severity == 0.0
    assert a.score == 1.0
    assert a.blame_graph.nodes == ()
    assert a.artifact_ids == ()


def test_merge_analyses_single_returns_same():
    a = CausalAnalysis(
        mechanism="m",
        severity=0.5,
        score=0.5,
        blame_graph=BlameGraph(nodes=(_node("a", 0.3),)),
    )
    assert merge_analyses([a]) is a


def test_merge_analyses_averages_severity_and_score():
    a1 = CausalAnalysis(
        mechanism="m1",
        severity=0.4,
        score=0.6,
        blame_graph=BlameGraph(nodes=(_node("a", 0.3),)),
    )
    a2 = CausalAnalysis(
        mechanism="m2",
        severity=0.6,
        score=0.8,
        blame_graph=BlameGraph(nodes=(_node("a", 0.5),)),
    )
    merged = merge_analyses([a1, a2])
    assert merged.severity == pytest.approx(0.5)
    assert merged.score == pytest.approx(0.7)
    assert merged.blame_graph.nodes[0].blame == pytest.approx(0.8)


def test_merge_analyses_clips_blame_at_one():
    a1 = CausalAnalysis(
        mechanism="m1",
        severity=0.5,
        score=0.5,
        blame_graph=BlameGraph(nodes=(_node("a", 0.7),)),
    )
    a2 = CausalAnalysis(
        mechanism="m2",
        severity=0.5,
        score=0.5,
        blame_graph=BlameGraph(nodes=(_node("a", 0.6),)),
    )
    merged = merge_analyses([a1, a2])
    assert merged.blame_graph.nodes[0].blame == 1.0


def test_merge_analyses_unions_edges_and_counterfactuals():
    g1 = BlameGraph(
        nodes=(_node("a", 0.3), _node("b", 0.2)),
        edges=(BlameEdge("a", "b", "m1"),),
    )
    g2 = BlameGraph(
        nodes=(_node("a", 0.3), _node("b", 0.2)),
        edges=(BlameEdge("a", "b", "m1"), BlameEdge("a", "b", "m2")),
    )
    a1 = CausalAnalysis(
        mechanism="m1", severity=0.5, score=0.5, blame_graph=g1,
        counterfactual_evidence=("c1", "c2"),
    )
    a2 = CausalAnalysis(
        mechanism="m2", severity=0.5, score=0.5, blame_graph=g2,
        counterfactual_evidence=("c2", "c3"),
    )
    merged = merge_analyses([a1, a2])
    assert {e.mechanism for e in merged.blame_graph.edges} == {"m1", "m2"}
    assert merged.counterfactual_evidence == ("c1", "c2", "c3")


def test_merge_analyses_rejects_empty():
    with pytest.raises(ValueError):
        merge_analyses([])
