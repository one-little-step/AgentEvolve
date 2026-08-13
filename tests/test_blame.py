"""Tests for the causal blame graph data model."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_evolve.core.blame import (
    BlameEdge,
    BlameGraph,
    BlameNode,
    CausalAnalysis,
    CausalFinding,
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


def _observed_finding(**overrides) -> CausalFinding:
    kwargs = dict(
        verdict_id="v-1",
        candidate_id="c-1",
        task_id="t-1",
        trace_id="tr-1",
        status="observed",
        mechanism_description="bad retrieval",
        mechanism_cluster_id="cluster-1",
        severity=0.8,
        confidence=0.9,
        blame_graph=BlameGraph(nodes=()),
        evidence_refs=("tr-1",),
        rationale="trace-backed mechanism",
    )
    kwargs.update(overrides)
    return CausalFinding(**kwargs)


def test_observed_finding_requires_trace_backed_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence_refs"):
        CausalFinding(status="observed", mechanism_description="bad retrieval", evidence_refs=())


def test_insufficient_evidence_is_not_coerced_to_blame() -> None:
    finding = CausalFinding(status="insufficient_evidence", rationale="trace lacks causal link")
    assert finding.mechanism_cluster_id is None


def test_observed_finding_requires_mechanism_description() -> None:
    with pytest.raises(ValidationError, match="mechanism_description"):
        _observed_finding(mechanism_description="")


def test_observed_finding_requires_severity_and_confidence() -> None:
    with pytest.raises(ValidationError, match="severity"):
        _observed_finding(severity=None)
    with pytest.raises(ValidationError, match="confidence"):
        _observed_finding(confidence=None)


def test_finding_rejects_severity_out_of_range() -> None:
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        _observed_finding(severity=1.5)


def test_finding_rejects_blank_evidence_refs() -> None:
    with pytest.raises(ValidationError, match="evidence_refs"):
        _observed_finding(evidence_refs=("",))


def test_observed_finding_constructs_and_is_frozen() -> None:
    finding = _observed_finding()
    assert finding.status == "observed"
    assert finding.mechanism_cluster_id == "cluster-1"
    with pytest.raises(Exception):
        finding.severity = 0.1  # type: ignore[misc]
