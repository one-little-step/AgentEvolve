"""Tests for task-local incremental mechanism clustering."""
from __future__ import annotations

import pytest

from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis, CausalFinding
from agent_evolve.core.clustering import (
    ClusterRegistry,
    LexicalEmbedder,
    MechanismClusterer,
)


def _analysis(mechanism: str, actor: str = "a", artifacts=()) -> CausalAnalysis:
    return CausalAnalysis(
        mechanism=mechanism,
        severity=0.5,
        score=0.5,
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id=actor, blame=0.5, artifacts=tuple(artifacts)),)
        ),
    )


def finding(text: str) -> CausalFinding:
    return CausalFinding(
        verdict_id=f"v-{text}",
        candidate_id="cand-1",
        task_id="task-a",
        trace_id="trace-1",
        status="observed",
        mechanism_description=text,
        mechanism_cluster_id="c",
        severity=0.5,
        confidence=0.5,
        evidence_refs=(f"{text}-evidence",),
        rationale="test",
        blame_graph=BlameGraph(nodes=()),
    )


class UnavailableEmbedder:
    dim = 64

    def embed(self, text: str) -> tuple[float, ...]:
        raise RuntimeError("embedding provider unavailable")


def test_lexical_embedder_returns_normalized_vector():
    e = LexicalEmbedder(dim=8)
    v = e.embed("hello world hello")
    assert len(v) == 8
    assert sum(x * x for x in v) == pytest.approx(1.0)


def test_lexical_embedder_deterministic():
    e = LexicalEmbedder(dim=16)
    assert e.embed("retrieve stale schema") == e.embed("retrieve stale schema")


def test_lexical_embedder_rejects_zero_dim():
    with pytest.raises(ValueError):
        LexicalEmbedder(dim=0)


def test_clusterer_requires_task_id():
    with pytest.raises(ValueError):
        MechanismClusterer(task_id="", embedder=LexicalEmbedder())


def test_clusterer_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        MechanismClusterer(
            task_id="t1", embedder=LexicalEmbedder(), join_threshold=1.5
        )


def test_anchor_creates_new_cluster():
    c = MechanismClusterer(task_id="t1", embedder=LexicalEmbedder())
    a = c.add_anchor("retriever returned stale schema")
    assert a.is_new_cluster
    assert c.cluster_count == 1


def test_similar_mechanism_joins_existing_cluster():
    c = MechanismClusterer(task_id="t1", embedder=LexicalEmbedder(), join_threshold=0.5)
    c.add_anchor("retriever returned stale schema")
    a = c.assign(_analysis("retriever returned stale schema"))
    assert not a.is_new_cluster
    assert c.cluster_size(a.cluster_id) == 2


def test_dissimilar_mechanism_creates_new_cluster():
    c = MechanismClusterer(task_id="t1", embedder=LexicalEmbedder(), join_threshold=0.95)
    c.add_anchor("retriever returned stale schema")
    a = c.assign(_analysis("executor crashed on null argument"))
    assert a.is_new_cluster
    assert c.cluster_count == 2


def test_begin_iteration_must_increase():
    c = MechanismClusterer(task_id="t1", embedder=LexicalEmbedder())
    c.begin_iteration(1)
    c.begin_iteration(2)
    with pytest.raises(ValueError):
        c.begin_iteration(2)
    with pytest.raises(ValueError):
        c.begin_iteration(0)


def test_cluster_freshness_increases_when_not_touched():
    c = MechanismClusterer(
        task_id="t1", embedder=LexicalEmbedder(), join_threshold=0.0
    )
    # join_threshold=0 forces a join whenever a cluster exists.
    c.begin_iteration(1)
    a = c.add_anchor("alpha beta")
    assert c.cluster_freshness(a.cluster_id) == 0
    c.begin_iteration(2)
    assert c.cluster_freshness(a.cluster_id) == 1
    # Touch it again with a similar-enough string.
    c.assign(_analysis("alpha beta gamma"))
    assert c.cluster_freshness(a.cluster_id) == 0


def test_cluster_registry_returns_one_clusterer_per_task():
    reg = ClusterRegistry(
        embedder_factory=lambda: LexicalEmbedder(),
        join_threshold=0.5,
    )
    c1 = reg.clusterer_for("t1")
    c2 = reg.clusterer_for("t1")
    c3 = reg.clusterer_for("t2")
    assert c1 is c2
    assert c1 is not c3


def test_cluster_registry_all_cluster_ids_namespaced_by_task():
    reg = ClusterRegistry(
        embedder_factory=lambda: LexicalEmbedder(),
        join_threshold=0.5,
    )
    c1 = reg.clusterer_for("t1")
    c2 = reg.clusterer_for("t2")
    c1.add_anchor("alpha beta")
    c2.add_anchor("gamma delta")
    ids = reg.all_cluster_ids()
    assert "t1:c0" in ids
    assert "t2:c0" in ids


def test_cluster_registry_propagates_iteration_barriers():
    reg = ClusterRegistry(embedder_factory=lambda: LexicalEmbedder())
    reg.clusterer_for("t1").begin_iteration(1)
    reg.begin_iteration(2)
    assert reg.clusterer_for("t1").current_iteration == 2


def test_centroid_drifts_with_assignments():
    """Confirm the running-mean update changes the centroid over time."""
    c = MechanismClusterer(
        task_id="t1", embedder=LexicalEmbedder(dim=16), join_threshold=0.0
    )
    # join_threshold=0 forces everything into one cluster.
    c.add_anchor("alpha beta")
    first_centroid = list(c._clusters["c0"].centroid)
    c.assign(_analysis("alpha beta gamma"))
    second_centroid = list(c._clusters["c0"].centroid)
    assert first_centroid != second_centroid
    assert c.cluster_size("c0") == 2


def test_assign_with_minimal_mechanism_text_still_returns():
    """Edge case: a mechanism whose tokens all collide to one bucket."""
    c = MechanismClusterer(task_id="t1", embedder=LexicalEmbedder(dim=4))
    a = c.assign(_analysis("alpha"))
    assert a.is_new_cluster


def test_same_mechanism_text_in_two_tasks_never_shares_cluster() -> None:
    registry = ClusterRegistry(embedder_factory=LexicalEmbedder)
    assert registry.assign("task-a", finding("stale schema")).cluster_id != registry.assign("task-b", finding("stale schema")).cluster_id


def test_lexical_fallback_is_recorded_when_provider_unavailable() -> None:
    assignment = MechanismClusterer(embedder=UnavailableEmbedder()).assign(finding("stale schema"))
    assert assignment.embedding_fallback_reason == "provider_unavailable"


def test_assign_finding_sets_task_id_and_freshness():
    c = MechanismClusterer(task_id="t1", embedder=LexicalEmbedder())
    c.begin_iteration(3)
    a = c.assign_finding(finding("retriever returned stale schema"))
    assert a.task_id == "t1"
    assert a.freshness_iteration == 3
    assert a.is_new_cluster


def test_assign_finding_joins_similar_cluster():
    c = MechanismClusterer(
        task_id="t1", embedder=LexicalEmbedder(), join_threshold=0.5
    )
    c.add_anchor("retriever returned stale schema")
    a = c.assign_finding(finding("retriever returned stale schema"))
    assert not a.is_new_cluster
    assert c.cluster_size(a.cluster_id) == 2


def test_cap_forces_join_to_nearest_cluster():
    c = MechanismClusterer(
        task_id="t1",
        embedder=LexicalEmbedder(),
        join_threshold=0.99,
        max_clusters_per_task=2,
    )
    c.add_anchor("retriever returned stale schema")
    c.add_anchor("executor crashed on null argument")
    assert c.cluster_count == 2
    a = c.assign(_analysis("completely different mechanism text"))
    assert not a.is_new_cluster
    assert c.cluster_count == 2


def test_max_clusters_per_task_must_be_positive():
    with pytest.raises(ValueError):
        MechanismClusterer(
            task_id="t1", embedder=LexicalEmbedder(), max_clusters_per_task=0
        )


def test_fallback_embedder_still_produces_assignment():
    c = MechanismClusterer(embedder=UnavailableEmbedder())
    a = c.assign(_analysis("retriever returned stale schema"))
    assert a.embedding_fallback_reason == "provider_unavailable"
    assert a.cluster_id == "c0"
