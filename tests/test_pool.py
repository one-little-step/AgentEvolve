"""Tests for the persistent pool, score tensor, and Pareto selection."""
from __future__ import annotations

import pytest

from agent_evolve.core.contracts import EvolutionCandidate
from agent_evolve.core.pool import (
    PersistentPool,
    PoolEntry,
    ScoreCell,
    ScoreProvenance,
)


def _candidate(cid: str, version: str | None = None, parents: tuple[str, ...] = ()) -> EvolutionCandidate:
    return EvolutionCandidate(
        candidate_id=cid,
        version=version or f"{cid}-v0",
        artifact_hashes={},
        parent_ids=parents,
    )


def _prov(task: str, mech: str, rollout: int = 0, score: float = 0.5) -> tuple[float, ScoreProvenance]:
    return score, ScoreProvenance(
        task_id=task,
        mechanism_cluster_id=mech,
        trace_id=f"trace-{task}-{mech}-{rollout}",
        rollout_seq=rollout,
        analyzer_model_id="fake-analyzer",
        judge_model_id="fake-judge",
        blame_confidence=0.8,
        blame_stability=0.7,
        artifact_versions={},
    )


# ---------------------------------------------------------------------- #
# ScoreProvenance
# ---------------------------------------------------------------------- #
def test_provenance_rejects_empty_task():
    with pytest.raises(ValueError):
        ScoreProvenance(
            task_id="",
            mechanism_cluster_id="c0",
            trace_id="t",
            rollout_seq=0,
            analyzer_model_id="",
            judge_model_id="",
            blame_confidence=0.5,
            blame_stability=0.5,
            artifact_versions={},
        )


def test_provenance_rejects_invalid_blame_confidence():
    with pytest.raises(ValueError):
        ScoreProvenance(
            task_id="t",
            mechanism_cluster_id="c0",
            trace_id="t",
            rollout_seq=0,
            analyzer_model_id="",
            judge_model_id="",
            blame_confidence=1.2,
            blame_stability=0.5,
            artifact_versions={},
        )


def test_provenance_rejects_negative_rollout_seq():
    with pytest.raises(ValueError):
        ScoreProvenance(
            task_id="t",
            mechanism_cluster_id="c0",
            trace_id="t",
            rollout_seq=-1,
            analyzer_model_id="",
            judge_model_id="",
            blame_confidence=0.5,
            blame_stability=0.5,
            artifact_versions={},
        )


def test_provenance_freezes_artifact_versions():
    src = {"a": "v1"}
    p = ScoreProvenance(
        task_id="t", mechanism_cluster_id="c0", trace_id="t", rollout_seq=0,
        analyzer_model_id="", judge_model_id="",
        blame_confidence=0.5, blame_stability=0.5, artifact_versions=src,
    )
    src["b"] = "v2"  # mutation of source dict
    assert "b" not in p.artifact_versions


# ---------------------------------------------------------------------- #
# ScoreCell
# ---------------------------------------------------------------------- #
def test_score_cell_rejects_invalid_score():
    c = ScoreCell()
    with pytest.raises(ValueError):
        c.add(-0.1, _prov("t", "c0")[1])


def test_score_cell_rejects_wrong_rollout_seq():
    c = ScoreCell()
    s, p = _prov("t", "c0", rollout=0)
    c.add(s, p)
    s2, p2 = _prov("t", "c0", rollout=5)  # wrong seq
    with pytest.raises(ValueError):
        c.add(s2, p2)


def test_score_cell_mean_and_max():
    c = ScoreCell()
    for r in range(3):
        s, p = _prov("t", "c0", rollout=r, score=0.1 * r)
        c.add(s, p)
    assert c.rollout_count == 3
    assert c.mean == pytest.approx((0.0 + 0.1 + 0.2) / 3)
    assert c.max == pytest.approx(0.2)


def test_score_cell_empty_returns_zero():
    c = ScoreCell()
    assert c.rollout_count == 0
    assert c.mean == 0.0
    assert c.max == 0.0


# ---------------------------------------------------------------------- #
# Pool basics
# ---------------------------------------------------------------------- #
def test_pool_add_base_sets_base_id():
    p = PersistentPool()
    p.add_base(_candidate("base"))
    assert p.base_id == "base"
    assert p.base.is_base is True
    assert len(p) == 1


def test_pool_rejects_second_base():
    p = PersistentPool()
    p.add_base(_candidate("base"))
    with pytest.raises(ValueError):
        p.add_base(_candidate("base2"))


def test_pool_add_candidate_records_origin_attempts():
    p = PersistentPool()
    p.add_base(_candidate("base"))
    e = p.add_candidate(_candidate("c1"), origin_attempt_ids=("att-1", "att-2"))
    assert e.origin_attempt_ids == ("att-1", "att-2")


def test_pool_rejects_duplicate_candidate_id():
    p = PersistentPool()
    p.add_base(_candidate("base"))
    p.add_candidate(_candidate("c1"))
    with pytest.raises(ValueError):
        p.add_candidate(_candidate("c1"))


def test_pool_record_score_unknown_candidate_raises():
    p = PersistentPool()
    p.add_base(_candidate("base"))
    with pytest.raises(KeyError):
        p.record_score("unknown", 0.5, _prov("t", "c0")[1])


def test_pool_base_id_raises_when_no_base():
    p = PersistentPool()
    with pytest.raises(ValueError):
        _ = p.base_id


def test_pool_rejects_invalid_min_comparable():
    with pytest.raises(ValueError):
        PersistentPool(min_comparable_rollouts=0)


def test_pool_contains_and_get():
    p = PersistentPool()
    p.add_base(_candidate("base"))
    p.add_candidate(_candidate("c1"))
    assert "c1" in p
    assert "unknown" not in p
    assert p.get("c1").candidate_id == "c1"
    with pytest.raises(KeyError):
        p.get("nope")


def test_pool_candidate_ids_in_insertion_order():
    p = PersistentPool()
    p.add_base(_candidate("base"))
    p.add_candidate(_candidate("c1"))
    p.add_candidate(_candidate("c2"))
    assert p.candidate_ids() == ("base", "c1", "c2")


# ---------------------------------------------------------------------- #
# Pareto
# ---------------------------------------------------------------------- #
def test_dominates_false_when_no_compatible_overlap():
    """Two candidates with disjoint task coverage do not dominate each other."""
    p = PersistentPool(min_comparable_rollouts=1)
    p.add_base(_candidate("base"))
    p.add_candidate(_candidate("c1"))
    # base has task A; c1 has task B.
    for r in range(1):
        s, prov = _prov("A", "c0", rollout=r, score=0.3)
        p.record_score("base", s, prov)
        s2, prov2 = _prov("B", "c0", rollout=r, score=0.9)
        p.record_score("c1", s2, prov2)
    assert not p.dominates("base", "c1")
    assert not p.dominates("c1", "base")


def test_dominates_true_when_strictly_better_on_overlap():
    p = PersistentPool(min_comparable_rollouts=2)
    p.add_base(_candidate("base"))
    p.add_candidate(_candidate("c1"))
    for r in range(2):
        p.record_score("base", *_prov("A", "c0", rollout=r, score=0.3))
        p.record_score("c1",   *_prov("A", "c0", rollout=r, score=0.9))
    assert p.dominates("c1", "base")
    assert not p.dominates("base", "c1")


def test_dominates_false_when_one_strictly_worse():
    """Mixed scores -> neither dominates."""
    p = PersistentPool(min_comparable_rollouts=2)
    p.add_base(_candidate("base"))
    p.add_candidate(_candidate("c1"))
    # c1 better on A, worse on B.
    for r in range(2):
        p.record_score("base", *_prov("A", "c0", rollout=r, score=0.3))
        p.record_score("c1",   *_prov("A", "c0", rollout=r, score=0.9))
        p.record_score("base", *_prov("B", "c0", rollout=r, score=0.9))
        p.record_score("c1",   *_prov("B", "c0", rollout=r, score=0.3))
    assert not p.dominates("c1", "base")
    assert not p.dominates("base", "c1")


def test_dominates_respects_min_comparable_rollouts():
    """Below the floor, no dominance."""
    p = PersistentPool(min_comparable_rollouts=2)
    p.add_base(_candidate("base"))
    p.add_candidate(_candidate("c1"))
    # Only 1 rollout each on the same cell.
    p.record_score("base", *_prov("A", "c0", rollout=0, score=0.1))
    p.record_score("c1",   *_prov("A", "c0", rollout=0, score=0.9))
    assert not p.dominates("c1", "base")


def test_pareto_frontier_returns_non_dominated():
    p = PersistentPool(min_comparable_rollouts=2)
    p.add_base(_candidate("base"))
    p.add_candidate(_candidate("c1"))
    p.add_candidate(_candidate("c2"))
    # c1 dominates base on A; c2 is dominated by c1 on A but equal elsewhere.
    for r in range(2):
        p.record_score("base", *_prov("A", "c0", rollout=r, score=0.3))
        p.record_score("c1",   *_prov("A", "c0", rollout=r, score=0.9))
        p.record_score("c2",   *_prov("A", "c0", rollout=r, score=0.5))
    frontier = set(p.pareto_frontier())
    assert "c1" in frontier
    assert "base" not in frontier
    assert "c2" not in frontier


def test_pareto_frontier_keeps_all_when_incomparable():
    p = PersistentPool(min_comparable_rollouts=1)
    p.add_base(_candidate("base"))
    p.add_candidate(_candidate("c1"))
    # Disjoint coverage; both stay on frontier.
    p.record_score("base", *_prov("A", "c0", rollout=0, score=0.5))
    p.record_score("c1",   *_prov("B", "c0", rollout=0, score=0.5))
    frontier = set(p.pareto_frontier())
    assert frontier == {"base", "c1"}


# ---------------------------------------------------------------------- #
# Prune
# ---------------------------------------------------------------------- #
def test_prune_removes_candidate():
    p = PersistentPool()
    p.add_base(_candidate("base"))
    p.add_candidate(_candidate("c1"))
    pruned = p.prune("c1")
    assert pruned.candidate_id == "c1"
    assert "c1" not in p
    assert len(p) == 1


def test_prune_refuses_to_remove_base():
    p = PersistentPool()
    p.add_base(_candidate("base"))
    with pytest.raises(ValueError):
        p.prune("base")


def test_prune_unknown_raises():
    p = PersistentPool()
    p.add_base(_candidate("base"))
    with pytest.raises(KeyError):
        p.prune("nope")


# ---------------------------------------------------------------------- #
# mean_score_per_task
# ---------------------------------------------------------------------- #
def test_mean_score_per_task_aggregates_across_mechanisms():
    p = PersistentPool()
    e = p.add_base(_candidate("base"))
    # 2 cells, same task, different mechanisms.
    e.cell("A", "c0").add(0.4, _prov("A", "c0")[1])
    e.cell("A", "c1").add(0.6, _prov("A", "c1")[1])
    means = e.mean_score_per_task()
    assert means["A"] == pytest.approx(0.5)
