"""Tests for entropy tracker and hierarchical DPP issue selection."""
from __future__ import annotations

import pytest

from agent_evolve.core.entropy import (
    CellKey,
    EntropyTracker,
    HierarchicalDPPSelector,
    Issue,
    _dpp_select,
)
import random


# ---------------------------------------------------------------------- #
# CellKey
# ---------------------------------------------------------------------- #
def test_cell_key_rejects_empty_parts():
    with pytest.raises(ValueError):
        CellKey(task_id="", mechanism_cluster_id="c0")
    with pytest.raises(ValueError):
        CellKey(task_id="t0", mechanism_cluster_id="")


# ---------------------------------------------------------------------- #
# EntropyTracker — basic record/lookup
# ---------------------------------------------------------------------- #
def test_entropy_tracker_records_score():
    e = EntropyTracker()
    e.record_score("t1", "c0", "h1", 0.5)
    assert e.cell_entropy("t1", "c0") == 0.0  # below evidence floor


def test_entropy_tracker_rejects_invalid_score():
    e = EntropyTracker()
    with pytest.raises(ValueError):
        e.record_score("t1", "c0", "h1", -0.1)
    with pytest.raises(ValueError):
        e.record_score("t1", "c0", "h1", 1.1)


def test_entropy_tracker_rejects_invalid_floor():
    with pytest.raises(ValueError):
        EntropyTracker(epsilon_floor=-0.1)
    with pytest.raises(ValueError):
        EntropyTracker(epsilon_floor=1.5)


def test_entropy_tracker_rejects_invalid_min_candidates():
    with pytest.raises(ValueError):
        EntropyTracker(min_comparable_candidates=0)


def test_entropy_tracker_rejects_invalid_min_rollouts():
    with pytest.raises(ValueError):
        EntropyTracker(min_rollouts_per_candidate=0)


# ---------------------------------------------------------------------- #
# EntropyTracker — gating
# ---------------------------------------------------------------------- #
def test_entropy_zero_below_min_candidates():
    """Need >= 3 comparable candidates per cell."""
    e = EntropyTracker(min_comparable_candidates=3, min_rollouts_per_candidate=2)
    for cand in ("h1", "h2"):
        e.mark_comparable("t1", "c0", cand)
        e.record_score("t1", "c0", cand, 0.5)
        e.record_score("t1", "c0", cand, 0.6)
    assert e.cell_entropy("t1", "c0") == 0.0


def test_entropy_zero_below_min_rollouts():
    """Need >= 2 rollouts per comparable candidate."""
    e = EntropyTracker(min_comparable_candidates=3, min_rollouts_per_candidate=2)
    for cand in ("h1", "h2", "h3"):
        e.mark_comparable("t1", "c0", cand)
        e.record_score("t1", "c0", cand, 0.5)  # only 1 rollout
    assert e.cell_entropy("t1", "c0") == 0.0


def test_entropy_nonzero_when_floor_met():
    e = EntropyTracker(min_comparable_candidates=3, min_rollouts_per_candidate=2)
    for cand, score in [("h1", 0.2), ("h2", 0.5), ("h3", 0.8)]:
        e.mark_comparable("t1", "c0", cand)
        e.record_score("t1", "c0", cand, score)
        e.record_score("t1", "c0", cand, score)
    e_val = e.cell_entropy("t1", "c0")
    assert e_val > 0.0


def test_entropy_uses_score_floor_when_max_is_low():
    """When max score is below epsilon_floor, floor kicks in."""
    e = EntropyTracker(
        min_comparable_candidates=3,
        min_rollouts_per_candidate=2,
        epsilon_floor=0.5,
    )
    # All scores well below 0.5 floor.
    for cand in ("h1", "h2", "h3"):
        e.mark_comparable("t1", "c0", cand)
        e.record_score("t1", "c0", cand, 0.01)
        e.record_score("t1", "c0", cand, 0.02)
    e_val = e.cell_entropy("t1", "c0")
    var = ((0.01 - 0.015) ** 2 + (0.02 - 0.015) ** 2) * 3 / 6
    # Variance over 6 samples; mean = 0.015
    # var = sum((x - mean)^2) / n where n=6
    scores = [0.01, 0.02, 0.01, 0.02, 0.01, 0.02]
    mean = sum(scores) / len(scores)
    var = sum((x - mean) ** 2 for x in scores) / len(scores)
    expected = var * 0.5  # floor used because max=0.02 < 0.5
    assert e_val == pytest.approx(expected, rel=1e-9)


def test_mark_comparable_does_not_double_count():
    e = EntropyTracker()
    e.mark_comparable("t1", "c0", "h1")
    e.mark_comparable("t1", "c0", "h1")  # idempotent
    cell = e._cells[CellKey("t1", "c0")]
    assert cell.comparable == {"h1"}


# ---------------------------------------------------------------------- #
# EntropyTracker — freshness
# ---------------------------------------------------------------------- #
def test_freshness_reduces_entropy_weight_with_age():
    e = EntropyTracker(min_comparable_candidates=3, min_rollouts_per_candidate=2)
    for cand in ("h1", "h2", "h3"):
        e.mark_comparable("t1", "c0", cand)
        e.record_score("t1", "c0", cand, 0.5)
        e.record_score("t1", "c0", cand, 0.6)
    e.refresh_at_barrier(0)
    e0 = e.entropy_weighted_with_freshness("t1", "c0", 0)
    e1 = e.entropy_weighted_with_freshness("t1", "c0", 1)
    e2 = e.entropy_weighted_with_freshness("t1", "c0", 2)
    assert e0 > e1 > e2 > 0.0


def test_refresh_at_barrier_rejects_negative():
    e = EntropyTracker()
    with pytest.raises(ValueError):
        e.refresh_at_barrier(-1)


# ---------------------------------------------------------------------- #
# EntropyTracker — heap
# ---------------------------------------------------------------------- #
def test_top_entropy_cells_orders_descending():
    e = EntropyTracker(min_comparable_candidates=3, min_rollouts_per_candidate=2)
    # Cell A: high variance.
    for cand, s in [("h1", 0.1), ("h2", 0.5), ("h3", 0.9)]:
        e.mark_comparable("tA", "c0", cand)
        e.record_score("tA", "c0", cand, s)
        e.record_score("tA", "c0", cand, s)
    # Cell B: lower variance (nonzero so it still enters the heap).
    for cand, s in [("h1", 0.4), ("h2", 0.45), ("h3", 0.5)]:
        e.mark_comparable("tB", "c0", cand)
        e.record_score("tB", "c0", cand, s)
        e.record_score("tB", "c0", cand, s)
    top = e.top_entropy_cells(2)
    assert len(top) == 2
    assert top[0][0].task_id == "tA"
    assert top[1][0].task_id == "tB"
    assert top[0][1] > top[1][1]


def test_top_entropy_cells_returns_at_most_k():
    e = EntropyTracker()
    assert e.top_entropy_cells(5) == ()


def test_top_entropy_cells_rejects_negative_k():
    e = EntropyTracker()
    with pytest.raises(ValueError):
        e.top_entropy_cells(-1)


# ---------------------------------------------------------------------- #
# HierarchicalDPPSelector
# ---------------------------------------------------------------------- #
def _issue(task: str, mech: str, severity: float, entropy: float = 0.5) -> Issue:
    return Issue(
        task_id=task,
        mechanism_cluster_id=mech,
        severity=severity,
        entropy=entropy,
        freshness_weight=1.0,
    )


def test_selector_rejects_unknown_mode():
    with pytest.raises(ValueError):
        HierarchicalDPPSelector(mode="bogus")


def test_selector_rejects_negative_lambda():
    with pytest.raises(ValueError):
        HierarchicalDPPSelector(lambda_div=-1.0)


def test_selector_returns_empty_for_empty_issues():
    s = HierarchicalDPPSelector()
    assert s.select((), 5, 5) == ()


def test_selector_rejects_negative_k():
    s = HierarchicalDPPSelector()
    with pytest.raises(ValueError):
        s.select((_issue("t1", "c0", 0.5),), -1, 1)


def test_selector_severity_rank_picks_highest_severity_first():
    s = HierarchicalDPPSelector(mode="severity_rank")
    issues = (
        _issue("t1", "c0", 0.3),
        _issue("t2", "c0", 0.9),
        _issue("t3", "c0", 0.5),
    )
    out = s.select(issues, 1, 1)
    assert len(out) == 1
    assert out[0].task_id == "t2"


def test_selector_random_is_seeded_reproducible():
    issues = tuple(_issue(f"t{i}", "c0", 0.5) for i in range(5))
    s1 = HierarchicalDPPSelector(mode="random", seed=42)
    s2 = HierarchicalDPPSelector(mode="random", seed=42)
    out1 = s1.select(issues, 2, 1)
    out2 = s2.select(issues, 2, 1)
    assert out1 == out2
    assert len(out1) == 2


def test_selector_dpp_uses_task_similarity_for_diversity():
    """Two highly similar tasks should not both be picked when k=1."""
    sim_calls: list[tuple[str, str]] = []

    def task_sim(a: str, b: str) -> float:
        sim_calls.append((a, b))
        return 1.0 if a == b else 0.0

    s = HierarchicalDPPSelector(mode="dpp", task_similarity=task_sim, lambda_div=0.5)
    issues = (
        _issue("t1", "c0", 0.6, entropy=0.5),
        _issue("t2", "c0", 0.6, entropy=0.5),
    )
    out = s.select(issues, 2, 1)
    assert len(out) == 2  # both picked since k=2
    # When k=1, the second pick should have benefited from diversity bonus.
    out1 = s.select(issues, 1, 1)
    assert len(out1) == 1


def test_selector_dpp_with_zero_similarity_behaves_like_greedy():
    s = HierarchicalDPPSelector(mode="dpp", task_similarity=lambda a, b: 0.0)
    issues = (
        _issue("t1", "c0", 0.9),
        _issue("t2", "c0", 0.3),
    )
    out = s.select(issues, 1, 1)
    assert out[0].task_id == "t1"


def test_dpp_select_returns_all_when_k_exceeds_items():
    items = [("a", 1.0, 1.0), ("b", 0.5, 1.0)]
    out = _dpp_select(items, 5, lambda a, b: 0.0, random.Random(0))
    assert set(out) == {"a", "b"}


def test_dpp_select_returns_empty_for_zero_k():
    out = _dpp_select([("a", 1.0, 1.0)], 0, lambda a, b: 0.0, random.Random(0))
    assert out == ()


def test_selector_mechanism_selection_within_task():
    """k_mechanisms_per_task limits mechanisms chosen per task."""
    s = HierarchicalDPPSelector(mode="severity_rank")
    issues = (
        _issue("t1", "c0", 0.9),
        _issue("t1", "c1", 0.5),
        _issue("t1", "c2", 0.3),
    )
    out = s.select(issues, 1, 2)
    assert len(out) == 2
    assert {i.mechanism_cluster_id for i in out} == {"c0", "c1"}
