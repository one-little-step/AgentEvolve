"""Task aggregation must use complete task IDs.

Binding mandate, docs/architecture/data-contracts.md:

    All IDs are non-empty strings, stable for the lifetime of a run, and
    compared by exact full value. Prefixes, substrings, first characters,
    truncations, and hashes of IDs are forbidden as aggregation or grouping
    keys.

The prototype aggregated by ``task_id[0]``, silently merging every task that
shared a first character into one Pareto objective.
"""
from __future__ import annotations

import pytest

from agent_evolve.core.contracts import EvolutionCandidate
from agent_evolve.core.pool import PoolEntry, ScoreProvenance


def _prov(task_id: str, cluster: str, seq: int = 0) -> ScoreProvenance:
    return ScoreProvenance(
        task_id=task_id,
        mechanism_cluster_id=cluster,
        trace_id=f"trace-{task_id}-{cluster}-{seq}",
        rollout_seq=seq,
        analyzer_model_id="analyzer-v1",
        judge_model_id="judge-v1",
        blame_confidence=0.9,
        blame_stability=0.8,
        artifact_versions={"artifact-a": "sha256:abc"},
    )


def _entry() -> PoolEntry:
    return PoolEntry(
        candidate=EvolutionCandidate(
            candidate_id="cand-1",
            version="v1",
            artifact_hashes={"artifact-a": "sha256:abc"},
        ),
        is_base=False,
    )


def test_mean_score_per_task_does_not_merge_tasks_sharing_a_first_character() -> None:
    """``task-a`` and ``task-b`` share a prefix but are distinct objectives."""
    entry = _entry()
    entry.cell("task-a", "cluster-1").add(1.0, _prov("task-a", "cluster-1"))
    entry.cell("task-b", "cluster-1").add(0.0, _prov("task-b", "cluster-1"))

    means = entry.mean_score_per_task()

    assert set(means) == {"task-a", "task-b"}, (
        "aggregation collapsed distinct task IDs; a prefix or slice is being "
        f"used as the grouping key. got keys: {sorted(means)}"
    )
    assert means["task-a"] == pytest.approx(1.0)
    assert means["task-b"] == pytest.approx(0.0)


def test_mean_score_per_task_keys_are_whole_task_ids() -> None:
    """No key may be a truncation of the task ID it represents."""
    entry = _entry()
    entry.cell("gaia-101", "cluster-1").add(0.5, _prov("gaia-101", "cluster-1"))
    entry.cell("gaia-999", "cluster-1").add(0.5, _prov("gaia-999", "cluster-1"))

    means = entry.mean_score_per_task()

    assert set(means) == {"gaia-101", "gaia-999"}
    for key in means:
        assert len(key) > 1, f"key {key!r} is a truncated task ID"


def test_mean_score_per_task_averages_across_mechanisms_within_one_task() -> None:
    """Multiple mechanism clusters for one task collapse to that task's mean."""
    entry = _entry()
    entry.cell("task-a", "cluster-1").add(1.0, _prov("task-a", "cluster-1"))
    entry.cell("task-a", "cluster-2").add(0.0, _prov("task-a", "cluster-2"))

    means = entry.mean_score_per_task()

    assert set(means) == {"task-a"}
    assert means["task-a"] == pytest.approx(0.5)


def test_mean_score_per_task_skips_cells_without_rollouts() -> None:
    """A cell with zero rollouts carries no evidence and must be excluded."""
    entry = _entry()
    entry.cell("task-a", "cluster-1").add(1.0, _prov("task-a", "cluster-1"))
    entry.cell("task-b", "cluster-1")  # created but never scored

    means = entry.mean_score_per_task()

    assert set(means) == {"task-a"}, (
        "an unevaluated cell leaked into aggregation as if it were evidence"
    )


def test_mean_weighted_score_per_task_uses_whole_task_ids() -> None:
    """Weighted per-task aggregation must not merge tasks sharing a prefix."""
    entry = _entry()
    entry.cell("task-a", "cluster-1").add(1.0, _prov("task-a", "cluster-1"))
    entry.cell("task-b", "cluster-1").add(0.0, _prov("task-b", "cluster-1"))

    means = entry.mean_weighted_score_per_task()

    assert set(means) == {"task-a", "task-b"}, (
        "weighted aggregation collapsed distinct task IDs; got keys: "
        f"{sorted(means)}"
    )
    assert means["task-a"] == pytest.approx(1.0)
    assert means["task-b"] == pytest.approx(0.0)
