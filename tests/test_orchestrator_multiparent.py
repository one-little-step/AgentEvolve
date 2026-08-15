"""Multi-parent selection and observed lineage (spec §7, §9).

Parent-set size is bounded so prompt size does not grow with the pool: the
primary keeps the architecture's frequency-proportional semantics, donors come
from the Pareto frontier.
"""
from __future__ import annotations

import pytest

from agent_evolve.core.contracts import EvolutionCandidate
from agent_evolve.core.blame import empty_analysis
from agent_evolve.core.editor import FocusedValidationReport
from agent_evolve.core.pool import PersistentPool, ScoreProvenance

# Reuse the established fake harness from the phase 6 tests.
from test_phase_6_orchestrator import _runner  # type: ignore


def _candidate(candidate_id: str) -> EvolutionCandidate:
    return EvolutionCandidate(
        candidate_id=candidate_id,
        version=f"{candidate_id}-v",
        artifact_hashes={"skills/retrieval": "sha256:" + "0" * 64},
    )


def test_select_parents_returns_primary_first() -> None:
    runner = _runner()
    parents = runner.select_parents(k=3)
    assert parents
    assert parents[0].candidate_id == runner.pool.base.candidate_id


def test_select_parents_is_bounded_by_k() -> None:
    runner = _runner()
    for i in range(6):
        runner.pool.add_candidate(_candidate(f"extra-{i}"))
    assert len(runner.select_parents(k=3)) <= 3


def test_select_parents_never_repeats_the_primary_as_a_donor() -> None:
    runner = _runner()
    for i in range(4):
        runner.pool.add_candidate(_candidate(f"extra-{i}"))
    parents = runner.select_parents(k=4)
    ids = [p.candidate_id for p in parents]
    assert len(ids) == len(set(ids))


def test_select_parents_with_k_one_returns_only_the_primary() -> None:
    runner = _runner()
    runner.pool.add_candidate(_candidate("extra-0"))
    assert len(runner.select_parents(k=1)) == 1


def test_select_parents_rejects_k_below_one() -> None:
    runner = _runner()
    with pytest.raises(ValueError, match="k must be >= 1"):
        runner.select_parents(k=0)


def test_select_parents_draws_donors_from_the_pareto_frontier() -> None:
    runner = _runner()
    runner.pool.add_candidate(_candidate("frontier-cand"))
    frontier = set(runner.pool.pareto_frontier())
    parents = runner.select_parents(k=3)
    donors = [p.candidate_id for p in parents[1:]]
    assert all(d in frontier for d in donors)


def test_commit_to_pool_records_only_the_primary_by_default() -> None:
    runner = _runner()
    entry = runner._commit_single_parent_for_test()
    assert entry.candidate.parent_ids == (runner.pool.base.candidate_id,)


def test_commit_to_pool_records_observed_extra_parents() -> None:
    """Lineage must reflect donors actually read, not donors merely offered."""
    runner = _runner()
    runner.pool.add_candidate(_candidate("donor-1"))
    entry = runner._commit_with_extra_parents_for_test(("donor-1",))
    assert set(entry.candidate.parent_ids) == {
        runner.pool.base.candidate_id, "donor-1",
    }


def test_observed_extra_parents_appear_in_ancestors() -> None:
    runner = _runner()
    runner.pool.add_candidate(_candidate("donor-1"))
    entry = runner._commit_with_extra_parents_for_test(("donor-1",))
    assert "donor-1" in entry.candidate.ancestor_ids


def test_parent_ids_are_deduplicated_and_sorted() -> None:
    runner = _runner()
    runner.pool.add_candidate(_candidate("donor-1"))
    base_id = runner.pool.base.candidate_id
    entry = runner._commit_with_extra_parents_for_test((base_id, "donor-1", "donor-1"))
    assert entry.candidate.parent_ids == tuple(sorted({base_id, "donor-1"}))


def test_lineage_of_is_stable_for_multiple_parents() -> None:
    """Confirms the qf30 §15 verification with a regression test."""
    from pathlib import Path

    from agent_evolve.core.contracts import CandidateWorkspace
    from agent_evolve.core.editor import lineage_of

    ws = CandidateWorkspace("att-1", "v1", Path("."), "v0")
    assert lineage_of(ws, ("v-b", "v-a")) == "v-a|v-b"
    assert lineage_of(ws, ("v-a", "v-b")) == "v-a|v-b"
    assert lineage_of(ws) == "v0"


def test_ancestors_accumulate_across_generations_without_loss() -> None:
    """Sorting ancestor_ids must not drop inherited ancestry.

    commit_to_pool switched from append-order concatenation to a sorted set
    union. There is no ordering contract on ancestor_ids, but membership is
    load-bearing for merge's common-ancestor traversal, so prove nothing is
    lost across two generations.
    """
    runner = _runner()
    base_id = runner.pool.base.candidate_id

    first = runner._commit_with_extra_parents_for_test(("donor-1",))
    assert base_id in first.candidate.ancestor_ids
    assert "donor-1" in first.candidate.ancestor_ids

    # A second generation parented on the first must retain generation-1 ancestry.
    second = runner.commit_to_pool(
        first,
        runner.adapter.materialize_candidate(first.candidate.version, "att-gen2"),
        "att-gen2",
        FocusedValidationReport(origin=(), worked=(), regression=()),
        empty_analysis(),
        extra_parent_ids=("donor-2",),
    )
    ancestors = set(second.candidate.ancestor_ids)
    assert {base_id, "donor-1", "donor-2", first.candidate_id} <= ancestors
