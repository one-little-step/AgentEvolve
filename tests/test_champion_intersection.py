"""SV-2 — ``outcome`` must not reward skipping a task.

``_champion_outcome`` is a two-level mean with no shared-cell restriction. Cells
with ``rollout_count == 0`` are skipped, which is right on its own -- no evidence
is not a zero -- but the resulting means are then compared *across candidates
measured on different tasks*. Reproduced with the register's numbers::

    base   ran easy(0.9) + hard(0.1)  ->  outcome = 0.500
    candA  ran easy(0.9) only         ->  outcome = 0.900   <- wins

``candA`` is identical to base on the only task both attempted. It wins by not
attempting the hard one. Under RHO's design -- base gets ``k x G``, candidates get
``k x R`` -- unequal task sets are the norm, so this is structural.

**The fix direction is the register's, already resolved as option 3**: gate on
``S_j > 0`` first (SV-4, closed), then rank survivors on the **pairwise
intersection** of cells both entries have evidence for, reusing the
``comparable_cells`` / ``min_comparable_rollouts`` machinery ``dominates`` and
``pareto_frontier`` already use, and report the intersection size.

**Why the ranking key cannot stay a per-candidate scalar.** An
intersection-restricted outcome is defined only relative to a second entry, and
two candidates may share no cell at all::

    base  vs candA: shared={easy}  0.9 vs 0.9  -> tie
    base  vs candC: shared={hard}  0.1 vs 0.4  -> candC
    candA vs candC: shared={}                  -> no verdict expressible

So these tests pin *comparison* behaviour -- who beats whom, and who is exported
-- and deliberately do not assert a particular absolute ``outcome`` float. A test
demanding a specific scalar would be pinning the very thing the fix has to stop
relying on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from agent_evolve.core.contracts import EvolutionCandidate
from agent_evolve.core.pool import PersistentPool, ScoreProvenance

_MECH = "m0"


def _cand(cid: str, parents: tuple[str, ...] = ()) -> EvolutionCandidate:
    return EvolutionCandidate(
        candidate_id=cid,
        version=cid,
        artifact_hashes={},
        parent_ids=parents,
        ancestor_ids=parents,
        attempt_ids=(),
    )


def _score(pool: PersistentPool, cid: str, task_id: str, value: float) -> None:
    seq = pool.get(cid).cell(task_id, _MECH).rollout_count
    pool.record_score(
        cid,
        value,
        ScoreProvenance(
            task_id=task_id,
            mechanism_cluster_id=_MECH,
            trace_id=f"tr-{cid}-{task_id}-{seq}",
            rollout_seq=seq,
            analyzer_model_id="a",
            judge_model_id="j",
            blame_confidence=0.0,
            blame_stability=0.0,
        ),
    )


def _skipper_pool() -> PersistentPool:
    """The register's SV-2 scenario.

    ``base`` ran easy(0.9) and hard(0.1). ``candA`` ran easy(0.9) only, and is
    therefore *identical to base on the only task both attempted*. Coverage is
    neutralised via a zero beta in the tests below so that this isolates SV-2
    rather than measuring SV-3.
    """
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(_cand("base"))
    pool.add_candidate(_cand("candA", ("base",)))
    pool.record_preference("candA", 0.5, available=2)
    _score(pool, "base", "easy", 0.9)
    _score(pool, "base", "hard", 0.1)
    _score(pool, "candA", "easy", 0.9)
    return pool


def _outcome_only_config():
    """Weights that isolate outcome: no coverage reward, no inert constants.

    SV-3 is a separate defect; leaving beta at 0.20 would let coverage decide these
    cases and the test would no longer be about SV-2.
    """
    from agent_evolve.core.config import resolve_profile

    return resolve_profile(
        "minimal",
        champion_alpha=1.0,
        champion_beta=0.0,
        champion_gamma=0.0,
        champion_delta=0.0,
    )


# --------------------------------------------------------------------------- #
# 1. Skipping a task must not win
# --------------------------------------------------------------------------- #


def test_skipping_the_hard_task_does_not_win_the_championship() -> None:
    """The core SV-2 defect.

    ``candA`` tied base on ``easy`` and never ran ``hard``. Tying on the shared
    evidence is not an improvement, so base must hold. Selecting ``candA`` means
    the means were compared across different task sets.
    """
    pool = _skipper_pool()

    report = pool.select_champion(config=_outcome_only_config())

    assert report.candidate_id == "base"


def test_a_candidate_better_on_the_shared_cell_still_wins() -> None:
    """The fix must not simply freeze the base.

    Same shape as the skipper, except ``candB`` actually beats base on the shared
    task. A change that made base unbeatable would pass the test above while
    destroying selection, so this is the necessary counterpart.
    """
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(_cand("base"))
    pool.add_candidate(_cand("candB", ("base",)))
    pool.record_preference("candB", 0.5, available=2)
    _score(pool, "base", "easy", 0.5)
    _score(pool, "base", "hard", 0.1)
    _score(pool, "candB", "easy", 0.9)

    report = pool.select_champion(config=_outcome_only_config())

    assert report.candidate_id == "candB"


def test_a_candidate_worse_on_the_shared_cell_loses() -> None:
    """Regressing on shared evidence must lose, even with a smaller task set."""
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(_cand("base"))
    pool.add_candidate(_cand("candD", ("base",)))
    pool.record_preference("candD", 0.5, available=2)
    _score(pool, "base", "easy", 0.9)
    _score(pool, "base", "hard", 0.1)
    _score(pool, "candD", "easy", 0.4)

    report = pool.select_champion(config=_outcome_only_config())

    assert report.candidate_id == "base"


# --------------------------------------------------------------------------- #
# 2. The intersection is reported
# --------------------------------------------------------------------------- #


def test_the_report_states_how_many_cells_backed_the_decision() -> None:
    """Register: "report the intersection size alongside it".

    A champion chosen on one shared cell and one chosen on forty are not equally
    trustworthy, and a manifest that cannot distinguish them hides the difference.
    """
    pool = _skipper_pool()

    report = pool.select_champion(config=_outcome_only_config())

    assert hasattr(report, "comparable_cells"), (
        "ChampionReport must expose the comparable-cell count behind the decision"
    )
    # base vs candA share exactly {easy}.
    assert report.comparable_cells == 1


def test_a_full_overlap_reports_every_shared_cell() -> None:
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(_cand("base"))
    pool.add_candidate(_cand("candE", ("base",)))
    pool.record_preference("candE", 0.5, available=2)
    for task_id in ("t1", "t2", "t3"):
        _score(pool, "base", task_id, 0.4)
        _score(pool, "candE", task_id, 0.6)

    report = pool.select_champion(config=_outcome_only_config())

    assert report.candidate_id == "candE"
    assert report.comparable_cells == 3


# --------------------------------------------------------------------------- #
# 3. No shared evidence is not a win
# --------------------------------------------------------------------------- #


def test_a_candidate_sharing_no_cell_with_the_base_cannot_displace_it() -> None:
    """Disjoint evidence yields no verdict, so the incumbent holds.

    ``candF`` measured a task base never ran and scores well on it. There is no
    comparable evidence, and "no evidence" must not read as "better".
    """
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(_cand("base"))
    pool.add_candidate(_cand("candF", ("base",)))
    pool.record_preference("candF", 0.5, available=2)
    _score(pool, "base", "easy", 0.5)
    _score(pool, "candF", "other", 0.99)

    report = pool.select_champion(config=_outcome_only_config())

    assert report.candidate_id == "base"


def test_the_rollout_floor_is_honoured_for_comparability() -> None:
    """A cell below ``min_comparable_rollouts`` is not comparable evidence.

    Consistent with ``comparable_cells``, which ``dominates`` and
    ``pareto_frontier`` already use. ``candG`` has a single rollout on the shared
    task while the pool requires two, so its apparent 0.99 must not promote it.
    """
    pool = PersistentPool(min_comparable_rollouts=2)
    pool.add_base(_cand("base"))
    pool.add_candidate(_cand("candG", ("base",)))
    pool.record_preference("candG", 0.5, available=2)
    for _ in range(2):
        _score(pool, "base", "easy", 0.5)
    _score(pool, "candG", "easy", 0.99)  # one rollout only

    report = pool.select_champion(config=_outcome_only_config())

    assert report.candidate_id == "base"


# --------------------------------------------------------------------------- #
# 4. Determinism
# --------------------------------------------------------------------------- #


def test_selection_is_deterministic_under_pairwise_comparison() -> None:
    """Pairwise ranking must not depend on dict iteration order.

    Three candidates with partially disjoint evidence is exactly the shape that
    admits intransitivity, so the result has to be pinned to insertion order.
    """
    def build() -> PersistentPool:
        pool = PersistentPool(min_comparable_rollouts=1)
        pool.add_base(_cand("base"))
        for cid in ("c1", "c2", "c3"):
            pool.add_candidate(_cand(cid, ("base",)))
            pool.record_preference(cid, 0.5, available=2)
        _score(pool, "base", "easy", 0.5)
        _score(pool, "base", "hard", 0.5)
        _score(pool, "c1", "easy", 0.6)
        _score(pool, "c2", "hard", 0.7)
        _score(pool, "c3", "easy", 0.55)
        _score(pool, "c3", "hard", 0.55)
        return pool

    config = _outcome_only_config()
    first = build().select_champion(config=config).candidate_id
    for _ in range(5):
        assert build().select_champion(config=config).candidate_id == first
