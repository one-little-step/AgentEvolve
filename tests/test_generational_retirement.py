"""SV-13 — generational retirement: a superseded parent leaves the gene pool.

**The proposal.** Offspring are targeted at their parent's diagnosed faults, so a
child that provably supersedes its parent makes the parent redundant as a
*breeding* candidate. Retiring it caps the Pareto pool, concentrates selection
pressure on the frontier, and gives a natural terminal condition: if the live pool
shrinks to one, that one is the winner.

**Three decisions this encodes, each recorded rather than assumed.**

1. **Soft retirement, not pruning.** A retired entry is excluded from parent
   sampling, the Pareto frontier and champion selection, but **its score cells stay
   in the tensor**. Hard-deleting the entry would destroy the comparable cells
   cross-candidate entropy needs (``core/entropy.py:110`` wants 3 comparable
   candidates per cell) and the negative evidence a later analysis wants -- it
   would make SV-12 worse in exchange for memory nobody asked to save.
   ``AGENTS.md`` requires base plus every proposal to be *retained*; retention of
   evidence is preserved exactly, only breeding eligibility changes.

2. **The judge decides, not the arithmetic.** Retirement is conditioned on the
   RHO symmetric pairwise preference, the same instrument that gates promotion
   (SV-4) and resolves the final winner. Numeric ``dominates()`` was the
   alternative and is deliberately *not* used as the trigger: it compares mean
   cell scores, which cannot see that a child solved the parent's actual failure
   *mechanism*. Using one criterion for retirement and another for selection would
   let a candidate be retired by one standard and promoted by the other.

3. **No verdict means no retirement.** An unavailable judgement is not evidence of
   supersession. This mirrors the SV-4 gate's conservative reading: a judge outage
   must never silently shrink the pool.

**The base is never retired.** It is the comparison incumbent and the guaranteed
fallback for ``select_parent``; retiring it could empty the eligible set and turn
an ordinary "nothing improved" outcome into a crash.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from agent_evolve.core.contracts import EvolutionCandidate
from agent_evolve.core.pool import PersistentPool, ScoreProvenance

_MECH = "m0"


def _prov(task_id: str, seq: int) -> ScoreProvenance:
    return ScoreProvenance(
        task_id=task_id,
        mechanism_cluster_id=_MECH,
        trace_id=f"tr-{task_id}-{seq}",
        rollout_seq=seq,
        analyzer_model_id="fake-analyzer",
        judge_model_id="fake-judge",
        blame_confidence=1.0,
        blame_stability=1.0,
    )


def _cand(cid: str, parents: tuple[str, ...] = ()) -> EvolutionCandidate:
    return EvolutionCandidate(
        candidate_id=cid,
        version=cid,
        artifact_hashes={},
        parent_ids=parents,
        ancestor_ids=parents,
        attempt_ids=(),
    )


def _pool() -> PersistentPool:
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(_cand("base"))
    return pool


def _score(pool: PersistentPool, cid: str, task: str, value: float) -> None:
    seq = pool.get(cid).cell(task, _MECH).rollout_count
    pool.record_score(cid, value, _prov(task, seq))


def _family() -> PersistentPool:
    """base -> child, both measured on the same two cells."""
    pool = _pool()
    pool.add_candidate(_cand("child", ("base",)))
    _score(pool, "base", "task-a", 0.2)
    _score(pool, "base", "task-b", 0.9)
    _score(pool, "child", "task-a", 1.0)
    _score(pool, "child", "task-b", 0.9)
    return pool


# --------------------------------------------------------------------------- #
# 1. The retirement flag and what it excludes
# --------------------------------------------------------------------------- #


def test_a_new_entry_is_not_retired() -> None:
    pool = _family()

    assert pool.get("child").retired is False
    assert pool.get("base").retired is False


def test_retiring_excludes_the_entry_from_parent_sampling() -> None:
    """The point of the mechanism: a retired parent stops breeding."""
    pool = _family()
    _score(pool, "child", "task-a", 1.0)  # give child a winning cell
    pool.retire("base", superseded_by="child")

    assert "base" not in pool.parent_frequencies()


def test_retiring_excludes_the_entry_from_the_pareto_frontier() -> None:
    """A retired entry must not occupy a frontier slot, or the cap does nothing."""
    pool = _family()
    pool.retire("base", superseded_by="child")

    assert "base" not in pool.pareto_frontier()


def test_retiring_excludes_the_entry_from_champion_selection() -> None:
    """A retired candidate must not be exported as the champion."""
    pool = _family()
    pool.record_preference("child", 0.5, available=2)
    pool.retire("base", superseded_by="child")

    report = pool.select_champion()

    assert report.entry.candidate_id == "child"
    assert "base" in report.disqualifications


# --------------------------------------------------------------------------- #
# 2. Evidence survives retirement -- the whole reason it is soft
# --------------------------------------------------------------------------- #


def test_a_retired_entry_keeps_its_score_cells() -> None:
    """Hard-pruning would delete these. Entropy comparability depends on them."""
    pool = _family()
    before = dict(pool.get("base").score_tensor)

    pool.retire("base", superseded_by="child")

    after = pool.get("base").score_tensor
    assert set(after) == set(before)
    assert all(after[k].rollout_count > 0 for k in after)


def test_a_retired_entry_is_still_comparable_for_evidence() -> None:
    """The retired parent must remain a comparison partner, or retiring it
    destroys exactly the cross-candidate evidence SV-12 is starved of."""
    pool = _family()
    pool.retire("base", superseded_by="child")

    assert pool.comparable_cells("child", "base")
    assert pool.is_comparable("child", "base")


def test_a_retired_entry_is_still_addressable_and_listed() -> None:
    """Retirement is a status, not a deletion; an audit must still find it."""
    pool = _family()
    pool.retire("base", superseded_by="child")

    assert pool.get("base").candidate_id == "base"
    assert "base" in [e.candidate_id for e in pool.all_entries()]


def test_retirement_records_who_superseded_the_entry() -> None:
    """Provenance: a pool that shrank must be able to say why."""
    pool = _family()
    pool.retire("base", superseded_by="child")

    assert pool.get("base").superseded_by == "child"


# --------------------------------------------------------------------------- #
# 3. Safety rails
# --------------------------------------------------------------------------- #


def test_the_base_cannot_be_retired_when_it_is_the_only_live_entry() -> None:
    """Emptying the live pool would make select_parent and champion selection
    raise on an ordinary "nothing improved" run."""
    pool = _pool()

    with pytest.raises(ValueError, match="only live"):
        pool.retire("base", superseded_by="nobody")


def test_retiring_an_unknown_candidate_raises() -> None:
    pool = _family()

    with pytest.raises(KeyError):
        pool.retire("ghost", superseded_by="child")


def test_a_candidate_cannot_be_superseded_by_itself() -> None:
    pool = _family()

    with pytest.raises(ValueError, match="itself"):
        pool.retire("child", superseded_by="child")


def test_retiring_twice_is_idempotent_not_an_error() -> None:
    """A retry or a replayed commit must not crash the run."""
    pool = _family()
    pool.retire("base", superseded_by="child")
    pool.retire("base", superseded_by="child")

    assert pool.get("base").retired is True


def test_retirement_never_empties_the_live_pool() -> None:
    """The invariant that keeps the terminal condition meaningful."""
    pool = _family()
    pool.retire("base", superseded_by="child")

    with pytest.raises(ValueError, match="only live"):
        pool.retire("child", superseded_by="base")

    assert pool.live_candidate_ids() == ("child",)


# --------------------------------------------------------------------------- #
# 4. The terminal condition
# --------------------------------------------------------------------------- #


def test_live_candidate_ids_excludes_retired_entries() -> None:
    pool = _family()

    assert set(pool.live_candidate_ids()) == {"base", "child"}

    pool.retire("base", superseded_by="child")

    assert pool.live_candidate_ids() == ("child",)


def test_a_single_live_candidate_is_the_outright_winner() -> None:
    """"If the pool shrinks to 1, declare that the winner" -- no judge needed."""
    pool = _family()
    pool.retire("base", superseded_by="child")

    assert pool.has_sole_survivor() is True
    assert pool.sole_survivor().candidate_id == "child"


def test_more_than_one_live_candidate_has_no_sole_survivor() -> None:
    """Two live entries must fall through to the pairwise ladder instead."""
    pool = _family()

    assert pool.has_sole_survivor() is False
    with pytest.raises(ValueError, match="not a sole survivor"):
        pool.sole_survivor()
