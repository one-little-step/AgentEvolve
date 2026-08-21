"""The aggregate fallback must honour the operator's champion configuration.

``resolve_final_candidate`` reaches ``select_champion`` only through
``_aggregate_fallback``, and that call originally passed no ``config``. Every
``champion_*`` value therefore reverted to its dataclass default on the one path
that still ranks by aggregate:

* ``champion_alpha/beta/gamma/delta`` silently became ``0.55/0.20/0.15/0.10``,
  so a reweighted aggregate was not the aggregate that ran.
* ``champion_min_coverage_fraction`` silently became ``0.0``, which is worse
  than a wrong weight: it is a *disqualification* the operator asked for and did
  not get.

The second is the reason this file exists. The floor exists to refuse a candidate
that measured too little to be trusted, and the fallback fires exactly when the
preference judge is unavailable -- a judge outage, a raising judge, an
unavailable verdict. So the guard was dropped at the precise moment it was the
only guard left. A run configured to demand 50% coverage would, on losing its
judge, quietly export a candidate measured on one cell of three.

These tests pin the configuration to the *behaviour it changes* -- which
candidate id wins -- and never to whether an argument was forwarded. A test
asserting ``config is not None`` at the call site would pass against a
``select_champion(config=config)`` that ignored its argument.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from agent_evolve.core.config import resolve_profile
from agent_evolve.core.contracts import EvolutionCandidate
from agent_evolve.core.pool import PersistentPool, ScoreProvenance
from agent_evolve.core.resolution import resolve_final_candidate

_MECH = "m0"

#: Three cells. A candidate measured on one of them has coverage 1/3.
_CELL_TASKS = ("task-a", "task-b", "task-c")


class _Task:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id


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


def _narrow_vs_broad_pool() -> PersistentPool:
    """A pool where the coverage floor is the only thing separating two winners.

    ``narrow`` measured one cell and scored well on it; ``broad`` measured all
    three and scored less well. Without a floor the aggregate prefers ``narrow``
    (this is the SV-2/SV-3 shape). With a 0.5 floor ``narrow``'s 1/3 coverage
    disqualifies it and ``broad`` must win instead.

    Both carry a positive recorded preference so the SV-4 gate keeps them
    eligible; the base is deliberately scored lowest so that it winning would
    signal the fallback collapsed to the base rather than honouring the floor.
    """
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(_cand("base"))
    pool.add_candidate(_cand("broad", ("base",)))
    pool.add_candidate(_cand("narrow", ("base",)))
    pool.record_preference("broad", 0.5, available=2)
    pool.record_preference("narrow", 0.5, available=2)

    for task_id in _CELL_TASKS:
        _score(pool, "base", task_id, 0.10)
        _score(pool, "broad", task_id, 0.50)
    _score(pool, "narrow", "task-a", 0.99)
    return pool


def test_the_configured_coverage_floor_survives_the_aggregate_fallback() -> None:
    """A configured floor must still disqualify when no judge is available.

    ``compare=None`` forces ``_aggregate_fallback``. With
    ``champion_min_coverage_fraction=0.5`` the single-cell ``narrow`` candidate is
    below the floor, so the winner must be ``broad``. Getting ``narrow`` here means
    the floor was dropped on the way to ``select_champion``.
    """
    pool = _narrow_vs_broad_pool()
    config = resolve_profile("minimal", champion_min_coverage_fraction=0.5)

    result = resolve_final_candidate(
        pool,
        tasks=[_Task(t) for t in _CELL_TASKS],
        traces={},
        compare=None,
        config=config,
    )

    assert result.method == "aggregate_fallback"
    assert result.candidate_id == "broad"


def test_without_a_floor_the_narrow_candidate_does_win() -> None:
    """Control for the test above: the floor is what changes the answer.

    If ``narrow`` did not win in the unconfigured case, the previous test would
    pass for the wrong reason -- it would be asserting a ranking that held anyway
    rather than one the floor produced.
    """
    pool = _narrow_vs_broad_pool()

    result = resolve_final_candidate(
        pool,
        tasks=[_Task(t) for t in _CELL_TASKS],
        traces={},
        compare=None,
    )

    assert result.method == "aggregate_fallback"
    assert result.candidate_id == "narrow"


def test_the_configured_weights_reach_the_reported_aggregate() -> None:
    """The weights are forwarded, but they no longer decide.

    This asserted the opposite when it was written: that a coverage-heavy weighting
    flips the fallback's winner to ``broad``. SV-2 then replaced aggregate ranking
    with pairwise intersection comparison, so no weighting can flip a winner --
    ``narrow`` is not worse than ``broad`` on the one cell both measured, so it
    holds regardless of ``beta``.

    What still matters is that the configured weights reach the *reported*
    aggregate: the manifest publishes the number, and a reader must be able to
    reproduce it from the flags the run was given. That is what this now pins. The
    coverage *floor* remains behaviourally load-bearing and is covered above.
    """
    pool = _narrow_vs_broad_pool()
    config = resolve_profile(
        "minimal",
        champion_alpha=0.05,
        champion_beta=0.95,
        champion_gamma=0.0,
        champion_delta=0.0,
        champion_min_coverage_fraction=0.0,
    )

    result = resolve_final_candidate(
        pool,
        tasks=[_Task(t) for t in _CELL_TASKS],
        traces={},
        compare=None,
        config=config,
    )

    assert result.method == "aggregate_fallback"
    # Ranking is pairwise now, so the narrow candidate is not displaced by weight.
    assert result.candidate_id == "narrow"
    # ...but the configured weights did reach select_champion: the reported
    # aggregate is computed from them, not from the 0.55/0.20 defaults.
    report = pool.select_champion(config=config)
    assert report.aggregate == pytest.approx(
        0.05 * report.outcome + 0.95 * report.coverage
    )


def test_a_judge_outage_still_honours_the_floor() -> None:
    """The realistic failure: a judge that raises, with a floor configured.

    This is the case the defect actually endangered. The judge is present but
    faulting, so resolution falls back mid-ladder -- and the floor must still
    apply on the way out.
    """

    def boom(task, baseline, candidate):  # noqa: ANN001
        raise RuntimeError("judge exploded")

    pool = _narrow_vs_broad_pool()
    tasks = [_Task(t) for t in _CELL_TASKS]
    traces = {
        cid: {t.task_id: object() for t in tasks}
        for cid in pool.live_candidate_ids()
    }
    config = resolve_profile("minimal", champion_min_coverage_fraction=0.5)

    result = resolve_final_candidate(
        pool, tasks=tasks, traces=traces, compare=boom, config=config
    )

    assert result.method == "aggregate_fallback"
    assert "error" in result.reason
    assert result.candidate_id == "broad"


def test_config_is_optional() -> None:
    """``config`` must stay keyword-optional.

    ``resolve_final_candidate`` is called from tests and from any caller without a
    resolved profile; requiring the argument would be a breaking change to a
    core-neutral signature.
    """
    pool = _narrow_vs_broad_pool()

    result = resolve_final_candidate(
        pool, tasks=[_Task(t) for t in _CELL_TASKS], traces={}, compare=None
    )

    assert result.candidate_id  # resolved without a config argument
