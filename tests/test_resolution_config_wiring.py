"""The champion configuration reaches resolution through the production path.

``resolve_final_candidate`` accepting a ``config`` argument is worth nothing if
``EvolutionStack.resolve_winner`` does not pass one. The core fix and its wiring
are separately breakable, so they are separately tested: ``test_resolution_config``
covers the pure function, this file covers the stack that calls it.

**Asserted through behaviour, not through the call.** A test that patched
``resolve_final_candidate`` and asserted ``config is not None`` would pass against
a stack that forwarded a config nobody read, and would also pass if
``select_champion`` ignored the argument. So these tests configure a floor that
*changes which candidate id wins* and assert the id.

The scenario is the one that matters operationally: no preference judge, so
resolution takes the aggregate fallback -- the path where a dropped floor stops
being a cosmetic issue and starts readmitting a candidate the operator
disqualified.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_evolve.core.contracts import EvolutionCandidate
from agent_evolve.core.pool import ScoreProvenance
from agent_evolve.pipeline import build_offline_stack

_MECH = "m0"
_CELLS = ("task-a", "task-b", "task-c")


def _cand(cid: str, parents: tuple[str, ...] = ()) -> EvolutionCandidate:
    return EvolutionCandidate(
        candidate_id=cid,
        version=cid,
        artifact_hashes={},
        parent_ids=parents,
        ancestor_ids=parents,
        attempt_ids=(),
    )


def _score(pool, cid: str, task_id: str, value: float) -> None:
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


def _stack_with_narrow_and_broad(*, floor: float | None):
    """An offline stack whose pool contains a narrow and a broad candidate.

    ``narrow`` measured one of three cells and scored 0.99 on it. ``broad``
    measured all three at 0.50. On the unfloored aggregate ``narrow`` wins; a 0.5
    coverage floor disqualifies it.

    No preference judge is bound, which forces the aggregate fallback -- the only
    path where ``config`` is read at all.
    """
    overrides: dict[str, object] = {}
    if floor is not None:
        overrides["champion_min_coverage_fraction"] = floor
    stack = build_offline_stack(
        preference_judge=None,
        config_overrides=overrides or None,
    )
    pool = stack.pool
    pool.min_comparable_rollouts = 1

    pool.add_candidate(_cand("broad", (pool.base.candidate_id,)))
    pool.add_candidate(_cand("narrow", (pool.base.candidate_id,)))
    pool.record_preference("broad", 0.5, available=2)
    pool.record_preference("narrow", 0.5, available=2)

    for task_id in _CELLS:
        _score(pool, pool.base.candidate_id, task_id, 0.10)
        _score(pool, "broad", task_id, 0.50)
    _score(pool, "narrow", "task-a", 0.99)
    return stack


def test_the_stack_forwards_the_coverage_floor_to_resolution() -> None:
    """With a floor configured, the under-measured candidate must not win.

    ``narrow`` holds the best score on the one cell it ran. If the stack drops the
    floor on the way to ``select_champion``, ``narrow`` is returned and this fails.
    """
    stack = _stack_with_narrow_and_broad(floor=0.5)

    resolution = stack.resolve_winner()

    assert resolution.method == "aggregate_fallback"
    assert resolution.candidate_id == "broad"


def test_without_a_configured_floor_the_narrow_candidate_wins() -> None:
    """Control: the floor is what changes the answer, not the fixture.

    Without this, the test above could be asserting a ranking that held regardless
    and would keep passing if the floor were dropped again.
    """
    stack = _stack_with_narrow_and_broad(floor=None)

    resolution = stack.resolve_winner()

    assert resolution.method == "aggregate_fallback"
    assert resolution.candidate_id == "narrow"


def test_the_memoised_winner_also_honours_the_floor() -> None:
    """``winner()`` is what export and measurement both read.

    ``resolve_winner`` being correct while the memoised accessor returned something
    else would mean the exported harness still ignored the floor.
    """
    stack = _stack_with_narrow_and_broad(floor=0.5)

    assert stack.winner().candidate_id == "broad"
    # Memoised: the second read is the same object, so export and measurement
    # cannot disagree about who won.
    assert stack.winner() is stack.winner()


def test_the_champion_version_honours_the_floor() -> None:
    """The version string handed to ``--harness`` must be the floored winner."""
    stack = _stack_with_narrow_and_broad(floor=0.5)

    assert stack.champion_version() == stack.pool.get("broad").version
