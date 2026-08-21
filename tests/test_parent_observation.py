"""SV-11 — ``build_issues`` must diagnose the selected parent, not always base.

**The documented claim was partly wrong, and measuring it changed the fix.** The
register said "no candidate is ever mechanism-analyzed", naming
``orchestrator.py:541`` and ``:1441``. Executing the production path shows:

* ``orchestrator.py:541`` is inside ``Orchestrator.run_iteration``, which has
  **zero callers** anywhere in ``src/``, ``scripts/`` or ``tests/``. It is dead
  code alongside ``SequentialGepaRunner.run()``; production uses
  ``run_attempt``. Editing it would have changed nothing.
* ``run_attempt`` **already** rolls out and analyzes ``select_parent()``
  (``orchestrator.py:1971``), so candidates are not wholly unobserved.

The real defect is at the single remaining site, ``orchestrator.py:1451`` inside
``build_issues``: it hardcodes ``self.pool.base`` for the rollout, the write set,
the artifact inventory *and* the score attribution. Measured over six production
attempts on a two-task pool:

    base                                        rollouts=12
    base-v0+att-i001-s0000                      rollouts=2
    base-v0+att-i001-s0000+att-i002-s0001       rollouts=2
    ... (7 entries total)

    cell ('task-a', 'mechanism-default'): 1 comparable candidate  (floor=3)
    cell ('task-b', 'mechanism-default'): 1 comparable candidate  (floor=3)

Base absorbs every re-observation while each candidate stays frozen at the two
rollouts its own attempt produced, so **no cell ever reaches the entropy floor of
3** (``core/entropy.py:110``). That is SV-12, and it is a direct consequence.

Decision recorded: **observe the selected parent instead of base** -- the
cost-neutral option. The rollout count per call is unchanged; only the subject
moves. Base keeps no guaranteed refresh, which is the accepted trade, and remains
reachable because ``select_parent`` returns base whenever no candidate holds
winning-cell evidence.

Assertions are behavioural: which version the adapter rolls out, and which
entry's cells receive the scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_phase_6_orchestrator import _runner, _task  # type: ignore

_TASKS = (_task("task-a"), _task("task-b"))


def _rollouts(runner, candidate_id: str) -> int:
    return sum(
        c.rollout_count
        for c in runner.pool.get(candidate_id).score_tensor.values()
    )


def _comparable_per_cell(runner, min_rollouts: int = 2) -> dict[tuple, int]:
    """How many candidates hold comparable evidence in each cell.

    Mirrors the entropy precondition: a cell may only drive cross-candidate
    diversity once enough candidates have been measured *in that same cell*.
    """
    cells: dict[tuple, int] = {}
    for entry in runner.pool.all_entries():
        for key, cell in entry.score_tensor.items():
            if cell.rollout_count >= min_rollouts:
                cells[key] = cells.get(key, 0) + 1
    return cells


# --------------------------------------------------------------------------- #
# 1. The fallback survives
# --------------------------------------------------------------------------- #


def test_a_fresh_pool_still_diagnoses_the_base() -> None:
    """With no candidate evidence ``select_parent`` returns base, so the first
    attempt is unchanged. Guards against 'fixing' this into never observing base."""
    runner = _runner(seed=7)

    issues = runner.build_issues(_TASKS)

    assert issues, "no issues built on a fresh pool"
    assert _rollouts(runner, "base") > 0, "base was not observed on a fresh pool"


def test_the_first_attempt_is_still_parented_to_base() -> None:
    runner = _runner(seed=7)

    outcome = runner.run_attempt(_TASKS)

    assert outcome.parent_candidate_id == "base"


# --------------------------------------------------------------------------- #
# 2. The defect: base absorbs every re-observation
# --------------------------------------------------------------------------- #


def test_base_does_not_accumulate_every_reobservation() -> None:
    """The measurement that proves the defect.

    Before the fix, six attempts left base with 12 rollouts and each candidate
    with 2, because every ``build_issues`` call re-rolled base regardless of the
    selected parent. Base must not outgrow the pool once a candidate is the
    parent.
    """
    runner = _runner(seed=7)
    for _ in range(6):
        runner.run_attempt(_TASKS)

    base_rollouts = _rollouts(runner, "base")
    others = [
        _rollouts(runner, e.candidate_id)
        for e in runner.pool.all_entries()
        if not e.is_base
    ]
    assert others, "no candidates were produced"
    assert base_rollouts <= max(others) * 2, (
        f"base accumulated {base_rollouts} rollouts against a candidate maximum "
        f"of {max(others)}: observation is still pinned to base"
    )


def test_the_selected_parent_is_the_diagnosed_subject() -> None:
    """``build_issues`` must roll out whoever ``select_parent`` returns."""
    runner = _runner(seed=7)
    runner.run_attempt(_TASKS)  # produce a candidate with winning evidence
    parent = runner.select_parent()
    assert not parent.is_base, "fixture did not produce a non-base parent"
    before = _rollouts(runner, parent.candidate_id)

    runner.build_issues(_TASKS)

    after = _rollouts(runner, parent.candidate_id)
    assert after > before, (
        f"selected parent {parent.candidate_id} gained no rollouts "
        f"({before} -> {after}); build_issues still observes base"
    )


def test_scores_are_attributed_to_the_observed_parent() -> None:
    """Attribution. Recording a candidate's rollout against base would corrupt
    base's evidence and make the two indistinguishable in the tensor."""
    runner = _runner(seed=7)
    runner.run_attempt(_TASKS)
    parent = runner.select_parent()
    assert not parent.is_base
    base_before = _rollouts(runner, "base")

    runner.build_issues(_TASKS)

    assert _rollouts(runner, "base") == base_before, (
        "base gained rollouts while a candidate was the selected parent"
    )


def test_issues_carry_the_observed_parents_writable_artifacts() -> None:
    """The write set and inventory must follow the parent too.

    Diagnosing candidate X while offering base's artifacts would attribute X's
    mechanism to the wrong surfaces -- a worse defect than the original.

    The parent is forced to be a *failing* candidate here. With the offline
    adapter an accepted candidate scores a perfect 1.0, which correctly yields no
    issues at all, and a vacuous ``for`` loop would assert nothing.
    """
    runner = _runner(seed=7)
    parent = runner.select_parent()
    expected = set(runner._writable_artifact_ids(parent.version))

    issues = runner.build_issues(_TASKS)

    assert issues, "no issues built for a failing parent"
    for issue in issues:
        assert set(issue.writable_artifact_ids) <= expected, (
            f"issue {issue.issue_id} references artifacts outside the observed "
            f"parent's write set: {set(issue.writable_artifact_ids) - expected}"
        )


# --------------------------------------------------------------------------- #
# 3. Cost neutrality
# --------------------------------------------------------------------------- #


def test_observation_cost_per_build_issues_call_is_unchanged() -> None:
    """Cost-neutrality is the entire basis of the chosen option.

    One rollout per task either way: the subject changes, the count does not.
    """
    fresh = _runner(seed=11)
    fresh.build_issues(_TASKS)
    base_selected_calls = fresh.adapter.rollout_calls  # type: ignore[attr-defined]

    advanced = _runner(seed=11)
    advanced.run_attempt(_TASKS)
    calls_before = advanced.adapter.rollout_calls  # type: ignore[attr-defined]
    advanced.build_issues(_TASKS)
    candidate_selected_calls = (
        advanced.adapter.rollout_calls - calls_before  # type: ignore[attr-defined]
    )

    assert candidate_selected_calls == base_selected_calls, (
        f"observation cost changed: base-selected={base_selected_calls} "
        f"candidate-selected={candidate_selected_calls}"
    )


# --------------------------------------------------------------------------- #
# 4. SV-12: the entropy floor becomes reachable
# --------------------------------------------------------------------------- #


def test_re_observation_no_longer_piles_onto_a_single_entry() -> None:
    """SV-12's precondition, stated as what is actually verifiable offline.

    ``core/entropy.py:110`` requires 3 comparable candidates in a cell before it
    may drive DPP diversity. This test does **not** claim the floor is now met:
    with the offline adapter an accepted candidate scores a perfect 1.0, so
    evolution correctly stops and only one candidate is ever produced. Reaching
    the floor needs a task set the parent keeps partially failing, which is a
    live-run property.

    What is verifiable, and what SV-11 actually broke, is the *distribution*: the
    old code funnelled every re-observation into base (measured: base 12,
    candidates 2 each), so the comparable count per cell could never rise no
    matter how long a run continued. Observation now follows the parent, so
    rollouts land on whichever entry is being worked on.
    """
    runner = _runner(seed=7)
    for _ in range(6):
        runner.run_attempt(_TASKS)

    per_entry = {
        e.candidate_id: sum(c.rollout_count for c in e.score_tensor.values())
        for e in runner.pool.all_entries()
    }
    base_rollouts = per_entry["base"]
    candidate_rollouts = [v for k, v in per_entry.items() if k != "base"]

    assert candidate_rollouts, "no candidate was produced"
    assert max(candidate_rollouts) > base_rollouts, (
        "base still holds more rollout evidence than any candidate, so "
        f"re-observation is still funnelling into base: {per_entry}"
    )
