"""SV-13c wiring — retirement fires from the production attempt loop.

``core/retirement.py`` decides; this covers the decision actually being *reached*
and *applied* by ``SequentialGepaRunner.run_attempt``.

**Cost, corrected downward by measurement.** The estimate handed to the user was
``2k`` judge calls **plus** ``k`` paired rollouts per accepted offspring. Reading
the loop shows the rollouts are already paid:

* ``build_issues`` rolls the **parent** on every coreset task (``orchestrator.py``
  ``:1466``, after SV-11 made the parent the observation subject).
* ``validate`` rolls the **child** on origin + every regression task, which is the
  same coreset (``:1814``).

Both trace sets exist at commit time and were simply discarded --
``ValidationResult`` keeps only ``trace_id``, not the trace. Retaining them makes
retirement cost ``2k`` **judge calls and zero extra rollouts**. A rollout is far
more expensive than a judge call in a live run, so this is the difference between
"acceptable" and "cheap".

**Retirement never blocks the edit.** A judge outage, a missing trace, or a raising
judge leaves the parent alive and the accepted candidate committed. The attempt's
result must not depend on an optional pool-shaping step succeeding.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_phase_6_orchestrator import _runner, _task  # type: ignore

_TASKS = (_task("task-a"), _task("task-b"))


class _Verdict:
    def __init__(self, score: float, available: bool = True) -> None:
        self.score = score
        self.available = available


class _RecordingJudge:
    """Always prefers the candidate; records every comparison."""

    def __init__(self, score: float = 0.8, available: bool = True) -> None:
        self._score = score
        self._available = available
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, task, baseline, candidate):  # noqa: ANN001
        self.calls.append((task.task_id, baseline.trace_id, candidate.trace_id))
        return _Verdict(self._score, self._available)


def _wired(judge=None, **kw):
    runner = _runner(seed=7, **kw)
    if judge is not None:
        runner.compare_preference = judge
    return runner


# --------------------------------------------------------------------------- #
# 1. The decision is reached and applied
# --------------------------------------------------------------------------- #


def test_a_preferred_child_retires_its_parent() -> None:
    """End of the chain: accepted edit -> judge prefers child -> parent retired."""
    judge = _RecordingJudge(score=0.9)
    runner = _wired(judge)

    outcome = runner.run_attempt(_TASKS)

    assert outcome.accepted, outcome.reason
    assert runner.pool.get(outcome.parent_candidate_id).retired is True
    assert runner.pool.get(outcome.parent_candidate_id).superseded_by == (
        outcome.result_candidate_id
    )


def test_a_child_the_judge_rejects_leaves_the_parent_alive() -> None:
    judge = _RecordingJudge(score=-0.7)
    runner = _wired(judge)

    outcome = runner.run_attempt(_TASKS)

    assert outcome.accepted, outcome.reason
    assert runner.pool.get(outcome.parent_candidate_id).retired is False


def test_a_tie_leaves_the_parent_alive() -> None:
    """Strict ``> 0``, consistent with the SV-4 promotion gate."""
    runner = _wired(_RecordingJudge(score=0.0))

    outcome = runner.run_attempt(_TASKS)

    assert runner.pool.get(outcome.parent_candidate_id).retired is False


def test_retirement_is_reported_on_the_attempt_outcome() -> None:
    """A pool that shrank must say so in the attempt record, not silently."""
    runner = _wired(_RecordingJudge(score=0.9))

    outcome = runner.run_attempt(_TASKS)

    assert outcome.retired_parent_id == outcome.parent_candidate_id


def test_no_retirement_is_reported_as_none() -> None:
    runner = _wired(_RecordingJudge(score=-0.5))

    outcome = runner.run_attempt(_TASKS)

    assert outcome.retired_parent_id is None


# --------------------------------------------------------------------------- #
# 2. Cost: judge calls per task, zero extra rollouts
# --------------------------------------------------------------------------- #


def test_one_judge_comparison_per_coreset_task() -> None:
    """``2k`` model calls for k tasks, since ``compare`` is the symmetric judge."""
    judge = _RecordingJudge(score=0.9)
    runner = _wired(judge)

    runner.run_attempt(_TASKS)

    assert {c[0] for c in judge.calls} == {t.task_id for t in _TASKS}
    assert len(judge.calls) == len(_TASKS)


def test_retirement_adds_no_rollouts() -> None:
    """The measurement that corrects the cost estimate.

    Both trace sets already exist -- parent from ``build_issues``, child from
    ``validate`` -- so reaching a retirement verdict must not roll anything again.
    """
    without = _wired()
    without.run_attempt(_TASKS)
    baseline_calls = without.adapter.rollout_calls  # type: ignore[attr-defined]

    with_judge = _wired(_RecordingJudge(score=0.9))
    with_judge.run_attempt(_TASKS)
    judged_calls = with_judge.adapter.rollout_calls  # type: ignore[attr-defined]

    assert judged_calls == baseline_calls, (
        f"retirement cost {judged_calls - baseline_calls} extra rollouts; the "
        "parent and child traces should both be reused"
    )


def test_the_parent_and_child_traces_are_distinct_slots() -> None:
    """Orientation: the two slots must carry different traces.

    Passing the same trace twice would make every verdict a self-comparison, and
    the judge would be measuring nothing. Slot *identity* (parent as baseline) is
    pinned by ``tests/test_retirement_decision.py``, which can assert on
    candidate ids directly.
    """
    judge = _RecordingJudge(score=0.9)
    runner = _wired(judge)

    runner.run_attempt(_TASKS)

    assert judge.calls, "no comparison was issued"
    for _task_id, baseline_trace_id, candidate_trace_id in judge.calls:
        assert baseline_trace_id != candidate_trace_id


# --------------------------------------------------------------------------- #
# 3. Retirement is optional and never breaks the attempt
# --------------------------------------------------------------------------- #


def test_without_a_judge_nothing_is_retired() -> None:
    """Default behaviour is unchanged: no judge injected, no retirement, and the
    attempt still succeeds. Keeps every existing offline path working."""
    runner = _wired()

    outcome = runner.run_attempt(_TASKS)

    assert outcome.accepted, outcome.reason
    assert runner.pool.get(outcome.parent_candidate_id).retired is False
    assert outcome.retired_parent_id is None


def test_an_unavailable_verdict_leaves_the_parent_alive() -> None:
    runner = _wired(_RecordingJudge(score=0.9, available=False))

    outcome = runner.run_attempt(_TASKS)

    assert outcome.accepted
    assert runner.pool.get(outcome.parent_candidate_id).retired is False


def test_a_raising_judge_does_not_lose_the_accepted_edit() -> None:
    """The edit is the expensive artifact; an optional pool-shaping step must not
    be able to discard it."""

    def boom(task, baseline, candidate):  # noqa: ANN001
        raise RuntimeError("judge exploded")

    runner = _wired(boom)

    outcome = runner.run_attempt(_TASKS)

    assert outcome.accepted, outcome.reason
    assert outcome.result_candidate_id is not None
    assert runner.pool.get(outcome.parent_candidate_id).retired is False


def test_a_rejected_edit_never_retires_the_parent() -> None:
    """No offspring, nothing to supersede. Retiring here would delete a live
    parent in exchange for nothing."""
    judge = _RecordingJudge(score=0.9)
    runner = _wired(judge, min_comparable_rollouts=1)
    # Force rejection by demanding an impossible net gain.
    runner.net_gain_threshold = 99.0

    outcome = runner.run_attempt(_TASKS)

    assert not outcome.accepted
    assert runner.pool.get(outcome.parent_candidate_id).retired is False
    assert judge.calls == []


# --------------------------------------------------------------------------- #
# 4. The pool actually shrinks toward a survivor
# --------------------------------------------------------------------------- #


def test_repeated_retirement_keeps_the_live_pool_bounded() -> None:
    """The proposal's purpose: the breeding population stops growing linearly.

    Without retirement six attempts leave seven live entries; with it, each
    superseded parent leaves, so the live count stays small while the *pool*
    keeps every entry for evidence.
    """
    runner = _wired(_RecordingJudge(score=0.9))
    for _ in range(6):
        runner.run_attempt(_TASKS)

    live = runner.pool.live_candidate_ids()
    total = len(runner.pool.all_entries())

    assert len(live) < total, (
        f"no entry was retired across six attempts: live={live}"
    )
    assert total >= 2, "evidence entries were deleted rather than retired"


def test_the_base_survives_when_it_is_the_only_live_entry() -> None:
    """The safety rail, reached through the production loop rather than directly."""
    runner = _wired(_RecordingJudge(score=0.9))
    for _ in range(6):
        runner.run_attempt(_TASKS)

    assert runner.pool.live_candidate_ids(), "the live pool was emptied"
