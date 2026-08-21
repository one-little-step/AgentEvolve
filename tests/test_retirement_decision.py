"""SV-13c — retirement is decided by the RHO pairwise preference judge.

The mechanism (``tests/test_generational_retirement.py``) enforces the structural
invariants. This module covers *who decides* and *what it costs*.

**Why the judge and not ``dominates()``.** Numeric dominance compares mean cell
scores. It cannot see that a child solved the parent's actual failure *mechanism*,
which is the thing per-parent issue tracking (SV-10) targets and the thing that
makes "the parent's faults are fixed in the child" true rather than hopeful. The
pairwise judge reads trajectories, so it can. Using dominance for retirement and
preference for final selection would also mean a candidate could be retired by one
standard and promoted by another.

**Cost, accepted explicitly.** The judge compares trajectories *on a task*, so
deciding "does the child supersede the parent?" needs both sides rolled out on the
same k coreset tasks and ``2`` symmetric calls per task -- ``2k`` calls per
accepted offspring. Recorded decision: acceptable, because k is 5-10, so 10-20
calls. The symmetric pair is not negotiable; a single call reintroduces exactly the
position bias ``compare_symmetric`` exists to cancel.

**Conservative on failure.** No verdict, a tie, or a judge outage all mean *no
retirement*. An unavailable judgement is not evidence of supersession, and a judge
outage must never silently shrink the breeding population. This mirrors the SV-4
promotion gate's reading of ``preference is None``.

``core/`` may not import an adapter, so the judge arrives as an injected
``compare(task, baseline_trace, candidate_trace) -> verdict`` callable -- the same
seam ``core/rho/rounds.py:240`` already uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_evolve.core.contracts import EvolutionCandidate
from agent_evolve.core.pool import PersistentPool, ScoreProvenance
from agent_evolve.core.retirement import (
    RetirementDecision,
    decide_retirement,
)

_MECH = "m0"


class _Verdict:
    """Minimal stand-in for the adapter's ``PreferenceVerdict``."""

    def __init__(self, score: float, available: bool = True) -> None:
        self.score = score
        self.available = available


class _Judge:
    """Records every comparison so cost and orientation can be asserted."""

    def __init__(self, *scores: float, available: bool = True) -> None:
        self._scores = list(scores)
        self._available = available
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, task, baseline, candidate):  # noqa: ANN001
        self.calls.append(
            (task.task_id, baseline.candidate_id, candidate.candidate_id)
        )
        score = self._scores.pop(0) if self._scores else 0.0
        return _Verdict(score, self._available)


class _Trace:
    def __init__(self, candidate_id: str) -> None:
        self.candidate_id = candidate_id
        self.trace_id = f"tr-{candidate_id}"


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


def _pool() -> PersistentPool:
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(_cand("base"))
    pool.add_candidate(_cand("child", ("base",)))
    for task in ("task-a", "task-b"):
        for cid, value in (("base", 0.2), ("child", 1.0)):
            seq = pool.get(cid).cell(task, _MECH).rollout_count
            pool.record_score(
                cid,
                value,
                ScoreProvenance(
                    task_id=task,
                    mechanism_cluster_id=_MECH,
                    trace_id=f"tr-{cid}-{task}",
                    rollout_seq=seq,
                    analyzer_model_id="a",
                    judge_model_id="j",
                    blame_confidence=1.0,
                    blame_stability=1.0,
                ),
            )
    return pool


_TASKS = (_Task("task-a"), _Task("task-b"))


def _traces(cid: str):
    return {t.task_id: _Trace(cid) for t in _TASKS}


# --------------------------------------------------------------------------- #
# 1. The decision
# --------------------------------------------------------------------------- #


def test_a_child_the_judge_prefers_retires_its_parent() -> None:
    judge = _Judge(0.6, 0.8)

    decision = decide_retirement(
        parent_id="base",
        child_id="child",
        tasks=_TASKS,
        parent_traces=_traces("base"),
        child_traces=_traces("child"),
        compare=judge,
    )

    assert decision.should_retire is True
    assert decision.mean_preference > 0.0


def test_a_child_the_judge_does_not_prefer_leaves_the_parent_alive() -> None:
    judge = _Judge(-0.5, -0.2)

    decision = decide_retirement(
        parent_id="base",
        child_id="child",
        tasks=_TASKS,
        parent_traces=_traces("base"),
        child_traces=_traces("child"),
        compare=judge,
    )

    assert decision.should_retire is False


def test_a_measured_tie_does_not_retire_the_parent() -> None:
    """Strictly ``> 0``, matching the SV-4 promotion gate. A tie is not evidence
    that the child superseded anything."""
    judge = _Judge(0.0, 0.0)

    decision = decide_retirement(
        parent_id="base",
        child_id="child",
        tasks=_TASKS,
        parent_traces=_traces("base"),
        child_traces=_traces("child"),
        compare=judge,
    )

    assert decision.should_retire is False
    assert decision.mean_preference == 0.0


def test_a_mixed_verdict_is_decided_on_the_mean() -> None:
    """One task lost, one won more: the aggregate decides, so a single bad task
    does not veto an otherwise clear supersession."""
    judge = _Judge(-0.2, 0.8)

    decision = decide_retirement(
        parent_id="base",
        child_id="child",
        tasks=_TASKS,
        parent_traces=_traces("base"),
        child_traces=_traces("child"),
        compare=judge,
    )

    assert round(decision.mean_preference, 6) == 0.3
    assert decision.should_retire is True


# --------------------------------------------------------------------------- #
# 2. Conservative on missing evidence
# --------------------------------------------------------------------------- #


def test_an_unavailable_verdict_never_retires() -> None:
    """A judge outage must not shrink the breeding population."""
    judge = _Judge(0.9, 0.9, available=False)

    decision = decide_retirement(
        parent_id="base",
        child_id="child",
        tasks=_TASKS,
        parent_traces=_traces("base"),
        child_traces=_traces("child"),
        compare=judge,
    )

    assert decision.should_retire is False
    assert decision.mean_preference is None
    assert "unavailable" in decision.reason


def test_a_missing_trace_never_retires() -> None:
    """A task the child could not be rolled out on yields no comparison, so the
    evidence is incomplete and the parent survives."""
    judge = _Judge(0.9)

    decision = decide_retirement(
        parent_id="base",
        child_id="child",
        tasks=_TASKS,
        parent_traces=_traces("base"),
        child_traces={"task-a": _Trace("child")},  # task-b missing
        compare=judge,
    )

    assert decision.should_retire is False
    assert "incomplete" in decision.reason


def test_no_tasks_never_retires() -> None:
    decision = decide_retirement(
        parent_id="base",
        child_id="child",
        tasks=(),
        parent_traces={},
        child_traces={},
        compare=_Judge(),
    )

    assert decision.should_retire is False


def test_a_raising_judge_is_contained_and_never_retires() -> None:
    """A judge exception must not abort the attempt that produced a valid edit."""

    def boom(task, baseline, candidate):  # noqa: ANN001
        raise RuntimeError("judge exploded")

    decision = decide_retirement(
        parent_id="base",
        child_id="child",
        tasks=_TASKS,
        parent_traces=_traces("base"),
        child_traces=_traces("child"),
        compare=boom,
    )

    assert decision.should_retire is False
    assert "error" in decision.reason


# --------------------------------------------------------------------------- #
# 3. Cost and orientation
# --------------------------------------------------------------------------- #


def test_one_comparison_per_task_is_issued() -> None:
    """The injected ``compare`` is the *symmetric* judge, which internally spends
    2 calls. This asserts one comparison per task -- 2k model calls for k tasks."""
    judge = _Judge(0.5, 0.5)

    decide_retirement(
        parent_id="base",
        child_id="child",
        tasks=_TASKS,
        parent_traces=_traces("base"),
        child_traces=_traces("child"),
        compare=judge,
    )

    assert len(judge.calls) == len(_TASKS)
    assert {c[0] for c in judge.calls} == {"task-a", "task-b"}


def test_the_parent_is_the_baseline_and_the_child_is_the_candidate() -> None:
    """Orientation matters: swapping the slots inverts the sign, which would
    retire exactly the wrong entry."""
    judge = _Judge(0.5, 0.5)

    decide_retirement(
        parent_id="base",
        child_id="child",
        tasks=_TASKS,
        parent_traces=_traces("base"),
        child_traces=_traces("child"),
        compare=judge,
    )

    for _task_id, baseline_id, candidate_id in judge.calls:
        assert baseline_id == "base"
        assert candidate_id == "child"


def test_the_decision_is_reported_not_applied() -> None:
    """``decide_retirement`` is pure: it returns a decision and mutates nothing.

    Applying it is the caller's job, so a dry run can measure what *would* be
    retired without shrinking the pool.
    """
    pool = _pool()
    judge = _Judge(0.9, 0.9)

    decision = decide_retirement(
        parent_id="base",
        child_id="child",
        tasks=_TASKS,
        parent_traces=_traces("base"),
        child_traces=_traces("child"),
        compare=judge,
    )

    assert isinstance(decision, RetirementDecision)
    assert decision.should_retire is True
    assert pool.get("base").retired is False
    assert pool.live_candidate_ids() == ("base", "child")
