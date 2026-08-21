"""Generational retirement decisions (SV-13).

Agent-neutral by construction: the preference judge arrives as an injected
``compare`` callable, the same seam ``core/rho/rounds.py`` uses. This module must
never import an adapter, ``cuga``, or ``litellm``.

**What this decides.** Whether an accepted offspring has *superseded* its parent,
such that the parent should leave the breeding population. The premise is that an
offspring is generated to fix its parent's diagnosed faults, so a child that the
judge prefers on the coreset has made the parent redundant as a *parent* --
evolution should not keep spending rollouts breeding from a version its own
descendant has already improved on.

**Why the pairwise judge rather than numeric dominance.** Dominance compares mean
cell scores; it cannot see whether the child solved the parent's failure
*mechanism*. The pairwise judge reads trajectories, so it can. Using dominance here
and preference for final selection would also split the criterion: a candidate
could be retired by one standard and promoted by the other.

**Cost.** ``compare`` is the *symmetric* judge, which spends 2 model calls per
invocation to cancel position bias. One invocation per coreset task means ``2k``
model calls per accepted offspring. Recorded as acceptable for k of 5-10.

**Failure is conservative.** No verdict, an incomplete trace set, a tie, or a
raising judge all mean *no retirement*. Missing evidence is not evidence of
supersession, and a judge outage must never silently shrink the pool. This mirrors
the SV-4 promotion gate's treatment of ``preference is None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

__all__ = ["RetirementDecision", "decide_retirement"]


class _HasTaskId(Protocol):
    """Anything carrying a task id. Read-only, so a frozen task satisfies it."""

    @property
    def task_id(self) -> str: ...


# compare(task, baseline_trace, candidate_trace) -> verdict with .score/.available
CompareFn = Callable[[Any, Any, Any], Any]


@dataclass(frozen=True, slots=True)
class RetirementDecision:
    """The verdict on one parent/child pair. Pure data; nothing is mutated.

    Returned rather than applied so a caller can log or dry-run what *would* be
    retired without shrinking the pool.
    """

    parent_id: str
    child_id: str
    should_retire: bool
    reason: str
    #: Mean oriented preference of child over parent, or ``None`` when no
    #: complete verdict was obtained. ``None`` is deliberately distinct from
    #: ``0.0``: a tie is a measurement, a missing verdict is not.
    mean_preference: float | None = None
    #: Tasks that produced a usable verdict, and tasks that did not.
    judged: int = 0
    unavailable: int = 0


def decide_retirement(
    *,
    parent_id: str,
    child_id: str,
    tasks: Sequence[_HasTaskId],
    parent_traces: Mapping[str, Any],
    child_traces: Mapping[str, Any],
    compare: CompareFn,
) -> RetirementDecision:
    """Decide whether ``child_id`` has superseded ``parent_id``.

    Orientation is fixed and load-bearing: the **parent is the baseline** and the
    **child is the candidate**, so a positive score means the child is preferred.
    Swapping the slots inverts the sign and would retire the wrong entry.

    Every task must yield a usable verdict. A partial comparison is treated as
    incomplete evidence rather than averaged, for the same reason
    ``compare_symmetric`` refuses to average one direction with a missing one:
    the surviving subset is not a random sample of the coreset, it is the subset
    that happened to work.
    """
    if not tasks:
        return RetirementDecision(
            parent_id=parent_id,
            child_id=child_id,
            should_retire=False,
            reason="no tasks to compare on; no evidence of supersession",
        )

    scores: list[float] = []
    unavailable = 0
    for task in tasks:
        baseline = parent_traces.get(task.task_id)
        candidate = child_traces.get(task.task_id)
        if baseline is None or candidate is None:
            return RetirementDecision(
                parent_id=parent_id,
                child_id=child_id,
                should_retire=False,
                reason=(
                    f"incomplete evidence: no comparable trace pair for task "
                    f"{task.task_id!r}"
                ),
                judged=len(scores),
                unavailable=unavailable + 1,
            )
        try:
            verdict = compare(task, baseline, candidate)
        except Exception as exc:  # noqa: BLE001 - a judge fault must not abort the attempt
            return RetirementDecision(
                parent_id=parent_id,
                child_id=child_id,
                should_retire=False,
                reason=f"judge error on task {task.task_id!r}: {exc}",
                judged=len(scores),
                unavailable=unavailable + 1,
            )
        if not bool(getattr(verdict, "available", False)):
            return RetirementDecision(
                parent_id=parent_id,
                child_id=child_id,
                should_retire=False,
                reason=(
                    f"verdict unavailable for task {task.task_id!r}; a judge "
                    "outage is not evidence of supersession"
                ),
                judged=len(scores),
                unavailable=unavailable + 1,
            )
        scores.append(float(getattr(verdict, "score", 0.0)))

    mean = sum(scores) / len(scores)
    # Strict '> 0', matching the SV-4 promotion gate: a measured tie is not
    # evidence that the child superseded anything.
    should = mean > 0.0
    return RetirementDecision(
        parent_id=parent_id,
        child_id=child_id,
        should_retire=should,
        reason=(
            f"child preferred over parent (mean S={mean:.4f})"
            if should
            else f"child not preferred over parent (mean S={mean:.4f})"
        ),
        mean_preference=mean,
        judged=len(scores),
        unavailable=unavailable,
    )
