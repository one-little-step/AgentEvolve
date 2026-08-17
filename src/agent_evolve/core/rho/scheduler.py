"""Two-level group/rollout concurrency with a global cap.

RHO's rollout phase is inherently grouped: ``k`` tasks x ``G`` rollouts, where a
group's diagnosis cannot start until all ``G`` of its rollouts finish. A single
flat worker count cannot express this -- with a flat cap of 6 the scheduler
cannot distinguish "6 tasks x 1 rollout" from "2 tasks x 3 rollouts".

The model is taken from the previous Gaia RHO runner
(``reference/evolve_run.py:99-105``)::

    group_workers    concurrently admitted task groups
    rollout_workers  concurrent rollouts within one group
    global_cap       hard ceiling on simultaneous executions

and its invariant (``reference/evolve_run.py:194``)::

    global_cap <= group_workers * rollout_workers

A global cap larger than the structure can produce is a configuration error, not
something to silently clamp. :meth:`ConcurrencyPlan.validated` and
:func:`validate_concurrency` therefore raise ``ValueError``; neither ever
adjusts a value to make it fit.

Group-major admission: completing whole groups early is preferred over spreading
thinly across all ``k`` tasks, so ``on_group_complete`` can dispatch diagnosis
for a finished group while later groups are still executing.

This module is agent-neutral. It schedules opaque callables and knows nothing
about CUGA, rollouts, or traces. Callers are responsible for any process-level
isolation their work requires -- notably, running more than one rollout at a
time against a process-global agent workspace requires the caller to supply
process isolation, which is outside this module's contract.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

__all__ = [
    "ConcurrencyPlan",
    "GroupResult",
    "run_groups",
    "validate_concurrency",
]


def validate_concurrency(
    group_workers: int, rollout_workers: int, global_cap: int
) -> None:
    """Refuse an incoherent two-level configuration.

    Pure: no I/O, no state. Intended for CLI preflight (the
    ``--max-workers <= --rho-group-workers * --rho-rollout-workers`` check) so
    the CLI can fail fast without building a plan. Raises ``ValueError``; never
    clamps.
    """
    for name, value in (
        ("group_workers", group_workers),
        ("rollout_workers", rollout_workers),
        ("global_cap", global_cap),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    structural = group_workers * rollout_workers
    if global_cap > structural:
        raise ValueError(
            f"global cap {global_cap} exceeds what the structure can produce "
            f"({group_workers} groups x {rollout_workers} rollouts = "
            f"{structural}); lower the cap or raise the worker counts"
        )


@dataclass(frozen=True, slots=True)
class ConcurrencyPlan:
    """A validated two-level concurrency configuration."""

    group_workers: int
    rollout_workers: int
    global_cap: int

    @classmethod
    def validated(
        cls, group_workers: int, rollout_workers: int, global_cap: int
    ) -> "ConcurrencyPlan":
        """Build a plan, refusing an incoherent configuration."""
        validate_concurrency(group_workers, rollout_workers, global_cap)
        return cls(
            group_workers=group_workers,
            rollout_workers=rollout_workers,
            global_cap=global_cap,
        )


@dataclass(frozen=True, slots=True)
class GroupResult:
    """One group's successful outcomes plus how many items failed."""

    group_id: str
    outcomes: tuple = ()
    failures: int = 0


def run_groups(
    groups: Sequence[tuple[str, Sequence[T]]],
    run_one: Callable[[str, T], object],
    plan: ConcurrencyPlan,
    on_group_complete: Callable[[GroupResult], None] | None = None,
) -> tuple[GroupResult, ...]:
    """Run every group's items, group-major, under the global cap.

    Results are returned in input group order regardless of completion order, so
    a caller's reporting is deterministic. An item that raises is counted as a
    failure and its siblings are preserved: one broken rollout must not discard
    a group's remaining evidence.

    ``on_group_complete`` is invoked from the worker thread that finished the
    group, as soon as that group finishes -- not batched at the end. It must be
    thread-safe.
    """
    if not groups:
        return ()

    # The global cap is enforced by a semaphore shared across every group, so
    # the two-level structure can never oversubscribe the real resource.
    gate = threading.Semaphore(plan.global_cap)

    def run_group(group_id: str, items: Sequence[T]) -> GroupResult:
        def guarded(item: T) -> tuple[bool, object]:
            with gate:
                try:
                    return True, run_one(group_id, item)
                except Exception:  # noqa: BLE001 - a failure is data
                    return False, None

        if not items:
            results: list[tuple[bool, object]] = []
        elif plan.rollout_workers == 1 or len(items) == 1:
            results = [guarded(item) for item in items]
        else:
            workers = min(plan.rollout_workers, len(items))
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix=f"rho-{group_id}"
            ) as pool:
                # map preserves input order regardless of completion order.
                results = list(pool.map(guarded, items))

        outcomes: list[object] = []
        failures = 0
        for ok, value in results:
            if ok:
                outcomes.append(value)
            else:
                failures += 1

        result = GroupResult(
            group_id=group_id, outcomes=tuple(outcomes), failures=failures
        )
        if on_group_complete is not None:
            on_group_complete(result)
        return result

    if plan.group_workers == 1 or len(groups) == 1:
        return tuple(run_group(gid, items) for gid, items in groups)

    workers = min(plan.group_workers, len(groups))
    collected: list[tuple[int, GroupResult]] = []
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="rho-group"
    ) as pool:
        futures = {
            pool.submit(run_group, gid, items): index
            for index, (gid, items) in enumerate(groups)
        }
        # as_completed so results are drained as groups finish; the input index
        # carried by the future restores deterministic ordering afterwards.
        # Indexing by position, not group_id, keeps duplicate ids harmless.
        for future in as_completed(futures):
            collected.append((futures[future], future.result()))
    collected.sort(key=lambda pair: pair[0])
    return tuple(result for _, result in collected)
