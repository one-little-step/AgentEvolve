"""Tests for two-level group/rollout concurrency.

Group-major admission matters because a group's diagnosis is blocked on ALL of
its rollouts: finishing whole groups early beats spreading thinly.

Concurrency assertions here use events/counters, never wall-clock ordering, so
the suite is deterministic on a loaded machine.
"""
from __future__ import annotations

import threading

import pytest

from agent_evolve.core.rho.scheduler import (
    ConcurrencyPlan,
    GroupResult,
    run_groups,
)


def test_invariant_rejects_a_global_cap_larger_than_the_structure() -> None:
    with pytest.raises(ValueError, match="global cap"):
        ConcurrencyPlan.validated(
            group_workers=2, rollout_workers=2, global_cap=5
        )


def test_invariant_accepts_a_cap_within_the_structure() -> None:
    plan = ConcurrencyPlan.validated(
        group_workers=4, rollout_workers=3, global_cap=6
    )

    assert plan.global_cap == 6


def test_rejects_non_positive_workers() -> None:
    with pytest.raises(ValueError):
        ConcurrencyPlan.validated(group_workers=0, rollout_workers=1, global_cap=1)


def test_rejects_boolean_workers() -> None:
    with pytest.raises(ValueError):
        ConcurrencyPlan.validated(group_workers=True, rollout_workers=1, global_cap=1)


def test_plan_is_frozen() -> None:
    plan = ConcurrencyPlan.validated(
        group_workers=1, rollout_workers=1, global_cap=1
    )

    with pytest.raises(Exception):
        plan.global_cap = 9  # type: ignore[misc]


def test_every_item_runs_exactly_once() -> None:
    groups = [(f"g{i}", [f"g{i}-r{j}" for j in range(3)]) for i in range(4)]
    seen: list[str] = []
    lock = threading.Lock()

    def run_one(group_id: str, item: str) -> str:
        with lock:
            seen.append(item)
        return item

    plan = ConcurrencyPlan.validated(
        group_workers=2, rollout_workers=3, global_cap=6
    )
    results = run_groups(groups, run_one, plan)

    assert len(results) == 4
    assert sorted(seen) == sorted(i for _, items in groups for i in items)
    assert all(len(r.outcomes) == 3 for r in results)


def test_global_cap_is_never_exceeded() -> None:
    """15 items, cap 4: the semaphore must hold the peak at or under the cap.

    Every item waits until the observed peak reaches the cap (or a bounded
    number of yields elapse), which forces maximum contention without relying
    on a sleep long enough to be "probably enough".
    """
    groups = [(f"g{i}", [f"g{i}-r{j}" for j in range(3)]) for i in range(5)]
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def run_one(group_id: str, item: str) -> str:
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        for _ in range(200):
            with lock:
                if state["peak"] >= 4:
                    break
            threading.Event().wait(0.001)
        with lock:
            state["now"] -= 1
        return item

    plan = ConcurrencyPlan.validated(
        group_workers=3, rollout_workers=3, global_cap=4
    )
    run_groups(groups, run_one, plan)

    assert state["peak"] <= 4
    assert state["now"] == 0


def test_rollout_workers_bound_concurrency_inside_one_group() -> None:
    groups = [("g0", [f"r{j}" for j in range(6)])]
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def run_one(group_id: str, item: str) -> str:
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        for _ in range(200):
            with lock:
                if state["peak"] >= 2:
                    break
            threading.Event().wait(0.001)
        with lock:
            state["now"] -= 1
        return item

    plan = ConcurrencyPlan.validated(
        group_workers=1, rollout_workers=2, global_cap=2
    )
    run_groups(groups, run_one, plan)

    assert state["peak"] <= 2


def test_group_completion_callback_fires_before_all_groups_finish() -> None:
    """g0's callback must run while later groups are still in flight.

    Later groups block on an event that only g0's completion callback sets, so
    the ordering assertion is causal rather than timing-based.
    """
    groups = [(f"g{i}", [f"g{i}-r{j}" for j in range(2)]) for i in range(4)]
    completed: list[str] = []
    lock = threading.Lock()
    g0_done = threading.Event()

    def run_one(group_id: str, item: str) -> str:
        if group_id != "g0":
            assert g0_done.wait(timeout=5.0), "g0 never completed"
        return item

    def on_complete(result: GroupResult) -> None:
        with lock:
            completed.append(result.group_id)
        if result.group_id == "g0":
            g0_done.set()

    plan = ConcurrencyPlan.validated(
        group_workers=2, rollout_workers=2, global_cap=4
    )
    run_groups(groups, run_one, plan, on_group_complete=on_complete)

    assert len(completed) == 4
    assert completed[0] == "g0"


def test_a_failing_item_does_not_lose_its_siblings() -> None:
    groups = [("g0", ["ok-1", "boom", "ok-2"])]

    def run_one(group_id: str, item: str) -> str:
        if item == "boom":
            raise RuntimeError("rollout failed")
        return item

    plan = ConcurrencyPlan.validated(
        group_workers=1, rollout_workers=3, global_cap=3
    )
    results = run_groups(groups, run_one, plan)

    assert results[0].failures == 1
    assert len(results[0].outcomes) == 2
    assert results[0].outcomes == ("ok-1", "ok-2")


def test_a_fully_failed_group_is_reported_with_zero_outcomes() -> None:
    groups = [("g0", ["a", "b"])]

    def run_one(group_id: str, item: str) -> str:
        raise RuntimeError("all dead")

    plan = ConcurrencyPlan.validated(
        group_workers=1, rollout_workers=2, global_cap=2
    )
    results = run_groups(groups, run_one, plan)

    assert results[0].outcomes == ()
    assert results[0].failures == 2


def test_serial_plan_still_works() -> None:
    groups = [("g0", ["a", "b"]), ("g1", ["c"])]

    def run_one(group_id: str, item: str) -> str:
        return item.upper()

    plan = ConcurrencyPlan.validated(
        group_workers=1, rollout_workers=1, global_cap=1
    )
    results = run_groups(groups, run_one, plan)

    assert results[0].outcomes == ("A", "B")
    assert results[1].outcomes == ("C",)


def test_results_are_in_input_group_order() -> None:
    """Completion order is scrambled by a reverse-release chain; output is not.

    Group ``gi`` waits for group ``g(i+1)`` to finish, so groups complete in
    strictly reverse order while results must still come back g0..g4.
    """
    groups = [(f"g{i}", [f"i{i}"]) for i in range(5)]
    done = {f"g{i}": threading.Event() for i in range(5)}

    def run_one(group_id: str, item: str) -> str:
        index = int(group_id[1:])
        if index < 4:
            assert done[f"g{index + 1}"].wait(timeout=5.0)
        done[group_id].set()
        return item

    plan = ConcurrencyPlan.validated(
        group_workers=5, rollout_workers=1, global_cap=5
    )
    results = run_groups(groups, run_one, plan)

    assert [r.group_id for r in results] == ["g0", "g1", "g2", "g3", "g4"]


def test_empty_group_list_returns_empty() -> None:
    plan = ConcurrencyPlan.validated(
        group_workers=2, rollout_workers=2, global_cap=2
    )

    assert run_groups([], lambda gid, item: item, plan) == ()


def test_a_group_with_no_items_is_still_reported() -> None:
    groups = [("g0", []), ("g1", ["a"])]

    def run_one(group_id: str, item: str) -> str:
        return item

    plan = ConcurrencyPlan.validated(
        group_workers=2, rollout_workers=2, global_cap=2
    )
    results = run_groups(groups, run_one, plan)

    assert results[0] == GroupResult(group_id="g0", outcomes=(), failures=0)
    assert results[1].outcomes == ("a",)


def test_scheduler_does_not_import_agent_specific_modules() -> None:
    import agent_evolve.core.rho.scheduler as module

    source = module.__file__
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "import cuga" not in text
    assert "import litellm" not in text
    assert "agent_evolve.adapters" not in text
