"""Behavioural tests for judge slot distinctness (SV-7).

SV-7's live symptom was the judge reporting ``Are events identical? True`` for a
baseline/candidate pair and scoring 0.0. Two readings are indistinguishable from
the outside: a genuine no-op edit, or a wiring bug feeding one trace into both
slots. The register is explicit that the judge *prompt* is not the suspect and
must not be "fixed" for this.

So these tests attack the two places the defect could actually live:

1. **The slot closures.** ``read_baseline``/``read_candidate`` must return
   different payloads when handed deliberately different traces. If they collapse,
   every preference score ever collected is void.
2. **The upstream grid.** ``_rollout_grid`` must stamp each version's traces with
   that version, so phase 9 hands two genuinely different objects to the judge.
   S1-1 (candidate rollouts stamped ``harness_version: base``) is the plausible
   mechanism, and it lives here rather than in the judge.

What these tests deliberately do NOT do: assert on prompt substrings. A test that
greps the rendered prompt for "baseline" would pass against a wiring that renders
the same trace under both labels, which is the exact bug.
"""
from __future__ import annotations

import json

import pytest

from agent_evolve.core.contracts import ExecutionTrace, TraceEvent


def _trace(trace_id: str, version: str, task_id: str, *, output: str, n_events: int = 2) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        candidate_id=version,
        task_id=task_id,
        events=tuple(
            TraceEvent(
                event_id=f"e{i}",
                kind="tool",
                actor_id="agent",
                parent_event_id=None,
                payload={"step": i, "version": version},
            )
            for i in range(n_events)
        ),
        final_output=output,
        status="ok",
    )


# ---------------------------------------------------------------------- #
# The slot closures themselves.
# ---------------------------------------------------------------------- #
def test_slot_closures_return_distinct_payloads_for_distinct_traces() -> None:
    """The core SV-7 assertion: two traces in, two different payloads out."""
    from agent_evolve.adapters.cuga_preference_judge import _build_callables

    base = _trace("t-base", "base-v0", "task-1", output="BASE ANSWER", n_events=2)
    cand = _trace("t-cand", "cand-v1", "task-1", output="CAND ANSWER", n_events=5)

    tools = _build_callables(
        task=_task("task-1"),
        baseline_slot=base,
        candidate_slot=cand,
        expected="",
        expected_kind="",
        baseline_summary="",
        candidate_summary="",
        plan={},
    )
    left = tools["read_baseline"]()
    right = tools["read_candidate"]()

    assert left != right, "slot closures collapsed to one trajectory (SV-7)"
    lp, rp = json.loads(left), json.loads(right)
    assert lp["final_output"] == "BASE ANSWER"
    assert rp["final_output"] == "CAND ANSWER"
    assert lp["events"] != rp["events"]


def test_slot_closures_do_not_leak_the_other_slots_trace() -> None:
    """Each closure must see only its own slot.

    A closure that rendered both would let the judge infer the labelling from
    content rather than from the slot, defeating the swap in
    ``compare_symmetric``.
    """
    from agent_evolve.adapters.cuga_preference_judge import _build_callables

    base = _trace("t-base", "base-v0", "task-1", output="ALPHA_ONLY", n_events=2)
    cand = _trace("t-cand", "cand-v1", "task-1", output="OMEGA_ONLY", n_events=3)
    tools = _build_callables(
        task=_task("task-1"),
        baseline_slot=base,
        candidate_slot=cand,
        expected="",
        expected_kind="",
        baseline_summary="",
        candidate_summary="",
        plan={},
    )
    assert "OMEGA_ONLY" not in tools["read_baseline"]()
    assert "ALPHA_ONLY" not in tools["read_candidate"]()


def test_identical_traces_are_reported_identically_not_hidden() -> None:
    """A genuine no-op edit must still render as identical.

    The fix for SV-7 is not "make the payloads differ"; it is to make a real
    difference reach the judge. When the two trajectories genuinely are the same,
    the payloads should say so -- that is honest evidence of a no-op edit, and
    masking it would replace one blind spot with another.
    """
    from agent_evolve.adapters.cuga_preference_judge import _build_callables

    same = _trace("t", "base-v0", "task-1", output="SAME", n_events=2)
    tools = _build_callables(
        task=_task("task-1"),
        baseline_slot=same,
        candidate_slot=same,
        expected="",
        expected_kind="",
        baseline_summary="",
        candidate_summary="",
        plan={},
    )
    assert tools["read_baseline"]() == tools["read_candidate"]()


def test_summaries_are_attached_to_their_own_slot() -> None:
    from agent_evolve.adapters.cuga_preference_judge import _build_callables

    base = _trace("t-base", "base-v0", "task-1", output="b", n_events=2)
    cand = _trace("t-cand", "cand-v1", "task-1", output="c", n_events=2)
    tools = _build_callables(
        task=_task("task-1"),
        baseline_slot=base,
        candidate_slot=cand,
        expected="",
        expected_kind="",
        baseline_summary="BASE_SUMMARY",
        candidate_summary="CAND_SUMMARY",
        plan={},
    )
    lp = json.loads(tools["read_baseline"]())
    rp = json.loads(tools["read_candidate"]())
    assert lp["harness_summary"] == "BASE_SUMMARY"
    assert rp["harness_summary"] == "CAND_SUMMARY"


# ---------------------------------------------------------------------- #
# Upstream: the grid must produce version-distinct traces.
# ---------------------------------------------------------------------- #
def test_rollout_grid_stamps_each_version_onto_its_own_traces() -> None:
    """Phase 9 must receive two genuinely different trace objects.

    This is where SV-7's plausible mechanism (S1-1) lives: if the grid returned
    traces stamped with the base version for a candidate rollout, the judge would
    be comparing the incumbent against itself while believing otherwise.
    """
    from agent_evolve.core.evaluation import RolloutOutcome
    from agent_evolve.core.rho.rounds import RhoHooks, _rollout_grid

    tasks = [_task("task-1")]

    def rollout(version: str, task, index: int):
        # The hook returns a RolloutOutcome-shaped object, not a bare trace:
        # _rollout_grid reads `.trace` off it.
        return RolloutOutcome(
            task=task,
            trace=_trace(
                f"t-{version}-{index}",
                version,
                task.task_id,
                output=f"answer from {version}",
            ),
        )

    hooks = RhoHooks(rollout=rollout)  # type: ignore[call-arg]
    base_groups, base_failures = _rollout_grid(hooks, "base-v0", tasks, 2, _plan())
    cand_groups, cand_failures = _rollout_grid(hooks, "cand-v1", tasks, 2, _plan())

    assert base_failures == 0 and cand_failures == 0
    base_traces = base_groups["task-1"]
    cand_traces = cand_groups["task-1"]
    assert base_traces and cand_traces
    assert {t.candidate_id for t in base_traces} == {"base-v0"}
    assert {t.candidate_id for t in cand_traces} == {"cand-v1"}
    # The pair phase 9 actually hands to the judge.
    assert base_traces[0].final_output != cand_traces[0].final_output


def _task(task_id: str):
    from agent_evolve.core.contracts import EvolutionTask

    return EvolutionTask(task_id=task_id, input_text=f"prompt for {task_id}")


def _plan():
    from agent_evolve.core.rho.scheduler import ConcurrencyPlan

    return ConcurrencyPlan.validated(1, 1, 1)
