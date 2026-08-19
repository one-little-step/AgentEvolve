"""SV-9: a crashed rollout is not evidence, and must not be scored.

Governing contracts:
* ``docs/architecture/data-contracts.md`` -- an unscorable rollout is never a
  wrong answer, and must not reach a score denominator.
* ``core/evaluation.py:61-65`` -- ``ANSWERED_TRACE_STATUSES`` is a deliberate
  *whitelist*; an unrecognised status means "no answer", not "answered".

Why this file exists
--------------------
Two independent paths admit rollouts, and only one of them filtered crashes.

The GEPA runner path is correct already: ``rollout_group`` marks a rollout
``scorable=False`` when its status is outside the whitelist, and
``_record_rollout_score`` *raises* rather than silently skipping one. Those
properties are pinned here so a future change cannot quietly drop them.

The RHO path is not. ``_rollout_grid`` (``core/rho/rounds.py:637-643``) drops a
rollout only when ``trace is None``. A CUGA crash returns a **real trace object**
with ``status="error"``, so it survives into ``usable``, gets scored by
``_record_scores``, and enters the entropy cell -- and it is also handed to the
diagnoser and the preference judge as though it were an observation.

This matters because of what the corpus looks like: in ``data/cachefish_traces``
all six 13-event rollouts were ``status=error`` with no answer, and they were the
*shortest* trajectories present. Any "fewer steps is better" efficiency signal
ranks those six crashes above all 23 successful rollouts.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.analyzer import FakeAnalyzerJudge  # noqa: E402
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.config import resolve_profile  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    EvolutionCandidate,
    EvolutionTask,
    ExecutionTrace,
)
from agent_evolve.core.evaluation import (  # noqa: E402
    ANSWERED_TRACE_STATUSES,
    ContractScorer,
)
from agent_evolve.core.fake_editor import FakeEditor  # noqa: E402
from agent_evolve.core.orchestrator import SequentialGepaRunner  # noqa: E402
from agent_evolve.core.pool import PersistentPool  # noqa: E402
from examples.fake_adapter import FakeAdapter  # noqa: E402

_TOKEN = "graphrag-retrieval"
_CLUSTER = "mechanism-default"

#: Statuses a crashed CUGA rollout actually reports. Each carries a real trace
#: object, which is exactly why a ``trace is None`` check does not catch them.
CRASH_STATUSES = ("error", "failed", "timeout", "cancelled", "")


def _task(task_id: str = "task-a") -> EvolutionTask:
    return EvolutionTask(
        task_id=task_id,
        input_text=f"produce {task_id}",
        expected_contract={"expected_substring": _TOKEN},
    )


def _trace(status: str, *, output: str = "", task_id: str = "task-a") -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=f"tr-{status or 'blank'}",
        candidate_id="base",
        task_id=task_id,
        events=(),
        final_output=output,
        status=status or "error",
    )


# ---------------------------------------------------------------------- #
# The whitelist itself
# ---------------------------------------------------------------------- #
def test_answered_statuses_are_a_closed_whitelist() -> None:
    """An unrecognised status must mean 'no answer', never 'answered'."""
    assert ANSWERED_TRACE_STATUSES == frozenset({"success", "ok", "completed"})


@pytest.mark.parametrize("status", CRASH_STATUSES)
def test_a_crashed_trace_is_unscorable_even_with_a_perfect_answer(status: str) -> None:
    """Status gates scorability before the answer is ever inspected.

    The answer text is deliberately correct here: a crashed rollout that happens
    to have the right substring in its buffer still produced no measurement.
    """
    scorer = ContractScorer()

    result = scorer.score_rollout(_task(), _trace(status, output=_TOKEN))

    assert result.scorable is False
    assert result.score == 0.0
    assert result.passed is not True
    assert result.reason


def test_a_successful_trace_with_the_answer_is_scorable() -> None:
    """The negative controls above mean nothing without this positive control."""
    scorer = ContractScorer()

    result = scorer.score_rollout(_task(), _trace("success", output=_TOKEN))

    assert result.scorable is True
    assert result.passed is True


# ---------------------------------------------------------------------- #
# The GEPA runner path: already correct, pinned here
# ---------------------------------------------------------------------- #
def _runner_with_crashing_rollouts(status: str) -> SequentialGepaRunner:
    class _CrashingAdapter(FakeAdapter):
        def capture_trace(self, rollout_result: object) -> ExecutionTrace:  # type: ignore[override]
            trace = super().capture_trace(rollout_result)  # type: ignore[misc]
            return dataclasses.replace(trace, status=status, final_output="")

    adapter = _CrashingAdapter()
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(
        EvolutionCandidate(
            candidate_id="base",
            version="base-v0",
            artifact_hashes={
                d.artifact_id: d.version_hash
                for d in adapter.artifact_inventory("base-v0")
            },
        )
    )
    return SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        analyzer_judge=FakeAnalyzerJudge(),
        editor=FakeEditor(),
        embedder=LexicalEmbedder(dim=32),
        config=resolve_profile("research_sequential", seed=0),
        mechanism_cluster_id=_CLUSTER,
        seed=0,
    )


def test_runner_marks_a_crashed_rollout_unscorable() -> None:
    runner = _runner_with_crashing_rollouts("error")

    observed = runner.rollout_group("base-v0", (_task(),), prefix="probe")

    assert len(observed) == 1
    assert observed[0].scorable is False


def test_runner_refuses_to_record_a_crashed_rollout_in_the_pool() -> None:
    """A silent skip would make a future miswiring invisible; it must raise."""
    runner = _runner_with_crashing_rollouts("error")
    observed = runner.rollout_group("base-v0", (_task(),), prefix="probe")

    with pytest.raises(ValueError, match="unscorable"):
        runner._record_rollout_score("base", observed[0])


def test_runner_leaves_the_score_tensor_empty_when_every_rollout_crashes() -> None:
    """The end-to-end property: crashes create no evidence cells."""
    runner = _runner_with_crashing_rollouts("error")

    runner.build_issues([_task()])

    base = runner.pool.base
    assert all(cell.rollout_count == 0 for cell in base.score_tensor.values())


# ---------------------------------------------------------------------- #
# The RHO path: crashes carry a real trace object, so `trace is None` misses them
# ---------------------------------------------------------------------- #
def test_rollout_grid_excludes_a_crashed_trace_from_usable_traces() -> None:
    """``_rollout_grid`` must filter on status, not only on ``trace is None``.

    This is the SV-9 defect proper. A CUGA crash yields
    ``RolloutOutcome(trace=<real trace, status='error'>)``, which passes a
    ``trace is None`` check and is then scored, diagnosed, and judged as if it
    were evidence.
    """
    from agent_evolve.core.evaluation import RolloutOutcome  # noqa: PLC0415
    from agent_evolve.core.rho.rounds import RhoHooks, _rollout_grid  # noqa: PLC0415
    from agent_evolve.core.rho.scheduler import ConcurrencyPlan  # noqa: PLC0415

    tasks = (_task("t1"),)

    def rollout(version: str, task: EvolutionTask, index: int) -> RolloutOutcome:
        # Rollout 0 crashes with a real trace; rollout 1 succeeds.
        if index == 0:
            return RolloutOutcome(
                task=task,
                trace=_trace("error", task_id=task.task_id),
            )
        return RolloutOutcome(
            task=task,
            trace=_trace("success", output=_TOKEN, task_id=task.task_id),
        )

    hooks = RhoHooks(rollout=rollout)  # type: ignore[call-arg]
    groups, failures = _rollout_grid(
        hooks,
        "base-v0",
        tasks,
        2,
        ConcurrencyPlan.validated(group_workers=1, rollout_workers=1, global_cap=1),
    )

    usable = groups["t1"]
    assert [t.status for t in usable] == ["success"], (
        "a status='error' trace must not be counted as a usable rollout"
    )
    assert failures == 1, "the crash must be counted as a failure, not dropped silently"


def test_rho_does_not_score_a_crashed_trace_into_the_entropy_cell() -> None:
    """A crash must not contribute a score to the cross-candidate entropy cell.

    Scoring one would corrupt the variance the DPP diversity term reads, and
    ``mark_comparable`` could promote a candidate whose rollout count was met
    only by counting crashes.
    """
    from agent_evolve.core.entropy import EntropyTracker  # noqa: PLC0415
    from agent_evolve.core.rho.rounds import (  # noqa: PLC0415
        RhoHooks,
        _record_scores,
        rho_cluster_id,
    )

    tasks = (_task("t1"),)
    scored: list[str] = []

    def score(task: EvolutionTask, trace: ExecutionTrace) -> float:
        scored.append(trace.status)
        return 1.0 if trace.status == "success" else 0.0

    tracker = EntropyTracker()
    hooks = RhoHooks(score=score)  # type: ignore[call-arg]
    groups = {
        "t1": (
            _trace("error", task_id="t1"),
            _trace("success", output=_TOKEN, task_id="t1"),
        )
    }

    _record_scores(hooks, tracker, "base-v0", tasks, groups)

    assert "error" not in scored, "a crashed trace must never be handed to the scorer"
    # Asserted through the public read API: core/entropy.py is protected, so no
    # accessor is added for the test's convenience.
    cell = tracker._cells[  # noqa: SLF001
        next(k for k in tracker.all_cells() if k.task_id == "t1")
    ]
    assert cell.scores["base-v0"] == [1.0], (
        "only the successful rollout's score belongs in the entropy cell"
    )
