"""J2B: the positivity judge protocol and the opened phase-3 gate (core side).

Governing contracts:
* ``docs/design/issue-lifecycle.md`` D5/D5.3/D5.6 -- Judge 2 analyzes
  successful rollouts and emits STRENGTHS (``CausalFinding`` with
  ``valence=-1``); each judge sees exactly one polarity, enforced structurally;
  strengths never enter ``EntropyTracker`` or score provenance (Q7).
* Output type is findings-on-the-rollout (new ``ObservedRollout.strengths``),
  NOT ``CausalAnalysis`` -- the flat fault record has no polarity, and the
  consumers of analyses are the consumers strengths must avoid.
* Default OFF: ``positivity_judge=None`` means byte-identical behavior and
  zero added cost. Turning it on is opt-in spend (+1 call per passing probe).
* TS2 interplay: the cross-attempt store keeps whole rollouts, so stored
  successes automatically carry strengths for the future signed index (IDX2);
  this file pins that too.
* Symmetric wall: a positivity judge returning any non-(-1) finding has its
  WHOLE batch refused and recorded -- refuse, never flip (house rule).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.analyzer import (  # noqa: E402
    FakeAnalyzerJudge,
    FakePositivityJudge,
)
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalFinding  # noqa: E402
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.config import resolve_profile  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    EvolutionCandidate,
    EvolutionTask,
)
from agent_evolve.core.fake_editor import FakeEditor  # noqa: E402
from agent_evolve.core.orchestrator import SequentialGepaRunner  # noqa: E402
from agent_evolve.core.pool import PersistentPool  # noqa: E402
from examples.fake_adapter import FakeAdapter  # noqa: E402

_TOKEN = "graphrag-retrieval"
_CLUSTER = "mechanism-default"

def _task(task_id: str = "task-a", expected: str = _TOKEN) -> EvolutionTask:
    return EvolutionTask(
        task_id=task_id,
        input_text=f"produce {task_id}",
        expected_contract={"expected_substring": expected},
    )


def _runner(positivity_judge: Any = None) -> SequentialGepaRunner:
    """A stack whose base ALREADY passes task-a, so rollouts succeed.

    ``positivity_judge`` is typed ``Any`` here because the frozen fake
    dataclass does not satisfy the Protocol's writable-attribute variance at
    type-check time; the runner accepts it structurally at runtime.
    """
    adapter = FakeAdapter(
        base_artifacts=(
            ("skills/retrieval", "skill", f"retrieve: answer with {_TOKEN}"),
            ("policies/execution", "policy", "execute(tool, args): return output"),
            ("prompts/system", "prompt", f"You are helpful. Always say {_TOKEN}."),
        )
    )
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
        positivity_judge=positivity_judge,
        config=resolve_profile("research_sequential", seed=0),
        mechanism_cluster_id=_CLUSTER,
        seed=0,
    )


# ---------------------------------------------------------------------- #
# Protocol + fake: the only legal output is strengths
# ---------------------------------------------------------------------- #
def test_fake_positivity_judge_emits_strength_findings() -> None:
    judge = FakePositivityJudge()

    findings = judge.analyze_success(_task(), _passing_trace())

    assert findings, "the fake must produce at least one strength"
    assert all(f.valence == -1 for f in findings)
    assert all(f.status == "observed" for f in findings)


def _passing_trace():
    from agent_evolve.core.contracts import ExecutionTrace

    return ExecutionTrace(
        trace_id="tr-pass",
        candidate_id="base-v0",
        task_id="task-a",
        events=(),
        final_output=f"answer containing {_TOKEN}",
        status="success",
    )


# ---------------------------------------------------------------------- #
# Default OFF: no judge, no calls, no strengths, unchanged behavior
# ---------------------------------------------------------------------- #
def test_default_off_passing_rollouts_carry_no_strengths_and_pay_nothing() -> None:
    runner = _runner(positivity_judge=None)

    observed = runner.rollout_group("base-v0", (_task(),), prefix="p")

    assert len(observed) == 1
    assert observed[0].score is not None and observed[0].score.passed
    assert observed[0].strengths == ()
    assert runner._positivity_calls == 0


# ---------------------------------------------------------------------- #
# Gate open: successes analyzed, strengths stored, TS2 carries them
# ---------------------------------------------------------------------- #
def test_gate_open_analyzes_passing_rollouts_into_stored_strengths() -> None:
    runner = _runner(
        cast(Any, FakePositivityJudge())
    )

    # Drive through validate(): that is a production capture route into TS2.
    workspace = runner.adapter.materialize_candidate("base-v0", "att-pos")
    report = runner.validate(workspace, _task())

    assert runner._positivity_calls == 1
    assert report.origin_passed, "the passing base should pass task-a"

    # TS2: the store keeps whole rollouts, so indexed evidence rides along.
    stored = runner.traces_for(workspace.version, "task-a")
    assert stored
    assert all(r.score is not None and r.score.passed for r in stored)
    strengths = [f for r in stored for f in r.strengths]
    assert strengths
    assert all(f.valence == -1 for f in strengths)


def test_failing_rollouts_do_not_reach_the_positivity_judge() -> None:
    """The positivity judge is for SUCCESSES; failures belong to Judge 1."""
    adapter = FakeAdapter()  # default artifacts lack the token -> failures
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
    runner2 = SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        analyzer_judge=FakeAnalyzerJudge(),
        editor=FakeEditor(),
        embedder=LexicalEmbedder(dim=32),
        positivity_judge=FakePositivityJudge(),
        config=resolve_profile("research_sequential", seed=0),
        mechanism_cluster_id=_CLUSTER,
        seed=0,
    )

    observed = runner2.rollout_group("base-v0", (_task(),), prefix="p")

    assert observed[0].score is not None and not observed[0].score.passed
    assert observed[0].strengths == ()
    assert runner2._positivity_calls == 0


# ---------------------------------------------------------------------- #
# Shared helpers for cross-file tests (correlation wiring)
# ---------------------------------------------------------------------- #
class FakePositivityJudgeProbe:
    """Records the ambient correlation visible at judgment time.

    Composition, not inheritance: ``FakePositivityJudge`` is a frozen
    slots dataclass, so subclasses cannot add attributes.
    """

    analyzer_model_id = FakePositivityJudge.analyzer_model_id

    def __init__(self, sink: list) -> None:
        self.sink = sink
        self._inner = FakePositivityJudge()

    def analyze_success(self, task, trace):  # type: ignore[no-untyped-def]
        from agent_evolve.core.correlation import current_correlation

        self.sink.append((task.task_id, current_correlation()))
        return self._inner.analyze_success(task, trace)


def passing_runner(positivity_judge) -> SequentialGepaRunner:  # type: ignore[no-untyped-def]
    """The passing-base harness whose validate() opens the gate."""
    return _runner(positivity_judge=positivity_judge)


# ---------------------------------------------------------------------- #
# Symmetric wall: smuggled faults are refused, never flipped
# ---------------------------------------------------------------------- #
class _FaultSmuggler:
    """Claims to be the positivity judge but reports a FAULT."""

    analyzer_model_id = "rogue-negativity"

    def analyze_success(self, task, trace):  # type: ignore[no-untyped-def]
        return (
            CausalFinding(
                verdict_id=f"v-{trace.trace_id}",
                candidate_id=trace.candidate_id,
                task_id=task.task_id,
                trace_id=trace.trace_id,
                valence=1,
                status="observed",
                mechanism_description="actually this failed",
                mechanism_cluster_id=_CLUSTER,
                severity=0.9,
                confidence=0.8,
                blame_graph=BlameGraph(
                    nodes=(
                        BlameNode(actor_id="planner", blame=1.0, artifacts=("skills/retrieval",)),
                    )
                ),
                evidence_refs=("skills/retrieval",),
                rationale="smuggled",
            ),
        )


def test_smuggled_fault_from_positivity_judge_is_refused_wholesale() -> None:
    runner = _runner(_FaultSmuggler())

    observed = runner.rollout_group("base-v0", (_task(),), prefix="p")

    assert observed[0].strengths == (), "a smuggled fault must not enter"
    failures = list(runner._positivity_failures)
    assert failures and any("polarity" in f[1] for f in failures)


def test_partial_batch_with_one_wrong_polarity_is_refused_entirely() -> None:
    """All-or-nothing: one bad finding poisons the batch; nothing enters."""

    class _MostlyStrengths(FakePositivityJudge):
        def analyze_success(self, task, trace):  # type: ignore[no-untyped-def]
            good = super().analyze_success(task, trace)
            mixed = good + (
                CausalFinding(
                    verdict_id="v-mixed",
                    candidate_id=trace.candidate_id,
                    task_id=task.task_id,
                    trace_id=trace.trace_id,
                    valence=1,
                    status="uncertain",
                    rationale="mixed in",
                ),
            )
            return mixed

    runner = _runner(_MostlyStrengths())

    observed = runner.rollout_group("base-v0", (_task(),), prefix="p")

    assert observed[0].strengths == ()
