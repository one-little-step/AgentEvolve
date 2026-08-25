"""?03: production correlation_scope wiring (the missing emit half).

Governing contracts:
* ``core/correlation.py`` -- ambient ``contextvars`` scopes render ``X-AE-*``
  headers; absent facts are omitted, never blanked; scopes nest and restore
  on exit. Until now NOTHING in src/ ever opened one, so every capture was
  uncorrelated.
* Wired phases here:
  - ``phase="diagnose"`` -- every fault analysis, BOTH the legacy sequential
    path and the parallel fan-out (labels travel INTO the worker threads,
    because pool threads do not inherit the submitter's context);
  - ``phase="positivity"`` -- every success analysis (Judge 2).
* ``candidate`` is the version string the rollout ran against; ``rollout`` is
  the item's index within its group; ``run`` is deliberately unset until a
  run-level identifier exists (omitted header, never empty-string).
* After any public entry point returns, NO scope may remain active -- a
  leaked label would misattribute later calls (worse than none).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.analyzer import FakeAnalyzerJudge  # noqa: E402
from agent_evolve.core.blame import CausalFinding  # noqa: E402
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.config import resolve_profile  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    EvolutionCandidate,
    EvolutionTask,
)
from agent_evolve.core.correlation import current_correlation  # noqa: E402
from agent_evolve.core.fake_editor import FakeEditor  # noqa: E402
from agent_evolve.core.orchestrator import SequentialGepaRunner  # noqa: E402
from agent_evolve.core.pool import PersistentPool  # noqa: E402
from examples.fake_adapter import FakeAdapter  # noqa: E402

_TOKEN = "graphrag-retrieval"
_CLUSTER = "mechanism-default"

CAPTURED: list[Any] = []


def _task(task_id: str = "task-a", expected: str = _TOKEN) -> EvolutionTask:
    return EvolutionTask(
        task_id=task_id,
        input_text=f"produce {task_id}",
        expected_contract={"expected_substring": expected},
    )


def _runner() -> SequentialGepaRunner:
    """Failing base: standard diagnose flow."""
    adapter = FakeAdapter()
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
        analyzer_judge=_CapturingAnalyzer(),
        editor=FakeEditor(),
        embedder=LexicalEmbedder(dim=32),
        config=resolve_profile("research_sequential", seed=0),
        mechanism_cluster_id=_CLUSTER,
        seed=0,
    )


class _CapturingAnalyzer(FakeAnalyzerJudge):
    """Records the ambient correlation seen at analysis time."""

    def analyze(self, task, trace):  # type: ignore[no-untyped-def]
        CAPTURED.append(("legacy", task.task_id, current_correlation()))
        return super().analyze(task, trace)


class _PassingWorkspaceRunnerHarness:
    """validate() over a materialized child so the positivity gate opens."""


# ---------------------------------------------------------------------- #
# Diagnose phase: legacy sequential path
# ---------------------------------------------------------------------- #
def test_diagnose_labels_the_legacy_path() -> None:
    CAPTURED.clear()
    runner = _runner()

    issues = runner.build_issues([_task("task-a")])
    assert issues, "expected a diagnosed issue from the failing base"

    assert CAPTURED, "analyzer never observed a correlation"
    for _route, task_id, ctx in CAPTURED:
        assert ctx is not None, "diagnose ran outside any correlation scope"
        assert ctx.phase == "diagnose"
        assert ctx.task == task_id
        assert ctx.candidate.startswith("base-v0")
        assert isinstance(ctx.rollout, int)

    # No leakage: the label dies with the operation.
    assert current_correlation() is None


# ---------------------------------------------------------------------- #
# Diagnose phase: parallel fan-out (labels must cross the thread boundary)
# ---------------------------------------------------------------------- #
def test_diagnose_labels_survive_the_parallel_fanout() -> None:
    from agent_evolve import pipeline

    CAPTURED.clear()

    class _ParallelCapturer:
        analyzer_model_id = "parallel-capturer"

        def __init__(self) -> None:
            self.judge = FakeAnalyzerJudge()

        def analyze(self, report):  # type: ignore[no-untyped-def]
            CAPTURED.append(("parallel", report.task_id, current_correlation()))
            # Report protocol: findings, not analyses. One grounded observed
            # fault per trace so build_issues has something to rank.
            from agent_evolve.core.blame import BlameGraph, BlameNode

            return (
                CausalFinding(
                    verdict_id=f"v-{report.task_id}",
                    candidate_id=report.candidate_id,
                    task_id=report.task_id,
                    trace_id=report.trace_refs[0],
                    valence=1,
                    status="observed",
                    mechanism_description=(
                        "planner retrieval timed out; answered without evidence"
                    ),
                    mechanism_cluster_id=_CLUSTER,
                    severity=0.8,
                    confidence=0.7,
                    blame_graph=BlameGraph(
                        nodes=(
                            BlameNode(
                                actor_id="planner",
                                blame=1.0,
                                artifacts=("skills/retrieval",),
                            ),
                        )
                    ),
                    evidence_refs=("skills/retrieval",),
                    rationale="parallel wiring capture",
                ),
            )

    stack = pipeline.build_offline_stack(
        task_count=2,
        analyzer_factory=_ParallelCapturer,
        analyzer_workers=2,
    )
    issues = stack.runner.build_issues(stack.tasks)
    assert issues, "expected diagnosed issues"

    assert CAPTURED and all(route == "parallel" for route, _, _ in CAPTURED)
    for _route, task_id, ctx in CAPTURED:
        assert ctx is not None, "worker thread saw no correlation"
        assert ctx.phase == "diagnose"
        assert ctx.task == task_id
    assert current_correlation() is None


def _fake_trace_for(report):  # type: ignore[no-untyped-def]
    from agent_evolve.core.contracts import ExecutionTrace

    return ExecutionTrace(
        trace_id=f"tr-{report.trace_refs[0]}",
        candidate_id=report.candidate_id,
        task_id=report.task_id,
        events=(),
        final_output="wrong",
        status="success",
    )


# ---------------------------------------------------------------------- #
# Positivity phase
# ---------------------------------------------------------------------- #
def test_positivity_labels_judge2_calls() -> None:
    from test_positivity_judge import (
        FakePositivityJudgeProbe,
        passing_runner,
    )

    CAPTURED.clear()
    runner2 = passing_runner(FakePositivityJudgeProbe(CAPTURED))
    ws = runner2.adapter.materialize_candidate("base-v0", "att-corr2")
    runner2.validate(ws, _task("task-a"))

    assert CAPTURED, "positivity judge never ran"
    for _task_id, ctx in CAPTURED:
        assert ctx is not None
        assert ctx.phase == "positivity"
        assert ctx.task == "task-a"
        assert ctx.candidate.startswith("base-v0+")
    assert current_correlation() is None


# ---------------------------------------------------------------------- #
# End-to-end header emission through a real wrapper path
# ---------------------------------------------------------------------- #
def test_wrapper_attaches_x_ae_headers_inside_a_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The REAL `_litellm_completion` wrapper must merge X-AE-* into extra_headers.

    Header attachment lives inside that wrapper, so this test must NOT inject
    completion_fn (that would bypass it). It patches litellm.completion instead
    and lets the genuine wrapper run.
    """
    import litellm

    from agent_evolve.adapters.cuga_positivity_judge import CugaPositivityJudge
    from agent_evolve.core.correlation import correlation_scope

    seen: dict = {}

    def fake_completion(**request: object) -> object:
        seen.update(request)  # type: ignore[arg-type]
        return {
            "choices": [{"message": {"content": json.dumps({"findings": []})}}]
        }

    monkeypatch.setattr(litellm, "completion", fake_completion)

    judge = CugaPositivityJudge(model="test-model")  # live wrapper, patched transport

    with correlation_scope(
        run=None, candidate="cand-X", task="task-a", rollout=2, phase="positivity"
    ):
        judge.analyze_success(_task(), _passing_trace())

    extra = seen.get("extra_headers") or {}
    assert extra.get("X-AE-Candidate") == "cand-X"
    assert extra.get("X-AE-Rollout") == "2"
    assert extra.get("X-AE-Phase") == "positivity"
    assert "X-AE-Run" not in extra  # absent fact omitted, never blanked


def _passing_trace():
    from agent_evolve.core.contracts import ExecutionTrace

    return ExecutionTrace(
        trace_id="tr-pass", candidate_id="v1", task_id="task-a",
        events=(), final_output=f"ok {_TOKEN}", status="success",
    )
