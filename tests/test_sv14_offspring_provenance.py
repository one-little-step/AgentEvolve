"""SV-14: an offspring must not join the pool carrying its parent's diagnosis.

Governing contracts:
* ``docs/SEVERE-OPEN-ISSUES.md`` SV-14 (opened 2026-08-21) -- two defects, one
  root, both silent:

  1. ``commit_to_pool`` stamps the PARENT's ``CausalAnalysis``
     (``analyzer_model_id``, ``blame_confidence``) onto every cell of the
     child -- a statement about a different candidate, and for regression
     cells a different task.
  2. ``validate`` pays ``rollout_group``'s documented diagnose phase for the
     child's failing probes and then discards every analysis; the offspring
     therefore files no entropy evidence at commit.

* Fix order under test (register §"Fix direction"): retain first, then correct
  provenance to per-task child analyses with EXPLICIT absence otherwise
  (``blame_confidence=None`` -- the entropy-report lesson that a blank is not
  a measurement), then file the retained child analyses through
  ``_record_entropy_evidence``.

Why the fake harness alone cannot show defect 2
-----------------------------------------------
The default offline harness passes every probe once edited, so the diagnose
gate (``orchestrator.py:1401``) never fires on the child and any retention
test would pass vacuously with a measured discard of 0. Every test here that
needs a child diagnosis forces one: either a scorer that fails a child probe
on a version condition (``trace.candidate_id`` starts with the materialized
``base-v0+`` prefix) or a hand-built analyzed rollout driven through the same
public ``commit_to_pool`` entry point production uses.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.analyzer import FakeAnalyzerJudge  # noqa: E402
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.config import resolve_profile  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    CandidateWorkspace,
    EvolutionCandidate,
    EvolutionTask,
    ExecutionTrace,
)
from agent_evolve.core.evaluation import ContractScorer, RolloutScore  # noqa: E402
from agent_evolve.core.editor import FocusedValidationReport, ValidationKind  # noqa: E402
from agent_evolve.core.evaluation import ObservedRollout  # noqa: E402
from agent_evolve.core.fake_editor import FakeEditor  # noqa: E402
from agent_evolve.core.orchestrator import SequentialGepaRunner  # noqa: E402
from agent_evolve.core.pool import PersistentPool  # noqa: E402
from examples.fake_adapter import FakeAdapter  # noqa: E402

_TOKEN = "graphrag-retrieval"
_CLUSTER = "mechanism-default"
_CHILD_PREFIX = "base-v0+"


def _task(task_id: str = "task-a", expected: str = _TOKEN) -> EvolutionTask:
    return EvolutionTask(
        task_id=task_id,
        input_text=f"produce {task_id}",
        expected_contract={"expected_substring": expected},
    )


class _ChildProbeFailureScorer(ContractScorer):
    """The real contract scorer, except one child probe is forced to fail.

    The condition is version-shaped, not task-shaped: only rollouts whose
    trace carries a materialized child version (``base-v0+<attempt>``) fail
    ``task-b``. The parent therefore fails it naturally (its artifacts contain
    no token), while the child fails it *with a real answered trace*, which is
    what makes the diagnose gate produce an analysis for the child.
    """

    def score_rollout(self, task: EvolutionTask, trace: ExecutionTrace) -> RolloutScore:
        result = super().score_rollout(task, trace)
        if (
            task.task_id == "task-b"
            and trace is not None
            and trace.candidate_id.startswith(_CHILD_PREFIX)
            and result.scorable
        ):
            return dataclasses.replace(result, score=0.0, passed=False)
        return result


def _runner(scorer: ContractScorer | None = None) -> SequentialGepaRunner:
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
        analyzer_judge=FakeAnalyzerJudge(),
        editor=FakeEditor(),
        embedder=LexicalEmbedder(dim=32),
        config=resolve_profile("research_sequential", seed=0),
        mechanism_cluster_id=_CLUSTER,
        seed=0,
        scorer=scorer,
    )


def _failing_child_rollout(
    runner: SequentialGepaRunner,
    *,
    attempt_id: str = "att-sv14",
    task_id: str = "task-b",
) -> tuple[ObservedRollout, CandidateWorkspace]:
    """A scorable, failing, *diagnosed* rollout for a materialized child."""
    workspace = runner.adapter.materialize_candidate("base-v0", attempt_id)
    task = _task(task_id, expected="quantum-entangle")
    trace = ExecutionTrace(
        trace_id=f"tr-{attempt_id}-{task_id}",
        candidate_id=workspace.version,
        task_id=task_id,
        events=(),
        final_output="answered but wrong",
        status="success",
    )
    score = RolloutScore(
        task_id=task_id,
        grader_name=runner.resolved_scorer.grader_name,
        score=0.0,
        scorable=True,
        passed=False,
    )
    analysis = FakeAnalyzerJudge().analyze(task, trace)
    rollout = ObservedRollout(task=task, trace=trace, score=score, analysis=analysis)
    assert workspace.version.startswith(_CHILD_PREFIX)
    return rollout, workspace


# ---------------------------------------------------------------------- #
# Defect 2: retention
# ---------------------------------------------------------------------- #
def test_validate_retains_the_childs_own_analyses() -> None:
    """A diagnosed child probe must survive ``validate``.

    Fate assertion, not call counting: the analyzer ran (the register shows it
    already does), so the question is whether its output is *retained*. With
    no retention there is nowhere to assert it from.
    """
    runner = _runner(scorer=_ChildProbeFailureScorer())

    report = runner.validate(
        runner.adapter.materialize_candidate("base-v0", "att-ret"),
        _task("task-a"),
        regression_tasks=(_task("task-b", expected="quantum-entangle"),),
    )

    # The probe set really did include a failing, diagnosed child rollout --
    # without this guard the assertion below would pass vacuously.
    failed = [r for r in report.all_results if not r.passed]
    assert failed, "forced child failure did not reach validation"

    analyses = runner._last_validation_analyses
    rollouts = runner._last_validation_rollouts
    retained = [ro for ro in rollouts if ro.analysis is not None]

    # The discard is gone: EVERY failed scorable probe keeps its diagnosis --
    # produced count equals retained count, keyed by task.
    assert set(analyses) == {r.task_id for r in failed}
    assert {ro.task.task_id for ro in retained} == {r.task_id for r in failed}
    assert all(
        ro.trace is not None and ro.trace.candidate_id.startswith(_CHILD_PREFIX)
        for ro in retained
    )

    # And it is a real diagnosis of the child's failure, not a placeholder.
    assert analyses["task-b"].analyzer_model_id == "fake-analyzer"
    assert analyses["task-b"].severity == 1.0


# ---------------------------------------------------------------------- #
# Defect 1: provenance
# ---------------------------------------------------------------------- #
def test_committed_offspring_cells_do_not_carry_the_parents_diagnosis() -> None:
    """End-to-end fate of defect 1.

    Parent fails the origin task and carries a real diagnosis
    (``fake-analyzer``, total blame 1.0). The child passes everything and is
    accepted. Its committed cells must record the *child's* diagnostic state:
    a passing probe has no diagnosis, so absence must be explicit -- not the
    parent's values copied across the row.
    """
    runner = _runner()

    outcome = runner.run_attempt([_task("task-a")])

    assert outcome.accepted, f"expected acceptance, got: {outcome.reason}"
    assert outcome.result_candidate_id is not None

    parent_cell = runner.pool.base.cell("task-a", _CLUSTER)
    parent_prov = parent_cell.provenance[-1]
    # Guard: the parent really had a diagnosis worth mis-copying.
    assert parent_prov.analyzer_model_id == "fake-analyzer"
    assert parent_prov.blame_confidence == 1.0

    child_entry = runner.pool.get(outcome.result_candidate_id)
    child_prov = child_entry.cell("task-a", _CLUSTER).provenance[-1]

    assert child_prov.analyzer_model_id == ""
    assert child_prov.blame_confidence is None


# ---------------------------------------------------------------------- #
# Defect 1 + entropy filing: commit-time behaviour per task
# ---------------------------------------------------------------------- #
def test_commit_writes_per_task_provenance_and_files_child_entropy() -> None:
    """Direct drive of the public commit entry point, production shape.

    One accepted-style report over two tasks; the child holds a diagnosis for
    exactly one of them. That cell takes the CHILD's analyzer identity and
    blame; the undiagnosed cell records explicit absence. The diagnosed
    rollout is filed into the mechanism-keyed entropy tracker under the CHILD.
    """
    from agent_evolve.core.editor import ValidationResult

    runner = _runner()
    rollout, workspace = _failing_child_rollout(runner)

    report = FocusedValidationReport(
        origin=(
            ValidationResult(
                kind=ValidationKind.ORIGIN,
                task_id="task-b",
                score=0.0,
                trace_id=rollout.trace.trace_id if rollout.trace else "missing",
                passed=False,
                mechanism_cluster_id=_CLUSTER,
            ),
        ),
        worked=(),
        regression=(
            ValidationResult(
                kind=ValidationKind.REGRESSION,
                task_id="task-a",
                score=1.0,
                trace_id="tr-child-task-a",
                passed=True,
                mechanism_cluster_id=_CLUSTER,
            ),
        ),
    )

    entry = runner.commit_to_pool(
        runner.pool.base,
        workspace,
        "att-sv14-commit",
        report,
        validation_rollouts=(rollout,),
    )

    diagnosed = entry.cell("task-b", _CLUSTER).provenance[-1]
    assert diagnosed.analyzer_model_id == "fake-analyzer"
    assert diagnosed.blame_confidence == 1.0

    undiagnosed = entry.cell("task-a", _CLUSTER).provenance[-1]
    assert undiagnosed.analyzer_model_id == ""
    assert undiagnosed.blame_confidence is None

    mechanism_key = runner._entropy_cluster_id(rollout)
    assert mechanism_key
    assert any(
        key.task_id == "task-b" and key.mechanism_cluster_id == mechanism_key
        for key in runner.entropy.all_cells()
    ), "the child's diagnosed probe was not filed into the entropy tracker"


def test_commit_without_a_child_diagnosis_records_absence_everywhere() -> None:
    """No child analysis anywhere: explicit absence, and no invented rows.

    A passing probe legitimately has no diagnosis. Committing it must leave
    provenance honestly empty (never the parent's, never a 0.0 that reads as
    a measured zero) and must not fabricate an entropy cell.
    """
    from agent_evolve.core.editor import ValidationResult

    runner = _runner()
    workspace = runner.adapter.materialize_candidate("base-v0", "att-sv14-absent")

    report = FocusedValidationReport(
        origin=(
            ValidationResult(
                kind=ValidationKind.ORIGIN,
                task_id="task-a",
                score=1.0,
                trace_id="tr-child-pass",
                passed=True,
                mechanism_cluster_id=_CLUSTER,
            ),
        ),
        worked=(),
        regression=(),
    )

    entry = runner.commit_to_pool(
        runner.pool.base,
        workspace,
        "att-sv14-absent",
        report,
    )

    prov = entry.cell("task-a", _CLUSTER).provenance[-1]
    assert prov.analyzer_model_id == ""
    assert prov.blame_confidence is None
    assert not any(key.task_id.startswith("task") for key in runner.entropy.all_cells())
