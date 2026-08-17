"""Composition-root tests: the wired pipeline, offline.

Every test here runs with no CUGA process, no model endpoint and no network.
The live stack is exercised only up to the point where it would spend a token:
its refusals (threaded real rollouts, unknown grader) are asserted, its
execution is not.

Governing constraints (measured, not assumed -- see
``src/agent_evolve/benchmarks/cuga_process_pool.py`` and
``src/agent_evolve/benchmarks/cuga_executor.py``):

* ``CUGA_FOLDER`` is process-global, so real parallel rollouts require process
  isolation. A threaded real run is refused, not warned about.
* A rollout that produced no answer is not a wrong answer. It must never reach a
  score denominator, because a broken harness scoring as 0 would fabricate a
  self-improvement delta.
* A pass rate is never reported without its denominator.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.analysis import RolloutGroupReport  # noqa: E402
from agent_evolve.core.blame import (  # noqa: E402
    BlameGraph,
    BlameNode,
    CausalFinding,
)
from agent_evolve.core.contracts import (  # noqa: E402
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)
from agent_evolve.core.evaluation import (  # noqa: E402
    BenchmarkScorer,
    ContractScorer,
    ObservedRollout,
    RolloutScore,
    ScoreTally,
    tally_scores,
)
from agent_evolve.core.pool import PersistentPool  # noqa: E402
from agent_evolve.core.run_logging import ALL_LOG_CHANNELS  # noqa: E402
from agent_evolve.benchmarks.base import (  # noqa: E402
    BenchmarkGrading,
    BenchmarkTask,
    GradingUnavailableError,
    TaskOutcome,
    UnknownGraderError,
)
from agent_evolve.benchmarks.cuga_executor import (  # noqa: E402
    VANILLA_HARNESS,
    HarnessVersion,
)
from agent_evolve.adapters.cuga_adapter import CugaAdapter  # noqa: E402
from agent_evolve.adapters.cuga_editor_state import EditStagingArea  # noqa: E402
from agent_evolve.cuga_wrapper import (  # noqa: E402
    CugaWrapper,
    InMemoryRuntime,
    RuntimeSettings,
)
from examples.fake_adapter import FakeAdapter  # noqa: E402

import agent_evolve.pipeline as pipeline  # noqa: E402
from scripts.run_evolution import main as run_evolution_main  # noqa: E402


_TOKEN = "graphrag-retrieval"


# --------------------------------------------------------------------------- #
# doubles
# --------------------------------------------------------------------------- #
class _RegexLikeBenchmark:
    """A benchmark double with the same shape and refusals as GaiaBenchmark.

    ``expected_regex`` grades live answers by substring; ``unavailable`` always
    refuses (standing in for ``recorded_llm_verdict``, which cannot grade a new
    answer). Neither grading value ever reaches a task-facing object.
    """

    name = "regex-like"

    def __init__(self, expected: dict[str, str]) -> None:
        self._expected = dict(expected)

    def graders(self) -> tuple[str, ...]:
        return ("expected_regex", "unavailable")

    def load_tasks(self) -> tuple[BenchmarkTask, ...]:
        return tuple(
            BenchmarkTask(task_id=task_id, question=f"produce {task_id}")
            for task_id in sorted(self._expected)
        )

    def grading_for(self, task_id: str) -> BenchmarkGrading | None:
        """Scorer-only material. Its ``repr`` is redacted by the base class."""
        if task_id not in self._expected:
            return None
        return BenchmarkGrading(
            task_id=task_id,
            grader_names=self.graders(),
            payload={"expected_regex": self._expected[task_id]},
        )

    def score_all(self, task_id: str, answer: str) -> dict[str, TaskOutcome]:
        outcomes: dict[str, TaskOutcome] = {}
        for grader in self.graders():
            try:
                outcomes[grader] = self.score(task_id, answer, grader=grader)
            except GradingUnavailableError:
                continue
        return outcomes

    def score(self, task_id: str, answer: str, *, grader: str) -> TaskOutcome:
        if grader not in self.graders():
            raise UnknownGraderError(f"unknown grader {grader!r}")
        if grader == "unavailable":
            raise GradingUnavailableError("this grader cannot score a new answer")
        if task_id not in self._expected:
            raise GradingUnavailableError(f"no material for {task_id!r}")
        passed = self._expected[task_id] in (answer or "")
        return TaskOutcome(
            task_id=task_id,
            score=1.0 if passed else 0.0,
            passed=passed,
            grader_name=grader,
        )


class _FailingRolloutAdapter(FakeAdapter):
    """A FakeAdapter whose rollouts never produce an answer.

    Models a broken harness: the trace exists but its status says the run
    failed, so nothing it "answered" may be scored.
    """

    def capture_trace(self, rollout_result: object) -> ExecutionTrace:
        trace = super().capture_trace(rollout_result)
        return ExecutionTrace(
            trace_id=trace.trace_id,
            candidate_id=trace.candidate_id,
            task_id=trace.task_id,
            events=trace.events,
            final_output="",
            status="error",
        )


def _trace(task_id: str = "task-a", *, output: str = "", status: str = "success") -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=f"tr-{task_id}",
        candidate_id="base",
        task_id=task_id,
        events=(
            TraceEvent(
                event_id="e0",
                kind="tool_call",
                actor_id="agent",
                parent_event_id=None,
                payload={"tool": "search"},
            ),
        ),
        final_output=output,
        status=status,
    )


def _task(task_id: str = "task-a", expected: str | None = None) -> EvolutionTask:
    contract = {} if expected is None else {"expected_substring": expected}
    return EvolutionTask(
        task_id=task_id, input_text=f"produce {task_id}", expected_contract=contract
    )


# --------------------------------------------------------------------------- #
# benchmark-driven scoring records its grader
# --------------------------------------------------------------------------- #
def test_benchmark_scorer_records_the_grader_name_on_every_result() -> None:
    scorer = BenchmarkScorer(
        benchmark=_RegexLikeBenchmark({"task-a": _TOKEN}), grader="expected_regex"
    )

    hit = scorer.score_rollout(_task(), _trace(output=f"answer {_TOKEN}"))
    miss = scorer.score_rollout(_task(), _trace(output="nope"))

    assert hit.grader_name == "expected_regex"
    assert miss.grader_name == "expected_regex"
    assert (hit.score, hit.scorable, hit.passed) == (1.0, True, True)
    assert (miss.score, miss.scorable, miss.passed) == (0.0, True, False)


def test_contract_scorer_keeps_the_expected_contract_behaviour_and_names_itself() -> None:
    scorer = ContractScorer()

    hit = scorer.score_rollout(_task(expected=_TOKEN), _trace(output=f"x {_TOKEN}"))
    miss = scorer.score_rollout(_task(expected=_TOKEN), _trace(output="x"))

    assert scorer.grader_name == "expected_contract"
    assert (hit.score, hit.scorable) == (1.0, True)
    assert (miss.score, miss.scorable) == (0.0, True)


def test_rollout_score_refuses_an_unnamed_grader() -> None:
    with pytest.raises(ValueError, match="grader_name"):
        RolloutScore(task_id="task-a", grader_name="", score=1.0, scorable=True)


def test_benchmark_scorer_rejects_an_unknown_grader_at_construction() -> None:
    """A typo'd grader must fail before the first billed rollout, not after."""
    with pytest.raises(UnknownGraderError):
        BenchmarkScorer(
            benchmark=_RegexLikeBenchmark({"task-a": _TOKEN}), grader="expectd_regex"
        )


# --------------------------------------------------------------------------- #
# a failed rollout never enters a denominator
# --------------------------------------------------------------------------- #
def test_a_failed_rollout_is_unscorable_rather_than_a_zero() -> None:
    scorer = BenchmarkScorer(
        benchmark=_RegexLikeBenchmark({"task-a": _TOKEN}), grader="expected_regex"
    )

    result = scorer.score_rollout(_task(), _trace(output="", status="error"))

    assert result.scorable is False
    assert result.score == 0.0
    assert result.passed is False
    assert "error" in result.reason


def test_an_ungradable_task_is_unscorable_rather_than_a_zero() -> None:
    scorer = BenchmarkScorer(
        benchmark=_RegexLikeBenchmark({"task-a": _TOKEN}), grader="unavailable"
    )

    result = scorer.score_rollout(_task(), _trace(output="anything"))

    assert result.scorable is False


def test_tally_excludes_unscorable_rollouts_from_the_denominator() -> None:
    scores = (
        RolloutScore(task_id="t1", grader_name="g", score=1.0, scorable=True, passed=True),
        RolloutScore(task_id="t2", grader_name="g", score=0.0, scorable=True),
        RolloutScore(task_id="t3", grader_name="g", score=0.0, scorable=False, reason="no answer"),
    )

    tally = tally_scores(scores, grader_name="g")

    assert isinstance(tally, ScoreTally)
    assert (tally.passed, tally.evaluated, tally.unscorable, tally.attempted) == (1, 2, 1, 3)
    assert tally.pass_rate == pytest.approx(0.5)
    assert "1/2" in tally.summary


def test_tally_reports_no_pass_rate_when_nothing_was_scored() -> None:
    tally = tally_scores(
        (RolloutScore(task_id="t1", grader_name="g", score=0.0, scorable=False, reason="x"),),
        grader_name="g",
    )

    assert tally.pass_rate is None
    assert "n/a" in tally.summary


def test_a_failed_rollout_is_never_recorded_in_the_pool() -> None:
    """The single most important property: no fabricated zero in the tensor."""
    stack = pipeline.build_offline_stack(
        adapter=_FailingRolloutAdapter(),
        benchmark=_RegexLikeBenchmark({"task-a": _TOKEN}),
        grader="expected_regex",
        tasks=(_task(),),
    )

    outcome = stack.runner.run_attempt(stack.tasks)

    base = stack.pool.base
    assert base.score_tensor == {}, "a failed rollout must not reach the score tensor"
    assert outcome.result_candidate_id is None
    assert len(stack.pool) == 1


def test_a_failed_rollout_produces_no_issue() -> None:
    stack = pipeline.build_offline_stack(
        adapter=_FailingRolloutAdapter(),
        benchmark=_RegexLikeBenchmark({"task-a": _TOKEN}),
        grader="expected_regex",
        tasks=(_task(),),
    )

    assert stack.runner.build_issues(stack.tasks) == ()


def test_an_unscorable_probe_never_becomes_a_passing_validation_result() -> None:
    """An edit cannot be accepted on evidence that does not exist."""
    adapter = _FailingRolloutAdapter()
    stack = pipeline.build_offline_stack(
        adapter=adapter,
        benchmark=_RegexLikeBenchmark({"task-a": _TOKEN}),
        grader="expected_regex",
        tasks=(_task(),),
    )
    workspace = adapter.materialize_candidate("base-v0", "probe-attempt")

    report = stack.runner.validate(workspace, _task())

    assert report.all_results == ()
    assert stack.runner.unscorable_probe_count == 1


# --------------------------------------------------------------------------- #
# the offline stack runs a full iteration with no network
# --------------------------------------------------------------------------- #
def test_offline_stack_runs_one_full_iteration_and_evolves_the_pool() -> None:
    stack = pipeline.build_offline_stack(task_count=2)

    summaries = stack.run_iterations(1)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.iteration == 1
    assert summary.attempts == 1
    assert summary.accepted + summary.rejected + summary.no_issue == 1
    assert summary.pool_size >= 1


def test_passing_regression_probes_do_not_subtract_from_net_gain() -> None:
    """The inverse of a characterization test, now that the defect is FIXED.

    ``FocusedValidationReport.weighted_net_gain`` previously weighted REGRESSION
    at ``-1.0 * score``, so a regression probe that **passed** (score 1.0)
    subtracted 1.0. A repair that fixed its origin task and broke nothing netted:

        1 origin pass (+1.0) + 2 passing regression probes (-2.0) = -1.0

    and was rejected. With >= 2 tasks no edit could ever be accepted, so every
    self-improvement delta was exactly zero for arithmetic reasons rather than
    agent quality.

    Fixed in ``core/editor.py``: a regression probe is charged only when it
    FAILED, and then in proportion to its shortfall (``1 - score``). Real
    producers set ``passed = score >= 0.5`` with score being the task score
    (``orchestrator.py:486,1735``), so a high-scoring probe is a task that still
    works and must not be penalized. Genuine regressions remain gated by
    ``regression_violated`` and protected floors.
    """
    from agent_evolve.core.editor import (
        FocusedValidationReport,
        ValidationKind,
        ValidationResult,
        decide_acceptance,
    )

    def probe(
        kind: ValidationKind, task_id: str, score: float = 1.0, passed: bool = True
    ) -> ValidationResult:
        return ValidationResult(
            kind=kind, task_id=task_id, score=score, trace_id="t", passed=passed
        )

    report = FocusedValidationReport(
        origin=(probe(ValidationKind.ORIGIN, "origin"),),
        worked=(),
        regression=(
            probe(ValidationKind.REGRESSION, "other-1"),
            probe(ValidationKind.REGRESSION, "other-2"),
        ),
    )

    assert report.regression_violated is False, "nothing actually regressed"
    assert report.weighted_net_gain() == pytest.approx(1.0)
    assert decide_acceptance(report).accepted is True

    # A genuine regression is still charged, in proportion to how far it fell.
    broken = FocusedValidationReport(
        origin=(probe(ValidationKind.ORIGIN, "origin"),),
        worked=(),
        regression=(
            probe(ValidationKind.REGRESSION, "other-1", score=0.0, passed=False),
        ),
    )
    assert broken.weighted_net_gain() == pytest.approx(0.0)
    assert broken.regression_violated is True
    assert decide_acceptance(broken).accepted is False


def test_a_repair_is_accepted_when_no_regression_probe_dilutes_it() -> None:
    """The loop does work end to end; the blocker above is arithmetic, not wiring.

    At one task there is no regression probe to subtract, so the same repair the
    two-task case rejects is accepted and committed. This is what isolates the
    defect to ``weighted_net_gain`` rather than to the pipeline.
    """
    stack = pipeline.build_offline_stack(task_count=1)

    summaries = stack.run_iterations(1)

    assert summaries[0].accepted == 1
    assert stack.pool_size() == 2


def test_offline_stack_accepts_a_repairing_edit_and_grows_the_pool() -> None:
    stack = pipeline.build_offline_stack(task_count=1)

    summaries = stack.run_iterations(1)

    assert summaries[0].accepted == 1
    assert stack.pool_size() == 2


def test_offline_stack_needs_no_cuga_model_configuration(monkeypatch) -> None:
    for name in ("CUGA_MODEL", "LITELLM_MODEL", "CUGA_API_KEY", "LITELLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    stack = pipeline.build_offline_stack(task_count=1)
    stack.run_iterations(1)

    assert stack.uses_real_agent is False


def test_offline_stack_measures_a_version_with_an_honest_denominator() -> None:
    stack = pipeline.build_offline_stack(task_count=2)

    tally = stack.measure("base-v0", prefix="baseline")

    assert tally.attempted == 2
    assert tally.evaluated == 2
    assert tally.pass_rate == pytest.approx(0.0)
    assert tally.grader_name == stack.grader_name


def test_offline_stack_header_names_every_choice_that_changes_the_number() -> None:
    stack = pipeline.build_offline_stack(task_count=2)

    header = "\n".join(stack.header_lines)

    for expected in ("grader", "analyzer workers", "knowledge store", "candidates", "rollout"):
        assert expected in header


def test_offline_stack_reports_one_candidate_because_no_rho_seeder_exists() -> None:
    stack = pipeline.build_offline_stack(task_count=1)

    assert stack.candidate_count() == 1


# --------------------------------------------------------------------------- #
# the base harness's editable surface
# --------------------------------------------------------------------------- #
def test_vanilla_harness_puts_instructions_in_the_editable_artifact_set() -> None:
    """``instructions`` is the strongest lever CUGA exposes and the editor can
    only reach it through this mapping.

    Before the vanilla harness owned an instructions string, this fell to the
    empty-skill fallback and the editor's entire write set was one empty skill.
    """
    artifacts = pipeline._harness_artifacts(VANILLA_HARNESS)

    assert "instructions" in artifacts
    assert artifacts["instructions"] == VANILLA_HARNESS.instructions


def test_a_harness_owning_nothing_still_gets_one_editable_slot() -> None:
    """The fallback stays: with an empty inventory every issue is dropped for
    lack of attribution and the loop can never act."""
    artifacts = pipeline._harness_artifacts(HarnessVersion(version="bare"))

    assert artifacts == {"skills/generated-evolved": ""}


def test_a_vanilla_based_candidate_registers_instructions_as_writable() -> None:
    """The write set the editor is handed comes from the adapter inventory, so
    an artifact that never registers is an artifact the editor cannot edit."""
    adapter = CugaAdapter(
        wrapper=CugaWrapper(InMemoryRuntime(), RuntimeSettings(model="test-model"))
    )
    adapter.register_candidate("base", pipeline._harness_artifacts(VANILLA_HARNESS))

    writable = {
        d.artifact_id for d in adapter.artifact_inventory("base") if d.writable
    }
    assert "instructions" in writable


def test_the_editor_may_stage_a_replacement_for_instructions() -> None:
    """Creation is capped behind ``creatable_prefix``, so instructions must be
    reachable by replacement instead -- otherwise the lever stays unusable."""
    area = EditStagingArea(
        write_set=tuple(sorted(pipeline._harness_artifacts(VANILLA_HARNESS))),
    )

    outcome = area.stage_replace("instructions", "revised instructions")

    assert outcome.accepted, outcome.reason
    assert "instructions" in area.staged_ids()


def test_an_edited_instructions_artifact_reaches_the_harness_config() -> None:
    """An accepted edit that never reaches CUGA is a reported success that
    changed nothing. Pins the whole path: stage, apply, materialize, config."""
    adapter = CugaAdapter(
        wrapper=CugaWrapper(InMemoryRuntime(), RuntimeSettings(model="test-model"))
    )
    artifacts = pipeline._harness_artifacts(VANILLA_HARNESS)
    adapter.register_candidate("base", artifacts)
    area = EditStagingArea(write_set=tuple(sorted(artifacts)))
    area.stage_replace("instructions", "revised instructions")

    workspace = adapter.materialize_candidate("base", "att-1")
    adapter.apply_structured_edits(workspace, area.edits())
    config = adapter._harness_config(
        workspace.version, EvolutionTask(task_id="t-1", input_text="a question")
    )

    assert config["instructions"] == "revised instructions"


# --------------------------------------------------------------------------- #
# the real analyzer protocol is accepted through the shim
# --------------------------------------------------------------------------- #
class _ReportAnalyzer:
    """A report-based analyzer, structurally identical to the CUGA one."""

    analyzer_model_id = "report-analyzer"

    def __init__(self) -> None:
        self.reports: list[RolloutGroupReport] = []

    def analyze(self, report: RolloutGroupReport) -> tuple[CausalFinding, ...]:
        self.reports.append(report)
        return tuple(
            CausalFinding(
                verdict_id=f"{report.task_id}:v",
                candidate_id=report.candidate_id,
                task_id=report.task_id,
                trace_id=trace_id,
                status="observed",
                mechanism_description="retriever returned no candidate passage",
                mechanism_cluster_id="c0",
                severity=0.8,
                confidence=0.7,
                blame_graph=BlameGraph(
                    nodes=(BlameNode(actor_id="agent", blame=1.0, artifacts=()),)
                ),
                evidence_refs=("e0",),
                rationale="grounded in the observed tool call",
            )
            for trace_id in report.trace_refs
        )


def test_a_report_based_analyzer_is_accepted_through_the_shim() -> None:
    analyzer = _ReportAnalyzer()
    stack = pipeline.build_offline_stack(task_count=1, analyzer=analyzer)

    issues = stack.runner.build_issues(stack.tasks)

    assert analyzer.reports, "the analyzer must have been called with a report"
    assert len(issues) == 1
    assert stack.runner.observed_mechanisms == (
        "retriever returned no candidate passage",
    )


def test_the_shim_never_lets_the_analyzer_see_the_expected_contract() -> None:
    analyzer = _ReportAnalyzer()
    stack = pipeline.build_offline_stack(
        task_count=1, analyzer=analyzer, task_token=_TOKEN
    )

    stack.runner.build_issues(stack.tasks)

    blob = repr(analyzer.reports)
    assert _TOKEN not in blob


def test_the_score_comes_from_the_scorer_not_from_the_analyzer() -> None:
    """A diagnosis is not a measurement; the scorer owns the number."""
    analyzer = _ReportAnalyzer()
    stack = pipeline.build_offline_stack(
        task_count=1,
        analyzer=analyzer,
        benchmark=_RegexLikeBenchmark({"task-1": _TOKEN}),
        grader="expected_regex",
        tasks=(_task("task-1"),),
    )

    observed = stack.runner.rollout_group("base-v0", stack.tasks, prefix="m")

    assert len(observed) == 1
    assert isinstance(observed[0], ObservedRollout)
    assert observed[0].score is not None
    assert observed[0].score.grader_name == "expected_regex"


# --------------------------------------------------------------------------- #
# max_analyzer_workers is honored
# --------------------------------------------------------------------------- #
class _BarrierAnalyzer:
    """Proves genuine concurrency: every call must meet at the barrier.

    A sequential runner can never satisfy a 3-party barrier, so the barrier
    timing out is a real failure signal rather than a flaky timing assertion.
    """

    analyzer_model_id = "barrier-analyzer"

    def __init__(self, parties: int) -> None:
        self.barrier = threading.Barrier(parties, timeout=10.0)

    def analyze(self, report: RolloutGroupReport) -> tuple[CausalFinding, ...]:
        self.barrier.wait()
        return tuple(
            CausalFinding(
                verdict_id=f"{report.task_id}:v",
                candidate_id=report.candidate_id,
                task_id=report.task_id,
                trace_id=trace_id,
                status="observed",
                mechanism_description=f"mechanism for {report.task_id}",
                mechanism_cluster_id="c0",
                severity=0.5,
                confidence=0.5,
                blame_graph=BlameGraph(
                    nodes=(BlameNode(actor_id="agent", blame=1.0, artifacts=()),)
                ),
                evidence_refs=("e0",),
                rationale="grounded",
            )
            for trace_id in report.trace_refs
        )


def test_max_analyzer_workers_actually_analyzes_in_parallel() -> None:
    analyzer = _BarrierAnalyzer(parties=3)
    stack = pipeline.build_offline_stack(
        task_count=3, analyzer_factory=lambda: analyzer, analyzer_workers=3
    )

    issues = stack.runner.build_issues(stack.tasks)

    assert stack.runner.analyzer_workers == 3
    assert len(issues) == 3


def test_analyzer_workers_default_to_one_and_stay_sequential() -> None:
    analyzer = _ReportAnalyzer()
    stack = pipeline.build_offline_stack(task_count=2, analyzer=analyzer)

    stack.runner.build_issues(stack.tasks)

    assert stack.runner.analyzer_workers == 1


def test_a_failing_analyzer_is_recorded_and_does_not_abort_the_batch() -> None:
    class _Boom:
        analyzer_model_id = "boom"

        def analyze(self, report: RolloutGroupReport) -> tuple[CausalFinding, ...]:
            raise RuntimeError("analyzer outage")

    stack = pipeline.build_offline_stack(
        task_count=2, analyzer_factory=_Boom, analyzer_workers=2
    )

    issues = stack.runner.build_issues(stack.tasks)

    assert issues == ()
    assert len(stack.runner.analysis_failures) == 2


# --------------------------------------------------------------------------- #
# real parallel rollouts refuse threads
# --------------------------------------------------------------------------- #
def test_real_parallel_rollouts_refuse_thread_isolation() -> None:
    from agent_evolve.benchmarks.cuga_executor import (
        ConcurrencyUnsupportedError,
        HarnessVersion,
    )

    with pytest.raises(ConcurrencyUnsupportedError):
        pipeline.CugaRolloutRunner(
            harness=HarnessVersion(version="vanilla"),
            benchmark=_RegexLikeBenchmark({"task-a": _TOKEN}),
            max_workers=2,
            worker_pool=None,
        )


def test_real_serial_rollouts_are_permitted_without_a_worker_pool() -> None:
    from agent_evolve.benchmarks.cuga_executor import HarnessVersion

    runner = pipeline.CugaRolloutRunner(
        harness=HarnessVersion(version="vanilla"),
        benchmark=_RegexLikeBenchmark({"task-a": _TOKEN}),
        max_workers=1,
        worker_pool=None,
    )

    assert runner.max_workers == 1
    assert runner.isolation == "thread"


def test_a_worker_pool_reports_process_isolation() -> None:
    from agent_evolve.benchmarks.cuga_executor import HarnessVersion

    class _Pool:
        knowledge_seed = None
        root = Path("data/cuga-workers")

        def lease(self, worker_id: str, harness_version: str) -> object:
            raise AssertionError("no rollout should run in this test")

        def run(self, lease: object, task_id: str, harness_config: object) -> object:
            raise AssertionError("no rollout should run in this test")

        def close(self) -> None:
            return None

    runner = pipeline.CugaRolloutRunner(
        harness=HarnessVersion(version="vanilla"),
        benchmark=_RegexLikeBenchmark({"task-a": _TOKEN}),
        max_workers=4,
        worker_pool=_Pool(),
    )

    assert runner.isolation == "process"


def test_the_offline_rollout_runner_is_serial_and_needs_no_isolation() -> None:
    stack = pipeline.build_offline_stack(task_count=2)

    assert stack.rollout_isolation == "in-process (fake adapter)"


# --------------------------------------------------------------------------- #
# knowledge-store parity is an explicit, printed choice
# --------------------------------------------------------------------------- #
def test_worker_knowledge_store_defaults_to_empty_and_is_stated() -> None:
    """The repo's .cuga/knowledge holds unrelated fixtures; seeding is opt-in."""
    assert pipeline.DEFAULT_WORKER_KNOWLEDGE_SEED is None

    described = pipeline.describe_knowledge_choice(None)

    assert "EMPTY" in described


def test_seeding_the_worker_knowledge_store_is_named_in_the_header() -> None:
    described = pipeline.describe_knowledge_choice(Path("/tmp/knowledge"))

    assert "/tmp/knowledge" in described


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_dry_run_cli_exits_zero_without_cuga_or_network(monkeypatch, capsys) -> None:
    for name in ("CUGA_MODEL", "LITELLM_MODEL", "CUGA_API_KEY", "LITELLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    code = run_evolution_main(["--dry-run", "--tasks", "3", "--iterations", "1"])

    out = capsys.readouterr().out
    assert code == 0
    assert "grader" in out
    assert "iteration 1" in out


def test_dry_run_cli_prints_the_noise_floor_next_to_the_delta(capsys) -> None:
    run_evolution_main(["--dry-run", "--tasks", "2", "--iterations", "1"])

    out = capsys.readouterr().out
    assert "16.67" in out
    assert "delta" in out


def test_cli_requires_a_dataset_for_a_live_run(capsys) -> None:
    code = run_evolution_main(["--tasks", "1", "--iterations", "1"])

    assert code == 2
    assert "--dataset" in capsys.readouterr().out


def test_cli_refuses_parallel_real_rollouts_without_process_isolation(capsys) -> None:
    code = run_evolution_main(
        [
            "--dataset",
            "datasets/gaia/gaia_l1_validation_tiny5__baseline__20260812_180239",
            "--grader",
            "expected_regex",
            "--harness",
            "vanilla",
            "--max-workers",
            "4",
            "--isolation",
            "thread",
            "--tasks",
            "2",
        ]
    )

    assert code == 2
    assert "isolation" in capsys.readouterr().out.lower()


def test_the_thread_refusal_does_not_depend_on_model_credentials(monkeypatch) -> None:
    """Regression: the refusal must fire before any CUGA wrapper is built.

    The first version of the composition root built ``CugaWrapper`` -- and so
    resolved ``RuntimeSettings.from_env()`` -- before checking isolation. With no
    model configured that reported "CUGA_MODEL is required", sending an operator
    to configure a model for a run that was going to be refused anyway, and
    hiding the actual defect. An unsafe worker count is unsafe with or without
    credentials.
    """
    from agent_evolve.benchmarks.cuga_executor import (
        ConcurrencyUnsupportedError,
        HarnessVersion,
    )

    for name in ("CUGA_MODEL", "LITELLM_MODEL", "CUGA_API_KEY", "LITELLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConcurrencyUnsupportedError):
        pipeline.require_safe_rollout_concurrency(
            HarnessVersion(version="vanilla"),
            max_workers=4,
            isolation="thread",
        )


def test_serial_and_process_isolated_concurrency_are_permitted() -> None:
    from agent_evolve.benchmarks.cuga_executor import HarnessVersion

    harness = HarnessVersion(version="vanilla")

    # Neither raises: one worker needs no isolation, and process isolation is
    # exactly the arrangement that makes many workers safe.
    pipeline.require_safe_rollout_concurrency(
        harness, max_workers=1, isolation="thread"
    )
    pipeline.require_safe_rollout_concurrency(
        harness, max_workers=8, isolation="process"
    )


def test_cli_dry_run_ignores_a_dataset_it_was_not_given(capsys) -> None:
    """--dry-run must never touch a dataset directory or a benchmark loader."""
    code = run_evolution_main(["--dry-run", "--tasks", "1", "--iterations", "1"])

    assert code == 0
    assert "fake" in capsys.readouterr().out.lower()


# --------------------------------------------------------------------------- #
# run-log capture threads through the composition root
# --------------------------------------------------------------------------- #
class _RecordingPool:
    """A CugaProcessPool double that reports the log_capture it was handed.

    The pool is the only component whose capture wiring already existed, so what
    matters here is that the composition root passes the operator's config down
    rather than constructing a fresh default.
    """

    root = Path("data/cuga-workers")
    knowledge_seed = None
    seen: dict = {}

    def __init__(self, **kwargs) -> None:
        type(self).seen = dict(kwargs)
        self.log_capture = kwargs.get("log_capture")

    def lease(self, worker_id: str, harness_version: str) -> object:
        raise AssertionError("no rollout should run in this test")

    def run(self, lease: object, task_id: str, harness_config: object) -> object:
        raise AssertionError("no rollout should run in this test")

    def close(self) -> None:
        return None


def _live_stack(monkeypatch, log_capture, **kwargs):
    """Build the live stack with the worker pool stubbed and no token spent."""
    from agent_evolve.benchmarks import cuga_process_pool
    from agent_evolve.benchmarks.cuga_executor import HarnessVersion

    for name, value in (
        ("CUGA_MODEL", "test/model"),
        ("CUGA_API_KEY", "test-key"),
        ("CUGA_BASE_URL", "http://localhost:1"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(cuga_process_pool, "CugaProcessPool", _RecordingPool)

    return pipeline.build_live_stack(
        benchmark=_RegexLikeBenchmark({"task-a": _TOKEN}),
        grader="expected_regex",
        harness=HarnessVersion(version="vanilla"),
        isolation="process",
        max_workers=2,
        log_capture=log_capture,
        **kwargs,
    )


def test_the_live_stack_records_the_operators_capture_config(monkeypatch, tmp_path):
    """The config a run reports must be the config the run was given.

    ``ResolvedConfig.manifest_payload`` is what a later reader consults to learn
    whether logs exist at all, so the composition root passing its own default
    would make a captured run look uncaptured.
    """
    from agent_evolve.core.run_logging import LogCaptureConfig

    capture = LogCaptureConfig(enabled=True, root=tmp_path / "logs")
    stack = _live_stack(monkeypatch, capture)
    try:
        assert stack.runner.config.log_capture is capture
        assert stack.runner.config.manifest_payload()["log_capture"]["enabled"] is True
    finally:
        stack.close()


def test_the_live_stack_hands_the_capture_config_to_the_worker_pool(
    monkeypatch, tmp_path
):
    """Worker stderr is the channel that was lost; the pool needs the config."""
    from agent_evolve.core.run_logging import LogCaptureConfig

    capture = LogCaptureConfig(enabled=True, root=tmp_path / "logs")
    stack = _live_stack(monkeypatch, capture)
    try:
        assert _RecordingPool.seen["log_capture"] is capture
    finally:
        stack.close()


def test_the_live_stack_gives_the_analyzer_and_editor_their_own_sinks(
    monkeypatch, tmp_path
):
    """A sink per channel, so narrowing --log-channels actually narrows writes."""
    from agent_evolve.core.run_logging import LogCaptureConfig

    capture = LogCaptureConfig(enabled=True, root=tmp_path / "logs")
    stack = _live_stack(monkeypatch, capture)
    try:
        assert stack.runner.analyzer_judge.log_sink.channel == "analyzer"
        assert stack.runner.editor.log_sink.channel == "editor"
        # The analyzer is rebuilt per worker thread, so the factory -- not just
        # the one instance the stack holds -- must carry the sink.
        assert stack.runner.analyzer_factory().log_sink.channel == "analyzer"
    finally:
        stack.close()


def test_capture_defaults_to_off_in_the_live_stack(monkeypatch, tmp_path):
    """Opt-in everywhere: an unconfigured live run writes nothing."""
    stack = _live_stack(monkeypatch, None)
    try:
        assert stack.runner.config.log_capture.enabled is False
        assert _RecordingPool.seen["log_capture"].enabled is False
        assert stack.runner.editor.log_sink.active is False
    finally:
        stack.close()
    assert list(tmp_path.iterdir()) == []


def test_closing_the_stack_closes_every_sink(monkeypatch, tmp_path):
    """A stream still held open at exit loses its final lines."""
    capture_root = tmp_path / "logs"
    from agent_evolve.core.run_logging import LogCaptureConfig

    stack = _live_stack(
        monkeypatch, LogCaptureConfig(enabled=True, root=capture_root)
    )
    sink = stack.runner.editor.log_sink
    stream = sink.open_stream("probe")
    assert stream is not None

    stack.close()

    assert stream.closed


def test_the_offline_stack_threads_capture_into_its_config(tmp_path):
    """The dry run rehearses capture too, or the flag is untested until live."""
    from agent_evolve.core.run_logging import LogCaptureConfig

    capture = LogCaptureConfig(enabled=True, root=tmp_path / "logs")
    stack = pipeline.build_offline_stack(task_count=1, log_capture=capture)
    try:
        assert stack.runner.config.log_capture is capture
    finally:
        stack.close()


def test_an_iteration_records_its_boundaries_and_tally_on_the_pipeline_channel(
    tmp_path,
):
    """Iteration boundaries and the measured tally, or a log cannot be aligned.

    Analyzer and editor records carry candidate/task names but no iteration, so
    without the pipeline channel there is no way to say which iteration a judge
    transcript belongs to, nor what the numbers were when it was written.
    """
    from agent_evolve.core.run_logging import LogCaptureConfig

    stack = pipeline.build_offline_stack(
        task_count=1,
        log_capture=LogCaptureConfig(enabled=True, root=tmp_path / "logs"),
    )
    try:
        stack.measure(stack.base_version, prefix="before")
        stack.run_iterations(1)
    finally:
        stack.close()

    events = [
        json.loads(line)
        for path in sorted((tmp_path / "logs" / "pipeline").iterdir())
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    kinds = {e["event"] for e in events}
    assert {"measured", "iteration_start", "iteration_end"} <= kinds
    measured = [e for e in events if e["event"] == "measured"][0]
    assert measured["version"] == stack.base_version
    assert measured["grader_name"] == stack.grader_name
    assert measured["evaluated"] == 1
    end = [e for e in events if e["event"] == "iteration_end"][0]
    assert end["iteration"] == 1
    assert end["accepted"] in (0, 1)


def test_the_offline_stack_writes_no_pipeline_records_when_capture_is_off(tmp_path):
    """Off must mean no directory, not an empty one."""
    stack = pipeline.build_offline_stack(task_count=1)
    try:
        stack.measure(stack.base_version)
        stack.run_iterations(1)
    finally:
        stack.close()

    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# the capture CLI flags
# --------------------------------------------------------------------------- #
def test_capture_flags_are_absent_by_default_so_a_run_writes_nothing():
    """Off is the default at the outermost layer too, not only inside."""
    from scripts.run_evolution import log_capture_from_args, build_parser

    args = build_parser().parse_args(["--dry-run"])

    assert args.capture_logs is False
    assert log_capture_from_args(args).enabled is False


def test_capture_logs_turns_capture_on_for_every_channel(tmp_path):
    """The bare flag captures everything: narrowing is the deliberate act."""
    from scripts.run_evolution import log_capture_from_args, build_parser

    args = build_parser().parse_args(
        ["--dry-run", "--capture-logs", "--log-root", str(tmp_path / "logs")]
    )
    capture = log_capture_from_args(args)

    assert capture.enabled is True
    assert capture.root == tmp_path / "logs"
    assert capture.channels == ALL_LOG_CHANNELS


def test_log_channels_narrows_capture_to_the_named_channels(tmp_path):
    """Workers is the expensive channel; debugging the editor must not pay it."""
    from scripts.run_evolution import log_capture_from_args, build_parser

    args = build_parser().parse_args(
        [
            "--dry-run",
            "--capture-logs",
            "--log-root",
            str(tmp_path),
            "--log-channels",
            "editor, analyzer",
        ]
    )

    assert log_capture_from_args(args).channels == ("editor", "analyzer")


def test_a_misspelled_channel_is_refused_rather_than_silently_dropped(tmp_path):
    """A typo'd channel would disable capture for the channel that matters."""
    from scripts.run_evolution import log_capture_from_args, build_parser

    args = build_parser().parse_args(
        [
            "--dry-run",
            "--capture-logs",
            "--log-root",
            str(tmp_path),
            "--log-channels",
            "wokers",
        ]
    )

    with pytest.raises(ValueError, match="unknown log channel"):
        log_capture_from_args(args)


def test_capture_defaults_its_root_beside_the_trace_root(tmp_path):
    """A capture with nowhere to write is refused, so a root is always derived.

    Traces and logs describe the same run, so the default keeps them adjacent
    rather than making --log-root mandatory alongside --capture-logs.
    """
    from scripts.run_evolution import log_capture_from_args, build_parser

    args = build_parser().parse_args(
        ["--dry-run", "--capture-logs", "--trace-root", str(tmp_path / "traces")]
    )
    capture = log_capture_from_args(args)

    assert capture.enabled is True
    assert capture.root == tmp_path / "traces" / "logs"


def test_the_dry_run_cli_captures_pipeline_records_when_asked(tmp_path, capsys):
    """End to end through the CLI: the flag must reach the stack, not just parse."""
    code = run_evolution_main(
        [
            "--dry-run",
            "--tasks",
            "1",
            "--iterations",
            "1",
            "--capture-logs",
            "--log-root",
            str(tmp_path / "logs"),
        ]
    )

    assert code == 0
    events = {
        json.loads(line)["event"]
        for path in (tmp_path / "logs" / "pipeline").iterdir()
        for line in path.read_text().splitlines()
        if line.strip()
    }
    assert {"measured", "iteration_start", "iteration_end"} <= events


def test_the_dry_run_cli_writes_no_log_directory_by_default(tmp_path, capsys):
    """The default path must leave the filesystem untouched."""
    code = run_evolution_main(
        ["--dry-run", "--tasks", "1", "--iterations", "1", "--log-root", str(tmp_path / "logs")]
    )

    assert code == 0
    assert not (tmp_path / "logs").exists()


def test_the_benchmark_cli_builds_a_workers_only_capture(tmp_path):
    """run_benchmark has no analyzer or editor; only worker stderr exists to capture."""
    from scripts.run_benchmark import log_capture_from_args, build_parser

    args = build_parser().parse_args(
        [
            "--dataset",
            "datasets/gaia/x",
            "--grader",
            "expected_regex",
            "--replay",
            "--capture-logs",
            "--log-root",
            str(tmp_path / "logs"),
        ]
    )
    capture = log_capture_from_args(args)

    assert capture.enabled is True
    assert capture.channels == ("workers",)
    assert capture.root == tmp_path / "logs"


def test_the_benchmark_cli_defaults_capture_to_off():
    """Opt-in here too: a measurement run pays nothing unless asked."""
    from scripts.run_benchmark import log_capture_from_args, build_parser

    args = build_parser().parse_args(
        ["--dataset", "datasets/gaia/x", "--grader", "expected_regex", "--replay"]
    )

    assert log_capture_from_args(args).enabled is False


def test_capture_does_not_change_a_single_number_the_run_reports(tmp_path):
    """The instrument must not perturb what it measures.

    Capture writes on the same code path that scores, selects and accepts, so a
    record written mid-iteration must be provably inert. Same seed, same task
    set, same components -- only the capture flag moves.
    """
    from agent_evolve.core.run_logging import LogCaptureConfig

    def observed(capture):
        rollouts: list[str] = []

        class _CountingAdapter(FakeAdapter):
            """Counts rollouts, so a record that costs one is not silently free."""

            def run_full_rollout(self, workspace, task, rollout_id):
                rollouts.append(rollout_id)
                return super().run_full_rollout(workspace, task, rollout_id)

        stack = pipeline.build_offline_stack(
            task_count=3, seed=7, adapter=_CountingAdapter(), log_capture=capture
        )
        try:
            before = stack.measure(stack.base_version, prefix="before")
            lines = tuple(s.line for s in stack.run_iterations(2))
            after = stack.measure(stack.champion_version(), prefix="after")
            return before.summary, after.summary, lines, len(rollouts)
        finally:
            stack.close()

    with_capture = observed(
        LogCaptureConfig(enabled=True, root=tmp_path / "logs")
    )
    without_capture = observed(None)

    assert with_capture == without_capture
    # And the capturing run really did write, or the comparison proves nothing.
    assert any((tmp_path / "logs" / "pipeline").iterdir())


def test_a_pipeline_sink_that_cannot_write_does_not_break_the_iteration(tmp_path):
    """A full disk must not throw away rollouts already paid for.

    The pipeline records sit inside ``run_iterations`` and ``measure``, on the
    same code path that scores and accepts, so an unguarded write would convert a
    logging failure into a lost run.
    """
    from agent_evolve.core.run_logging import LogCaptureConfig, RunLogSink

    class _BrokenSink(RunLogSink):
        def write_record(self, name, record):
            raise OSError("no space left on device")

    stack = pipeline.build_offline_stack(
        task_count=1, log_capture=LogCaptureConfig(enabled=True, root=tmp_path)
    )
    stack.log_sinks = dict(stack.log_sinks) | {
        "pipeline": _BrokenSink(
            config=LogCaptureConfig(enabled=True, root=tmp_path),
            channel="pipeline",
        )
    }
    try:
        tally = stack.measure(stack.base_version, prefix="before")
        summaries = stack.run_iterations(1)
    finally:
        stack.close()

    assert tally.attempted == 1
    assert summaries[0].attempts == 1


def test_the_reported_delta_states_the_unscorable_count_on_both_sides() -> None:
    """An operator must see how many rollouts left each denominator.

    A pass rate whose denominator silently shrank between before and after is
    the exact way non-answer exclusion could mislead instead of correct.
    """
    before = ScoreTally(
        grader_name="expected_regex",
        passed=17,
        evaluated=32,
        attempted=42,
        unscorable=10,
    )
    after = ScoreTally(
        grader_name="expected_regex",
        passed=20,
        evaluated=32,
        attempted=42,
        unscorable=10,
    )

    lines = pipeline.format_delta(before, after)

    assert any("unscorable=10" in line for line in lines)
    assert any("17/32" in line for line in lines)
    assert any("delta" in line for line in lines)


def test_no_delta_is_reported_when_every_rollout_was_a_non_answer() -> None:
    """An all-non-answer run has no rate, so it cannot produce a delta."""
    before = ScoreTally(
        grader_name="expected_regex",
        passed=0,
        evaluated=0,
        attempted=5,
        unscorable=5,
    )
    after = ScoreTally(
        grader_name="expected_regex",
        passed=3,
        evaluated=5,
        attempted=5,
        unscorable=0,
    )

    lines = pipeline.format_delta(before, after)

    assert any("NOT COMPUTABLE" in line for line in lines)


# --------------------------------------------------------------------------- #
# exporting an evolved harness
# --------------------------------------------------------------------------- #
def test_an_exported_champion_round_trips_back_through_the_harness_flag(tmp_path):
    """An export that ``--harness`` cannot load is worthless.

    Before export existed, a finished run printed a pass rate and destroyed the
    harness that earned it: ``CugaAdapter._workspaces`` is in-memory only. The
    export is only a result if the same file can seed the next run, so this pins
    the round trip rather than merely the file's existence.
    """
    adapter = CugaAdapter(
        wrapper=CugaWrapper(InMemoryRuntime(), RuntimeSettings(model="test-model"))
    )
    artifacts = {
        "instructions": "base instructions",
        "skills/retrieval": "retrieve(q)",
        "policies/execution": "call the tool",
        "memory/notes": "remembered",
    }
    adapter.register_candidate("base", artifacts)
    target = tmp_path / "champion.json"

    pipeline.export_harness(adapter, version="base", candidate_id="base", path=target)
    loaded = HarnessVersion.from_path(target)

    assert loaded.instructions == "base instructions"
    assert loaded.skills == {"retrieval": "retrieve(q)"}
    assert loaded.policies == {"execution": "call the tool"}
    assert loaded.memory == {"notes": "remembered"}


def test_an_exported_harness_reproduces_an_edited_instructions_artifact(tmp_path):
    """The edit surviving the round trip is the entire point of exporting.

    An export that emitted the base text would report a successful evolution
    while shipping the unevolved harness, which is the failure mode the export
    exists to prevent.
    """
    adapter = CugaAdapter(
        wrapper=CugaWrapper(InMemoryRuntime(), RuntimeSettings(model="test-model"))
    )
    adapter.register_candidate("base", pipeline._harness_artifacts(VANILLA_HARNESS))
    workspace = adapter.materialize_candidate("base", "att-1")
    area = EditStagingArea(write_set=("instructions",))
    area.stage_replace("instructions", "revised instructions")
    adapter.apply_structured_edits(workspace, area.edits())
    target = tmp_path / "evolved.json"

    pipeline.export_harness(
        adapter, version=workspace.version, candidate_id="cand-1", path=target
    )

    assert HarnessVersion.from_path(target).instructions == "revised instructions"


def test_an_exported_harness_declares_a_version_naming_its_candidate(tmp_path):
    """``from_path`` refuses to guess a version from the filename, and should:
    the version is stamped onto every trace and is the only way to attribute a
    later result to this candidate."""
    adapter = CugaAdapter(
        wrapper=CugaWrapper(InMemoryRuntime(), RuntimeSettings(model="test-model"))
    )
    adapter.register_candidate("base", {"instructions": "x"})
    target = tmp_path / "out.json"

    pipeline.export_harness(
        adapter, version="base", candidate_id="i1-a2", path=target
    )
    version = HarnessVersion.from_path(target).version

    assert version.strip()
    assert "i1-a2" in version


def test_exported_provenance_does_not_stop_the_file_from_loading(tmp_path):
    """Provenance is only worth writing if ``from_path`` tolerates extra keys.

    It does -- it reads named keys via ``raw.get`` and ignores the rest -- so
    lineage travels inside the harness file itself instead of a sibling that can
    be separated from it.
    """
    adapter = CugaAdapter(
        wrapper=CugaWrapper(InMemoryRuntime(), RuntimeSettings(model="test-model"))
    )
    adapter.register_candidate("base", {"instructions": "x"})
    target = tmp_path / "out.json"

    pipeline.export_harness(
        adapter,
        version="base",
        candidate_id="cand-9",
        path=target,
        provenance={"parent_ids": ["base"], "score": 0.5},
    )
    raw = json.loads(target.read_text())

    assert raw["provenance"]["parent_ids"] == ["base"]
    assert HarnessVersion.from_path(target).instructions == "x"


def test_exporting_a_pool_writes_every_candidate_not_only_the_champion(tmp_path):
    """RHO seeding makes the frontier the interesting object, not the winner.

    Exporting only the champion would discard every sibling proposal the run
    paid rollouts for, and those are exactly what the next run seeds from.
    """
    stack = pipeline.build_offline_stack(task_count=2)
    try:
        stack.run_iterations(1)
        exported = stack.export_pool(tmp_path / "harnesses")
    finally:
        stack.close()

    assert len(stack.pool) > 1, "no sibling was accepted, so nothing was proven"
    candidate_files = sorted((tmp_path / "harnesses").glob("candidate-*.json"))
    assert len(candidate_files) == len(stack.pool)
    assert set(candidate_files) <= set(exported)
    versions = {HarnessVersion.from_path(p).version for p in candidate_files}
    assert len(versions) == len(stack.pool)


def test_an_artifact_with_no_harness_slot_is_preserved_rather_than_dropped(tmp_path):
    """An adapter may hold artifacts CUGA has no slot for (``FakeAdapter`` holds
    ``prompts/system``).

    Two wrong options: drop it, and the export silently claims to be the measured
    harness while missing an artifact; reinterpret it as a skill, and the next run
    loads something the agent never had. It is kept verbatim under provenance,
    where it is recoverable but cannot be mistaken for something CUGA loaded.
    """
    adapter = FakeAdapter()
    target = tmp_path / "out.json"

    pipeline.export_harness(
        adapter, version="base-v0", candidate_id="base", path=target
    )
    payload = json.loads(target.read_text())

    assert "prompts/system" not in payload
    assert payload["provenance"]["unexported_artifacts"]["prompts/system"] == (
        "You are a helpful assistant."
    )
    assert sorted(payload["skills"]) == ["retrieval"]
    assert HarnessVersion.from_path(target).skills == {
        "retrieval": "retrieve(query): return top_k docs by bm25"
    }


def test_exporting_a_pool_names_the_champion_file_unambiguously(tmp_path):
    """An operator must be able to find the winner without re-deriving selection."""
    stack = pipeline.build_offline_stack(task_count=2)
    try:
        stack.run_iterations(1)
        stack.export_pool(tmp_path / "harnesses")
        champion = stack.champion_version()
    finally:
        stack.close()

    loaded = HarnessVersion.from_path(tmp_path / "harnesses" / "champion.json")
    assert loaded.version == pipeline.harness_version_name(
        stack.pool.select_champion(config=stack.runner.config).candidate_id
    )
    assert json.loads(
        (tmp_path / "harnesses" / "champion.json").read_text()
    )["provenance"]["candidate_version"] == champion


def test_the_dry_run_cli_exports_a_loadable_harness_directory(tmp_path):
    """End to end through the CLI: the flag must reach the stack, not just parse."""
    target = tmp_path / "harnesses"

    code = run_evolution_main(
        ["--dry-run", "--tasks", "1", "--iterations", "1", "--export-harness", str(target)]
    )

    assert code == 0
    assert HarnessVersion.from_path(target / "champion.json").version.strip()


def test_the_dry_run_cli_exports_a_single_file_when_the_path_ends_in_json(tmp_path):
    """A ``.json`` target means "just the champion", so a caller wiring the next
    run's ``--harness`` needs no directory listing."""
    target = tmp_path / "champion.json"

    code = run_evolution_main(
        ["--dry-run", "--tasks", "1", "--iterations", "1", "--export-harness", str(target)]
    )

    assert code == 0
    assert target.is_file()
    assert HarnessVersion.from_path(target).version.strip()


def test_export_is_off_by_default_and_creates_no_file(tmp_path, monkeypatch):
    """Default must leave the filesystem untouched, like log capture does.

    Runs from an empty directory so a stray relative default -- the easy way to
    write files a measurement run never asked for -- shows up as a new entry.
    """
    from scripts.run_evolution import build_parser

    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["--dry-run"])
    code = run_evolution_main(["--dry-run", "--tasks", "1", "--iterations", "1"])

    assert code == 0
    assert args.export_harness is None
    assert list(tmp_path.iterdir()) == []


def test_the_inert_run_warning_does_not_claim_multi_task_inertness():
    """The old warning told operators multi-task runs could never accept an edit.

    That was true of the pre-fix ``weighted_net_gain`` and is false now (see
    ``tests/test_editor.py::test_passing_regression_probes_are_free_at_every_probe_count``).
    Asserted against the text itself rather than a run's stdout, because a run
    that accepts an edit prints no warning and would pass vacuously.
    """
    text = pipeline.nothing_accepted_warning(3).lower()

    assert "arithmetically inert" not in text
    assert "--tasks 1" not in text
    assert "-1.0" not in text
    assert "weighted_net_gain" not in text


def test_the_inert_run_warning_still_names_the_real_reasons(capsys):
    """A silently inert run is worse than a loud one, so the diagnostic stays --
    it just has to list causes an operator can actually act on."""
    stack = pipeline.build_offline_stack(task_count=3)
    try:
        text = pipeline.nothing_accepted_warning(len(stack.tasks))
    finally:
        stack.close()

    lowered = text.lower()
    assert "nothing was accepted" in lowered
    for reason in ("no issue", "declined", "validation", "budget"):
        assert reason in lowered
