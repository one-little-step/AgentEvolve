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
from agent_evolve.benchmarks.base import (  # noqa: E402
    BenchmarkGrading,
    BenchmarkTask,
    GradingUnavailableError,
    TaskOutcome,
    UnknownGraderError,
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
