"""Tests for the parallel benchmark runner.

No network, no model calls. Concurrency is exercised with sleeps and events so
the properties under test (input ordering, per-worker executor identity, failure
isolation, timeout accounting, denominator honesty) are observable.
"""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path
from typing import Mapping

import pytest

from agent_evolve.benchmarks import (
    BenchmarkRunResult,
    BenchmarkTask,
    GradingUnavailableError,
    TaskExecution,
    TaskOutcome,
    UnknownTaskError,
    run_benchmark,
)

# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #

GRADER_EXACT = "exact"


class FakeBenchmark:
    """Minimal in-memory Benchmark: passes when the answer equals the expected.

    ``ungradable`` names tasks for which the grader has no material, so it
    raises ``GradingUnavailableError`` -- the "no measurement" case.
    """

    name = "fake"

    def __init__(
        self,
        answers: Mapping[str, str],
        *,
        ungradable: frozenset[str] = frozenset(),
    ) -> None:
        self._answers = dict(answers)
        self._ungradable = frozenset(ungradable)

    def load_tasks(self) -> tuple[BenchmarkTask, ...]:
        return tuple(
            BenchmarkTask(task_id=tid, question=f"question for {tid}")
            for tid in self._answers
        )

    def grading_for(self, task_id: str):  # pragma: no cover - unused here
        return None

    def graders(self) -> tuple[str, ...]:
        return (GRADER_EXACT,)

    def score(self, task_id: str, answer: str, *, grader: str) -> TaskOutcome:
        if task_id not in self._answers:
            raise UnknownTaskError(f"unknown task_id {task_id!r}")
        if task_id in self._ungradable:
            raise GradingUnavailableError(f"no material for {task_id!r}")
        passed = answer == self._answers[task_id]
        return TaskOutcome(
            task_id=task_id,
            score=1.0 if passed else 0.0,
            passed=passed,
            grader_name=grader,
        )

    def score_all(self, task_id: str, answer: str) -> Mapping[str, TaskOutcome]:
        return {GRADER_EXACT: self.score(task_id, answer, grader=GRADER_EXACT)}


def _bench(n: int, **kwargs) -> FakeBenchmark:
    return FakeBenchmark({f"t{i:02d}": f"answer-{i:02d}" for i in range(n)}, **kwargs)


def _correct_factory():
    """Executor factory whose executor answers every task correctly."""

    def factory():
        def execute(task: BenchmarkTask) -> str:
            return f"answer-{task.task_id[1:]}"

        return execute

    return factory


# --------------------------------------------------------------------------- #
# ordering
# --------------------------------------------------------------------------- #


def test_results_are_returned_in_input_order_under_randomized_delays():
    tasks = _bench(24).load_tasks()
    bench = _bench(24)
    rng = random.Random(1234)
    delays = {t.task_id: rng.uniform(0.001, 0.05) for t in tasks}

    def factory():
        def execute(task: BenchmarkTask) -> str:
            time.sleep(delays[task.task_id])
            return f"answer-{task.task_id[1:]}"

        return execute

    result = run_benchmark(
        bench, factory, grader=GRADER_EXACT, max_workers=8, tasks=tasks
    )

    assert [e.task.task_id for e in result.executions] == [t.task_id for t in tasks]
    assert [o.task_id for o in result.outcomes] == [t.task_id for t in tasks]


def test_outcomes_follow_input_order_not_completion_order():
    """The first task finishes last; its outcome must still be first."""
    bench = _bench(4)
    tasks = bench.load_tasks()

    def factory():
        def execute(task: BenchmarkTask) -> str:
            if task.task_id == "t00":
                time.sleep(0.2)
            return f"answer-{task.task_id[1:]}"

        return execute

    result = run_benchmark(
        bench, factory, grader=GRADER_EXACT, max_workers=4, tasks=tasks
    )
    assert result.outcomes[0].task_id == "t00"


# --------------------------------------------------------------------------- #
# one executor per thread
# --------------------------------------------------------------------------- #


def test_one_executor_is_built_per_worker_thread_and_reused():
    bench = _bench(24)
    lock = threading.Lock()
    built = 0
    seen: list[tuple[int, int]] = []  # (executor id, thread id)

    def factory():
        nonlocal built
        with lock:
            built += 1
            my_id = built

        def execute(task: BenchmarkTask) -> str:
            with lock:
                seen.append((my_id, threading.get_ident()))
            time.sleep(0.002)
            return f"answer-{task.task_id[1:]}"

        return execute

    result = run_benchmark(bench, factory, grader=GRADER_EXACT, max_workers=4)

    assert result.ok_count == 24
    assert built <= 4, "must not build one executor per task"
    # Every executor instance was used by exactly one thread, and every thread
    # used exactly one executor instance.
    threads_per_executor: dict[int, set[int]] = {}
    executors_per_thread: dict[int, set[int]] = {}
    for exec_id, thread_id in seen:
        threads_per_executor.setdefault(exec_id, set()).add(thread_id)
        executors_per_thread.setdefault(thread_id, set()).add(exec_id)
    assert all(len(v) == 1 for v in threads_per_executor.values())
    assert all(len(v) == 1 for v in executors_per_thread.values())


def test_executor_construction_failure_is_isolated_not_raised():
    bench = _bench(3)

    def factory():
        raise RuntimeError("no credentials")

    result = run_benchmark(bench, factory, grader=GRADER_EXACT, max_workers=2)

    assert result.ok_count == 0
    assert result.failed_count == 3
    assert all("no credentials" in e.error for e in result.executions)
    assert result.pass_rate is None


# --------------------------------------------------------------------------- #
# failure isolation
# --------------------------------------------------------------------------- #


def test_one_exploding_task_does_not_lose_the_others():
    bench = _bench(20)

    def factory():
        def execute(task: BenchmarkTask) -> str:
            if task.task_id == "t07":
                raise ValueError("boom")
            return f"answer-{task.task_id[1:]}"

        return execute

    result = run_benchmark(bench, factory, grader=GRADER_EXACT, max_workers=8)

    assert result.executed_count == 20
    assert result.ok_count == 19
    assert result.failed_count == 1
    failed = [e for e in result.executions if not e.ok]
    assert failed[0].task.task_id == "t07"
    assert failed[0].answer is None
    assert "ValueError: boom" in failed[0].error
    assert failed[0].timed_out is False
    # the failed task produced no outcome at all
    assert [o.task_id for o in result.outcomes] == [
        t for t in sorted(bench._answers) if t != "t07"
    ]


def test_failed_execution_is_not_a_wrong_answer():
    """The single most important property: denominator excludes non-answers."""
    bench = _bench(10)

    def factory():
        def execute(task: BenchmarkTask) -> str:
            if task.task_id in {"t00", "t01", "t02"}:
                raise RuntimeError("network down")
            return f"answer-{task.task_id[1:]}"

        return execute

    result = run_benchmark(bench, factory, grader=GRADER_EXACT, max_workers=4)

    assert result.pass_count == 7
    assert result.scored_count == 7
    assert result.pass_rate == pytest.approx(7 / 7)
    assert result.pass_rate != pytest.approx(7 / 10)
    stats = result.grader_stats
    assert stats.evaluated == 7
    assert stats.total_tasks == 10
    assert stats.is_partial is True
    assert "PARTIAL" in stats.denominator_label
    assert stats.denominator_label.startswith("7/7 of 10 tasks")


def test_full_coverage_is_not_marked_partial():
    bench = _bench(6)
    result = run_benchmark(
        bench, _correct_factory(), grader=GRADER_EXACT, max_workers=3
    )
    assert result.pass_rate == pytest.approx(1.0)
    assert result.grader_stats.is_partial is False
    assert "PARTIAL" not in result.grader_stats.denominator_label


def test_wrong_answers_are_scored_not_treated_as_failures():
    bench = _bench(4)

    def factory():
        def execute(task: BenchmarkTask) -> str:
            return "definitely wrong"

        return execute

    result = run_benchmark(bench, factory, grader=GRADER_EXACT, max_workers=2)
    assert result.ok_count == 4
    assert result.failed_count == 0
    assert result.scored_count == 4
    assert result.pass_count == 0
    assert result.pass_rate == pytest.approx(0.0)
    assert result.grader_stats.is_partial is False


# --------------------------------------------------------------------------- #
# unscorable
# --------------------------------------------------------------------------- #


def test_grading_unavailable_is_counted_separately_and_excluded():
    bench = _bench(10, ungradable=frozenset({"t00", "t01"}))
    result = run_benchmark(
        bench, _correct_factory(), grader=GRADER_EXACT, max_workers=4
    )

    assert result.ok_count == 10
    assert result.unscorable_count == 2
    assert result.scored_count == 8
    assert result.pass_count == 8
    assert result.pass_rate == pytest.approx(1.0)
    assert result.grader_stats.unavailable == 2
    assert result.grader_stats.is_partial is True
    assert set(result.unscorable_task_ids) == {"t00", "t01"}


def test_scoring_error_does_not_abort_the_run():
    """An UnknownTaskError must not discard every other task's result."""
    bench = _bench(5)
    tasks = bench.load_tasks() + (
        BenchmarkTask(task_id="not-in-benchmark", question="?"),
    )

    result = run_benchmark(
        bench, _correct_factory(), grader=GRADER_EXACT, max_workers=3, tasks=tasks
    )

    assert result.executed_count == 6
    assert result.scored_count == 5
    assert result.pass_count == 5
    assert len(result.scoring_errors) == 1
    assert result.scoring_errors[0][0] == "not-in-benchmark"
    assert "UnknownTaskError" in result.scoring_errors[0][1]


# --------------------------------------------------------------------------- #
# timeout
# --------------------------------------------------------------------------- #


def test_hung_task_is_recorded_as_timed_out_and_does_not_stall_the_run():
    bench = _bench(4)
    release = threading.Event()

    def factory():
        def execute(task: BenchmarkTask) -> str:
            if task.task_id == "t00":
                release.wait(30)
            return f"answer-{task.task_id[1:]}"

        return execute

    try:
        result = run_benchmark(
            bench,
            factory,
            grader=GRADER_EXACT,
            max_workers=4,
            task_timeout_seconds=0.2,
        )
    finally:
        release.set()

    hung = result.executions[0]
    assert hung.task.task_id == "t00"
    assert hung.ok is False
    assert hung.timed_out is True
    assert hung.answer is None
    assert hung.error
    assert result.timeout_count == 1
    assert result.ok_count == 3
    assert result.scored_count == 3
    assert result.grader_stats.is_partial is True


def test_run_terminates_when_every_worker_is_lost_to_a_timeout():
    """A hung thread cannot be killed, so its worker slot is gone for good.

    The run must still terminate and record the never-started tasks rather
    than block forever waiting for a worker that will never free up.
    """
    bench = _bench(6)
    release = threading.Event()

    def factory():
        def execute(task: BenchmarkTask) -> str:
            if task.task_id in {"t00", "t01"}:
                release.wait(30)
            return f"answer-{task.task_id[1:]}"

        return execute

    try:
        result = run_benchmark(
            bench,
            factory,
            grader=GRADER_EXACT,
            max_workers=2,
            task_timeout_seconds=0.2,
        )
    finally:
        release.set()

    assert result.executed_count == 6
    assert result.timeout_count == 2
    assert result.ok_count == 0
    # the four tasks that never got a worker are recorded, not silently dropped
    abandoned = [e for e in result.executions if not e.ok and not e.timed_out]
    assert len(abandoned) == 4
    assert all(e.answer is None and e.error for e in abandoned)
    assert result.pass_rate is None
    assert result.grader_stats.evaluated == 0


def test_inline_mode_rejects_an_overrunning_task_after_the_fact():
    bench = _bench(2)

    def factory():
        def execute(task: BenchmarkTask) -> str:
            if task.task_id == "t00":
                time.sleep(0.25)
            return f"answer-{task.task_id[1:]}"

        return execute

    result = run_benchmark(
        bench, factory, grader=GRADER_EXACT, max_workers=1, task_timeout_seconds=0.05
    )
    assert result.executions[0].timed_out is True
    assert result.executions[0].ok is False
    assert result.executions[0].answer is None
    assert result.executions[1].ok is True


def test_elapsed_seconds_is_recorded():
    bench = _bench(2)

    def factory():
        def execute(task: BenchmarkTask) -> str:
            time.sleep(0.05)
            return f"answer-{task.task_id[1:]}"

        return execute

    result = run_benchmark(bench, factory, grader=GRADER_EXACT, max_workers=2)
    assert all(e.elapsed_seconds >= 0.04 for e in result.executions)


# --------------------------------------------------------------------------- #
# inline mode / validation
# --------------------------------------------------------------------------- #


def test_max_workers_one_runs_inline_on_the_calling_thread():
    bench = _bench(5)
    caller = threading.get_ident()
    seen: list[int] = []

    def factory():
        def execute(task: BenchmarkTask) -> str:
            seen.append(threading.get_ident())
            return f"answer-{task.task_id[1:]}"

        return execute

    result = run_benchmark(bench, factory, grader=GRADER_EXACT, max_workers=1)
    assert result.ok_count == 5
    assert set(seen) == {caller}


@pytest.mark.parametrize("bad", [0, -1, -10])
def test_max_workers_must_be_at_least_one(bad):
    with pytest.raises(ValueError, match="max_workers"):
        run_benchmark(_bench(2), _correct_factory(), grader=GRADER_EXACT, max_workers=bad)


@pytest.mark.parametrize("bad", [True, 2.5, "4", None])
def test_max_workers_must_be_an_integer(bad):
    with pytest.raises(ValueError, match="max_workers"):
        run_benchmark(_bench(2), _correct_factory(), grader=GRADER_EXACT, max_workers=bad)


def test_task_timeout_must_be_positive_when_supplied():
    with pytest.raises(ValueError, match="task_timeout_seconds"):
        run_benchmark(
            _bench(2),
            _correct_factory(),
            grader=GRADER_EXACT,
            task_timeout_seconds=0,
        )


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #


def test_progress_callback_fires_once_per_task_in_input_order():
    bench = _bench(20)
    rng = random.Random(7)
    delays = {f"t{i:02d}": rng.uniform(0.001, 0.03) for i in range(20)}
    seen: list[TaskExecution] = []

    def factory():
        def execute(task: BenchmarkTask) -> str:
            time.sleep(delays[task.task_id])
            return f"answer-{task.task_id[1:]}"

        return execute

    result = run_benchmark(
        bench,
        factory,
        grader=GRADER_EXACT,
        max_workers=8,
        progress=seen.append,
    )

    assert len(seen) == 20
    assert [e.task.task_id for e in seen] == [
        e.task.task_id for e in result.executions
    ]


def test_progress_callback_failure_does_not_break_the_run():
    bench = _bench(4)

    def boom(_execution: TaskExecution) -> None:
        raise RuntimeError("bad progress hook")

    result = run_benchmark(
        bench, _correct_factory(), grader=GRADER_EXACT, max_workers=2, progress=boom
    )
    assert result.ok_count == 4


def test_progress_runs_on_the_coordinator_thread():
    bench = _bench(6)
    caller = threading.get_ident()
    seen: list[int] = []

    result = run_benchmark(
        bench,
        _correct_factory(),
        grader=GRADER_EXACT,
        max_workers=4,
        progress=lambda _e: seen.append(threading.get_ident()),
    )
    assert result.ok_count == 6
    assert set(seen) == {caller}


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def _fingerprint(result: BenchmarkRunResult):
    return (
        tuple(
            (e.task.task_id, e.ok, e.answer, e.error, e.timed_out)
            for e in result.executions
        ),
        tuple((o.task_id, o.passed, o.score, o.grader_name) for o in result.outcomes),
        result.pass_count,
        result.scored_count,
        result.unscorable_count,
        result.pass_rate,
    )


def test_identical_results_at_one_and_eight_workers():
    rng = random.Random(99)
    delays = {f"t{i:02d}": rng.uniform(0.0, 0.01) for i in range(20)}

    def factory():
        def execute(task: BenchmarkTask) -> str:
            time.sleep(delays[task.task_id])
            if task.task_id == "t05":
                raise RuntimeError("deterministic failure")
            if task.task_id == "t06":
                return "wrong"
            return f"answer-{task.task_id[1:]}"

        return execute

    a = run_benchmark(
        _bench(20, ungradable=frozenset({"t09"})),
        factory,
        grader=GRADER_EXACT,
        max_workers=1,
    )
    b = run_benchmark(
        _bench(20, ungradable=frozenset({"t09"})),
        factory,
        grader=GRADER_EXACT,
        max_workers=8,
    )
    assert _fingerprint(a) == _fingerprint(b)


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #


def test_empty_task_list_produces_an_empty_result_not_a_crash():
    bench = FakeBenchmark({})
    result = run_benchmark(bench, _correct_factory(), grader=GRADER_EXACT)
    assert result.executions == ()
    assert result.outcomes == ()
    assert result.executed_count == 0
    assert result.pass_rate is None
    assert result.grader_stats.evaluated == 0
    assert result.grader_stats.total_tasks == 0
    assert result.grader_stats.is_partial is False


def test_explicit_empty_tasks_sequence_is_honoured():
    result = run_benchmark(
        _bench(5), _correct_factory(), grader=GRADER_EXACT, tasks=[]
    )
    assert result.executed_count == 0


@pytest.mark.parametrize("bad_value", [42, None, ["a"], {"a": 1}, 3.5])
def test_non_string_answer_is_recorded_as_a_failure_not_coerced(bad_value):
    bench = _bench(3)

    def factory():
        def execute(task: BenchmarkTask):
            if task.task_id == "t01":
                return bad_value
            return f"answer-{task.task_id[1:]}"

        return execute

    result = run_benchmark(bench, factory, grader=GRADER_EXACT, max_workers=2)

    bad = result.executions[1]
    assert bad.task.task_id == "t01"
    assert bad.ok is False
    assert bad.answer is None
    assert type(bad_value).__name__ in bad.error
    assert result.scored_count == 2
    assert result.grader_stats.is_partial is True


def test_default_tasks_come_from_the_benchmark():
    bench = _bench(7)
    result = run_benchmark(bench, _correct_factory(), grader=GRADER_EXACT)
    assert result.executed_count == 7
    assert result.grader_name == GRADER_EXACT


def test_grader_name_is_recorded_on_the_result():
    result = run_benchmark(_bench(2), _correct_factory(), grader=GRADER_EXACT)
    assert result.grader_name == GRADER_EXACT
    assert result.grader_stats.grader_name == GRADER_EXACT


def test_execution_invariants_are_enforced():
    with pytest.raises(ValueError):
        TaskExecution(
            task=BenchmarkTask(task_id="x", question="q"),
            answer="a",
            ok=False,  # a failure may not carry an answer
            error="boom",
            elapsed_seconds=0.0,
            timed_out=False,
        )
    with pytest.raises(ValueError):
        TaskExecution(
            task=BenchmarkTask(task_id="x", question="q"),
            answer="a",
            ok=True,
            error="boom",  # a success may not carry an error
            elapsed_seconds=0.0,
            timed_out=False,
        )
    with pytest.raises(ValueError):
        TaskExecution(
            task=BenchmarkTask(task_id="x", question="q"),
            answer=None,
            ok=False,
            error="",  # a failure requires an error
            elapsed_seconds=0.0,
            timed_out=False,
        )


# --------------------------------------------------------------------------- #
# concurrency is real
# --------------------------------------------------------------------------- #


def test_parallel_execution_overlaps_in_time():
    """Not a timing assertion on speedup, but proof work actually overlaps."""
    bench = _bench(8)
    lock = threading.Lock()
    concurrent = 0
    peak = 0

    def factory():
        def execute(task: BenchmarkTask) -> str:
            nonlocal concurrent, peak
            with lock:
                concurrent += 1
                peak = max(peak, concurrent)
            time.sleep(0.05)
            with lock:
                concurrent -= 1
            return f"answer-{task.task_id[1:]}"

        return execute

    run_benchmark(bench, factory, grader=GRADER_EXACT, max_workers=4)
    assert peak > 1, "tasks did not overlap; execution was serial"


# --------------------------------------------------------------------------- #
# replay against real data (skipped when datasets/ is absent)
# --------------------------------------------------------------------------- #

REAL_GAIA_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "gaia"
KNOWN_RUN = REAL_GAIA_ROOT / "gaia_l1_validation__baseline__20260813_035541"

#: Pinned from ``scripts/inspect_benchmark.py`` on this run. Replaying the
#: recorded answers through the runner must reproduce it exactly; if it does
#: not, either the runner miscounts or the dataset changed.
KNOWN_EXPECTED_REGEX_PASSES = 17
KNOWN_TASK_COUNT = 42

#: Non-answer detection changed this run's *denominator*, not its numerator.
#: 10 of the 42 recorded answers end in an explicit statement of inability
#: ("I'm unable to verify the poem's stanza formatting from the available
#: sources.", gaia_23dd907f, and 9 more). None of the 10 matched its
#: ``expected_regex``, so all 10 previously counted as wrong answers and padded
#: the denominator: 17/42 = 40.48%. They committed to no claim, so they are now
#: unscorable and the honest rate is 17/32 = 53.13%. The numerator is
#: deliberately unchanged -- verified: no flagged answer passed the regex.
KNOWN_NON_ANSWERS = 10
KNOWN_SCORED_COUNT = KNOWN_TASK_COUNT - KNOWN_NON_ANSWERS


def _load_cli():
    """Import ``scripts/run_benchmark.py`` by path.

    The repo root is not on ``sys.path`` under the project's pytest config, and
    adding a conftest just for this would be a heavier change than the tests
    need. Loading by path keeps the CLI's replay logic under test without
    mutating import state for every other test module.
    """
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("_cli_run_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not KNOWN_RUN.is_dir(), reason="known Gaia run not present")
def test_replay_reproduces_the_known_pass_rate_of_a_real_run():
    """The numerator is pinned; the denominator excludes the 10 non-answers.

    ``pass_count`` must stay at 17 exactly: non-answer detection may only ever
    remove ungradeable rollouts from the denominator, never change which answers
    matched. A drop here would mean the predicate ate a passing answer.
    """
    from agent_evolve.benchmarks import GaiaBenchmark

    bench = GaiaBenchmark.from_run_dir(KNOWN_RUN)
    result = run_benchmark(
        bench,
        _load_cli().make_replay_factory(bench),
        grader="expected_regex",
        max_workers=10,
    )

    assert result.executed_count == KNOWN_TASK_COUNT
    assert result.failed_count == 0
    assert result.non_answer_count == KNOWN_NON_ANSWERS
    assert result.scored_count == KNOWN_SCORED_COUNT
    assert result.pass_count == KNOWN_EXPECTED_REGEX_PASSES
    assert result.pass_rate == pytest.approx(
        KNOWN_EXPECTED_REGEX_PASSES / KNOWN_SCORED_COUNT
    )
    assert result.grader_stats.is_partial is True


@pytest.mark.skipif(not KNOWN_RUN.is_dir(), reason="known Gaia run not present")
def test_no_flagged_non_answer_would_have_passed_the_regex_grader():
    """Proves the denominator shrank without the numerator moving.

    This is the audit that makes the 40.48% -> 53.13% change trustworthy: if any
    excluded answer had matched its ``expected_regex``, the exclusion would be
    hiding a pass and the new rate would be understated rather than corrected.
    """
    from agent_evolve.benchmarks import GaiaBenchmark

    bench = GaiaBenchmark.from_run_dir(KNOWN_RUN)
    result = run_benchmark(
        bench,
        _load_cli().make_replay_factory(bench),
        grader="expected_regex",
        max_workers=10,
    )

    for task_id in result.non_answer_task_ids:
        answer = bench.recorded_answer(task_id)
        assert answer is not None
        outcome = bench.try_score(task_id, answer, grader="expected_regex")
        assert outcome is None or not outcome.passed


@pytest.mark.skipif(not KNOWN_RUN.is_dir(), reason="known Gaia run not present")
def test_replay_is_identical_at_one_and_ten_workers_on_real_data():
    from agent_evolve.benchmarks import GaiaBenchmark

    cli = _load_cli()
    bench = GaiaBenchmark.from_run_dir(KNOWN_RUN)
    a = run_benchmark(
        bench, cli.make_replay_factory(bench), grader="expected_regex", max_workers=1
    )
    b = run_benchmark(
        bench, cli.make_replay_factory(bench), grader="expected_regex", max_workers=10
    )
    assert _fingerprint(a) == _fingerprint(b)


def test_replay_records_a_missing_answer_as_a_failure_not_an_empty_answer():
    """A task with no recorded answer must not be graded as answering "".

    Substituting an empty string would put the task in the denominator and
    count it as wrong, turning a harness that never ran into a bad score.
    """
    cli = _load_cli()

    class PartialRun:
        """Stands in for a run where some task dirs have no result.json."""

        name = "partial"

        def load_tasks(self):
            return (
                BenchmarkTask(task_id="has-answer", question="q1"),
                BenchmarkTask(task_id="no-answer", question="q2"),
            )

        def recorded_answer(self, task_id):
            return "recorded" if task_id == "has-answer" else None

        def graders(self):
            return (GRADER_EXACT,)

        def score(self, task_id, answer, *, grader):
            return TaskOutcome(
                task_id=task_id,
                score=1.0,
                passed=True,
                grader_name=grader,
            )

        def grading_for(self, task_id):  # pragma: no cover - unused
            return None

        def score_all(self, task_id, answer):  # pragma: no cover - unused
            return {}

    bench = PartialRun()
    result = run_benchmark(
        bench, cli.make_replay_factory(bench), grader=GRADER_EXACT, max_workers=2
    )

    assert result.ok_count == 1
    assert result.failed_count == 1
    assert cli.MissingRecordedAnswer.__name__ in result.executions[1].error
    assert result.executions[1].answer is None
    # the missing-answer task is absent from the denominator entirely
    assert result.scored_count == 1
    assert result.pass_rate == pytest.approx(1.0)
    assert result.grader_stats.is_partial is True
    assert result.grader_stats.denominator_label == "1/1 of 2 tasks PARTIAL"



# --------------------------------------------------------------------------- #
# non-answer detection (give-up text is unscorable, not wrong)
# --------------------------------------------------------------------------- #


def _giveup_factory(giving_up: frozenset[str]):
    """Executor factory where named tasks return real observed give-up text.

    The string is verbatim from ``data/traces/19f5417b.../causal-trace.json``.
    """

    def factory():
        def execute(task: BenchmarkTask) -> str:
            if task.task_id in giving_up:
                return "I\u2019m unable to execute the tool call in this turn."
            return f"answer-{task.task_id[1:]}"

        return execute

    return factory


def test_a_give_up_answer_is_unscorable_and_leaves_the_denominator():
    """A rollout that narrates inability is not a failure-to-match.

    Grading it would put a rollout that never committed to a claim into the
    denominator, so the pass rate would measure tool availability rather than
    agent skill.
    """
    bench = _bench(10)
    result = run_benchmark(
        bench,
        _giveup_factory(frozenset({"t00", "t01"})),
        grader=GRADER_EXACT,
        max_workers=4,
    )

    assert result.ok_count == 10
    assert result.non_answer_count == 2
    assert result.unscorable_count == 2
    assert result.scored_count == 8
    assert result.pass_count == 8
    assert result.pass_rate == pytest.approx(1.0)
    assert set(result.non_answer_task_ids) == {"t00", "t01"}


def test_a_wrong_but_committed_answer_still_counts_against_the_pass_rate():
    """The critical anti-regression: over-detection would fake a delta.

    'The answer is 42' is a real, gradeable, wrong answer. If non-answer
    detection swallowed it, every genuine failure could vanish from the
    denominator and the reported pass rate would rise for free.
    """
    bench = _bench(4)

    def factory():
        def execute(task: BenchmarkTask) -> str:
            return "The answer is 42"

        return execute

    result = run_benchmark(bench, factory, grader=GRADER_EXACT, max_workers=2)

    assert result.non_answer_count == 0
    assert result.scored_count == 4
    assert result.pass_count == 0
    assert result.pass_rate == pytest.approx(0.0)


def test_pass_rate_is_none_when_every_answer_is_a_non_answer():
    """Nothing committed means nothing to score -- not a 0% pass rate.

    Mirrors the dead-worker guarantee in test_cuga_executor.py: an empty
    denominator reports None rather than manufacturing a zero.
    """
    bench = _bench(5)
    result = run_benchmark(
        bench,
        _giveup_factory(frozenset(f"t{i:02d}" for i in range(5))),
        grader=GRADER_EXACT,
        max_workers=3,
    )

    assert result.ok_count == 5
    assert result.non_answer_count == 5
    assert result.scored_count == 0
    assert result.pass_rate is None


def test_an_empty_answer_string_is_a_non_answer_not_a_wrong_answer():
    """Observed in data/traces (5 of 235 final_outputs are empty).

    The executor returned a string, so the execution is 'ok'; the content is
    still not an answer.
    """
    bench = _bench(3)

    def factory():
        def execute(task: BenchmarkTask) -> str:
            return "   " if task.task_id == "t00" else f"answer-{task.task_id[1:]}"

        return execute

    result = run_benchmark(bench, factory, grader=GRADER_EXACT, max_workers=2)

    assert result.failed_count == 0
    assert result.non_answer_count == 1
    assert result.scored_count == 2
    assert result.pass_rate == pytest.approx(1.0)


def test_non_answers_are_visible_in_the_summary_line():
    """An invisible exclusion is how an inflated denominator went unnoticed."""
    bench = _bench(4)
    result = run_benchmark(
        bench,
        _giveup_factory(frozenset({"t00"})),
        grader=GRADER_EXACT,
        max_workers=2,
    )

    assert "non_answer=1" in result.summary


def test_grading_unavailable_and_non_answer_are_counted_separately():
    """Different facts: 'we had no key' vs 'the agent never answered'.

    Collapsing them would hide which one is degrading a run.
    """
    bench = _bench(6, ungradable=frozenset({"t00"}))
    result = run_benchmark(
        bench,
        _giveup_factory(frozenset({"t01"})),
        grader=GRADER_EXACT,
        max_workers=3,
    )

    assert result.non_answer_count == 1
    assert result.ungradable_count == 1
    assert result.unscorable_count == 2
    assert result.scored_count == 4
    assert result.non_answer_task_ids == ("t01",)
    assert result.ungradable_task_ids == ("t00",)
