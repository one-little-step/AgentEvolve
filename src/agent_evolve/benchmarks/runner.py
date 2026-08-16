"""Bounded-concurrency execution of a benchmark's tasks, then scoring.

Why this module exists
---------------------
Both of the layer's consumers need the same primitive:

* an **inference / baseline run** executes every task of a benchmark once and
  reports a pass rate;
* **rollout execution during evolution** executes a selected subset of tasks
  with a candidate's harness and reports per-task outcomes.

Neither wants to reimplement fan-out, failure isolation, timeout accounting, or
denominator bookkeeping, and the two must not drift apart -- a baseline computed
one way and a rollout scored another way are not comparable numbers.

Conventions inherited from :mod:`agent_evolve.core.parallel_analysis`
--------------------------------------------------------------------
* **A factory, not a shared executor.** An agent runtime is stateful (a CUGA
  agent carries conversation state across calls), so sharing one instance across
  threads would interleave two task trajectories into one conversation. Callers
  pass ``executor_factory``; each worker thread builds exactly one executor and
  reuses it for every task it handles, because construction is expensive.
* **Input order out, never completion order.** Selection, entropy accounting and
  run-to-run diffing must not vary with thread scheduling.
* **Per-item failure is data, not an exception.** One exploding task must not
  discard the other 41.
* **``max_workers=1`` runs inline**, with no executor and no worker threads, so
  a sequential debugging session shows a straightforward stack.
* **No shared mutable accounting inside workers.** Scoring and all counting
  happen on the coordinator thread.

The correctness property that matters most
-----------------------------------------
A task that produced no answer is **not** a wrong answer. Real runs routinely
contain tasks that never wrote a result (one observed Gaia run had 10 such task
directories out of 42). Dividing passes by total-tasks in that situation reports
a lower score for a *broken harness* as though the agent had merely answered
badly, which is precisely how a broken run silently becomes a bad measurement.
:attr:`BenchmarkRunResult.pass_rate` therefore divides by the number of tasks
actually scored, and :attr:`BenchmarkRunResult.grader_stats` carries the same
``is_partial`` / ``denominator_label`` discipline as
:class:`agent_evolve.benchmarks.base.GraderStats`.

This module is agent-neutral and imports no agent implementation.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from dataclasses import dataclass, field
from time import monotonic
from typing import Callable, Sequence

from .base import (
    Benchmark,
    BenchmarkTask,
    GraderStats,
    GradingUnavailableError,
    TaskOutcome,
    UnknownGraderError,
)

__all__ = [
    "BenchmarkRunResult",
    "TaskExecution",
    "TaskExecutor",
    "TaskExecutorFactory",
    "run_benchmark",
]

#: Executes exactly one task and returns the agent's answer.
TaskExecutor = Callable[[BenchmarkTask], str]

#: Builds one :data:`TaskExecutor`. Called once per worker thread.
TaskExecutorFactory = Callable[[], TaskExecutor]

#: How often the coordinator re-checks a future while waiting. Small enough that
#: a timeout is honoured promptly, large enough not to spin.
_POLL_SECONDS = 0.01


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TaskExecution:
    """What happened when one task was executed. A failure is data, not a raise.

    ``ok=False`` means *no answer was produced*: the executor raised, the task
    exceeded its timeout, it never got a worker, or it returned a non-string.
    Such a task is excluded from the scoring denominator entirely -- it is not a
    wrong answer, and conflating the two corrupts the measurement.

    ``timed_out=True`` narrows that to the specific case of a task that started
    and overran its deadline. A failed-but-not-timed-out task either raised or
    never started; see ``error`` for which.
    """

    task: BenchmarkTask
    answer: str | None
    ok: bool
    error: str
    elapsed_seconds: float
    timed_out: bool

    def __post_init__(self) -> None:
        if self.elapsed_seconds < 0:
            raise ValueError("TaskExecution.elapsed_seconds must be >= 0")
        if self.ok:
            if self.error:
                raise ValueError("a successful execution must not carry an error")
            if not isinstance(self.answer, str):
                raise ValueError("a successful execution must carry a string answer")
            if self.timed_out:
                raise ValueError("a successful execution must not be marked timed_out")
        else:
            if not self.error:
                raise ValueError("a failed execution requires a non-empty error")
            if self.answer is not None:
                raise ValueError(
                    "a failed execution must not carry an answer: 'no answer' and "
                    "'wrong answer' are different facts"
                )


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    """Executions and outcomes for one run of one grader, in input order.

    ``outcomes`` contains an entry only for a task that both produced an answer
    and could be graded, so ``len(outcomes)`` is the honest denominator and is
    generally smaller than ``len(executions)``.
    """

    executions: tuple[TaskExecution, ...]
    outcomes: tuple[TaskOutcome, ...]
    grader_name: str
    unscorable_task_ids: tuple[str, ...] = ()
    scoring_errors: tuple[tuple[str, str], ...] = ()
    wall_seconds: float = 0.0
    max_workers: int = 1
    task_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "executions", tuple(self.executions))
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        object.__setattr__(self, "unscorable_task_ids", tuple(self.unscorable_task_ids))
        object.__setattr__(self, "scoring_errors", tuple(self.scoring_errors))

    # -- execution-side counts -------------------------------------------- #

    @property
    def executed_count(self) -> int:
        """Tasks the run attempted. This is *not* a scoring denominator."""
        return len(self.executions)

    @property
    def ok_count(self) -> int:
        """Tasks that produced an answer."""
        return sum(1 for e in self.executions if e.ok)

    @property
    def failed_count(self) -> int:
        """Tasks that produced no answer, for any reason."""
        return sum(1 for e in self.executions if not e.ok)

    @property
    def timeout_count(self) -> int:
        """Tasks that started and overran their deadline."""
        return sum(1 for e in self.executions if e.timed_out)

    @property
    def failed_executions(self) -> tuple[TaskExecution, ...]:
        return tuple(e for e in self.executions if not e.ok)

    # -- scoring-side counts ---------------------------------------------- #

    @property
    def scored_count(self) -> int:
        """The denominator: tasks that produced an answer *and* were graded."""
        return len(self.outcomes)

    @property
    def pass_count(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def unscorable_count(self) -> int:
        """Answered tasks the grader had no material for (not failures)."""
        return len(self.unscorable_task_ids)

    @property
    def pass_rate(self) -> float | None:
        """Passes over tasks actually scored; ``None`` when nothing was scored.

        Never divides by :attr:`executed_count`. Read
        :attr:`grader_stats` (or :attr:`summary`) to see the denominator
        alongside the number.
        """
        return self.grader_stats.pass_rate

    @property
    def grader_stats(self) -> GraderStats:
        """The run expressed in the same shape as a historical run's stats.

        ``evaluated`` is what was graded; ``total_tasks`` is what was attempted.
        ``is_partial`` is therefore true whenever any task failed to execute or
        could not be graded, which is exactly when a bare pass rate would lie.
        """
        return GraderStats(
            grader_name=self.grader_name,
            passed=self.pass_count,
            evaluated=self.scored_count,
            total_tasks=self.executed_count,
            unavailable=self.unscorable_count,
            unavailable_task_ids=self.unscorable_task_ids,
        )

    @property
    def summary(self) -> str:
        """One honest line: never a pass rate without its denominator."""
        stats = self.grader_stats
        rate = stats.pass_rate
        rate_text = "n/a" if rate is None else f"{rate * 100:.2f}%"
        return (
            f"grader={self.grader_name} {stats.denominator_label} "
            f"pass_rate={rate_text} "
            f"attempted={self.executed_count} answered={self.ok_count} "
            f"failed={self.failed_count} timed_out={self.timeout_count} "
            f"unscorable={self.unscorable_count} "
            f"scoring_errors={len(self.scoring_errors)}"
        )


# --------------------------------------------------------------------------- #
# internal per-task slot
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Slot:
    """Coordinator-visible state for one queued task.

    ``started_at`` is written once by the worker thread that picks the task up
    and only ever read by the coordinator. A single float assignment needs no
    lock, and a stale read merely costs one extra poll.
    """

    task: BenchmarkTask
    started_at: float | None = None


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def run_benchmark(
    benchmark: Benchmark,
    executor_factory: TaskExecutorFactory,
    *,
    grader: str,
    max_workers: int = 10,
    task_timeout_seconds: float | None = None,
    tasks: Sequence[BenchmarkTask] | None = None,
    progress: Callable[[TaskExecution], None] | None = None,
) -> BenchmarkRunResult:
    """Execute ``tasks`` concurrently, then score the answers with ``grader``.

    :param benchmark: supplies the tasks (when ``tasks`` is omitted) and does
        the scoring. Scoring runs on the calling thread, never in a worker.
    :param executor_factory: called **once per worker thread**; the returned
        callable is reused for every task that thread handles. Pass a factory,
        not an executor: a stateful agent shared across threads would interleave
        two task trajectories into one conversation.
    :param grader: the grader name, always explicit. Validated up front so a
        typo costs nothing instead of a full run.
    :param max_workers: bounded concurrency, ``>= 1``. ``1`` runs inline with no
        executor. Defaults to 10, matching the observed baseline configuration.
    :param task_timeout_seconds: per-task wall-clock budget, or ``None`` for no
        limit. See the guarantees below.
    :param tasks: subset to run; defaults to ``benchmark.load_tasks()``.
    :param progress: called once per task, **on the calling thread**, in input
        order, as each task's result is resolved. An exception raised by the
        callback is swallowed: an observability hook must not fail a run. Note
        the ordering consequence: with ``max_workers=10``, if task 0 is the
        slowest, no progress fires until task 0 resolves, after which the
        already-finished tasks 1..n report in a burst. Ordered, reproducible
        progress was chosen over completion-order progress because the returned
        results are input-ordered and a progress stream that disagreed with them
        would be misleading.

    What the timeout guarantees, and what it does not
    ------------------------------------------------
    **Guaranteed.** No single task can stall the run: once a task exceeds
    ``task_timeout_seconds`` the coordinator stops waiting for it, records
    ``timed_out=True`` with no answer, and moves on. The run terminates even if
    every worker is lost to a hung task, and ``run_benchmark`` itself returns.

    **Not guaranteed: the task actually stops.** Python threads cannot be
    forcibly killed. A timed-out task's thread keeps running its executor to
    completion, which means that after a timeout is recorded:

    * the thread may still hold a browser session, a socket, a subprocess or a
      file handle, and may still be billing model tokens;
    * its worker slot stays occupied, so effective concurrency drops. Once all
      ``max_workers`` slots are held by timed-out-but-still-running tasks, no
      queued task can start; those tasks are cancelled and recorded as failures
      with ``timed_out=False`` (they never ran) rather than left to hang;
    * whatever the executor eventually returns or raises is discarded, and any
      side effect it performs after the deadline still happens;
    * the interpreter will not exit while such a thread lives, because the pool
      is not shut down with ``wait=True``. Callers who need hard termination
      must give the executor its own cancellation mechanism (process isolation,
      an SDK-level deadline, or a cooperative cancel token). This runner cannot
      provide it, and does not pretend to.

    With ``max_workers=1`` the check is necessarily post-hoc: the task runs to
    completion inline and is then *rejected* for having overrun. Nothing is
    interrupted; the elapsed time recorded is the true elapsed time. The same
    post-hoc rejection also applies in parallel mode to a task that finishes
    just past its deadline, so the two modes agree on what "over budget" means.

    Determinism
    -----------
    For a deterministic executor the result is identical at any ``max_workers``:
    execution order does not affect returned order, and scoring is sequential on
    the coordinator. Only ``elapsed_seconds``, ``wall_seconds`` and
    timeout-dependent outcomes can differ.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise ValueError(f"max_workers must be a positive integer; got {max_workers!r}")
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1; got {max_workers}")
    if task_timeout_seconds is not None:
        if isinstance(task_timeout_seconds, bool) or not isinstance(
            task_timeout_seconds, (int, float)
        ):
            raise ValueError(
                f"task_timeout_seconds must be a positive number or None; "
                f"got {task_timeout_seconds!r}"
            )
        if task_timeout_seconds <= 0:
            raise ValueError(
                f"task_timeout_seconds must be > 0 when supplied; "
                f"got {task_timeout_seconds!r}"
            )

    available = _graders_of(benchmark)
    if available is not None and grader not in available:
        raise UnknownGraderError(
            f"unknown grader {grader!r} for benchmark "
            f"{getattr(benchmark, 'name', type(benchmark).__name__)!r}; "
            f"available: {available}"
        )

    selected = tuple(benchmark.load_tasks() if tasks is None else tasks)

    started = monotonic()
    if not selected:
        executions: tuple[TaskExecution, ...] = ()
    elif max_workers == 1:
        executions = _execute_inline(
            selected,
            executor_factory,
            task_timeout_seconds=task_timeout_seconds,
            progress=progress,
        )
    else:
        executions = _execute_parallel(
            selected,
            executor_factory,
            max_workers=max_workers,
            task_timeout_seconds=task_timeout_seconds,
            progress=progress,
        )
    wall = monotonic() - started

    outcomes, unscorable, scoring_errors = _score_all(benchmark, executions, grader)

    return BenchmarkRunResult(
        executions=executions,
        outcomes=outcomes,
        grader_name=grader,
        unscorable_task_ids=unscorable,
        scoring_errors=scoring_errors,
        wall_seconds=wall,
        max_workers=max_workers,
        task_timeout_seconds=task_timeout_seconds,
    )


def _graders_of(benchmark: Benchmark) -> tuple[str, ...] | None:
    """Declared grader names, or ``None`` when the benchmark cannot say."""
    getter = getattr(benchmark, "graders", None)
    if getter is None:
        return None
    try:
        return tuple(getter())
    except Exception:  # noqa: BLE001 - a benchmark that cannot enumerate graders
        return None


# --------------------------------------------------------------------------- #
# execution: inline
# --------------------------------------------------------------------------- #


def _execute_inline(
    tasks: Sequence[BenchmarkTask],
    executor_factory: TaskExecutorFactory,
    *,
    task_timeout_seconds: float | None,
    progress: Callable[[TaskExecution], None] | None,
) -> tuple[TaskExecution, ...]:
    """Run every task on the calling thread. No executor, no worker threads."""
    holder: list[TaskExecutor] = []
    executions: list[TaskExecution] = []
    for task in tasks:
        execution = _reject_if_over_budget(
            _execute_one(task, executor_factory, holder), task_timeout_seconds
        )
        executions.append(execution)
        _notify(progress, execution)
    return tuple(executions)


def _reject_if_over_budget(
    execution: TaskExecution, task_timeout_seconds: float | None
) -> TaskExecution:
    """Reject a *finished* task that took longer than its budget.

    Applied identically in inline and parallel mode, and judged against the
    task's own ``elapsed_seconds`` rather than against the coordinator's
    wall clock. Judging by wall clock would mark a task that finished in 1ms as
    timed out merely because the coordinator only got around to reading its
    result after the budget had passed -- making the verdict depend on thread
    scheduling, which is exactly what must not happen.

    Nothing is interrupted here: the task already ran to completion. This is a
    budget rejection, so its result is discarded rather than graded.
    """
    if (
        not execution.ok
        or task_timeout_seconds is None
        or execution.elapsed_seconds <= task_timeout_seconds
    ):
        return execution
    return TaskExecution(
        task=execution.task,
        answer=None,
        ok=False,
        error=(
            f"task exceeded task_timeout_seconds={task_timeout_seconds}: took "
            f"{execution.elapsed_seconds:.3f}s; it ran to completion (a Python "
            f"thread cannot be interrupted) and its answer was rejected as "
            f"over budget"
        ),
        elapsed_seconds=execution.elapsed_seconds,
        timed_out=True,
    )


# --------------------------------------------------------------------------- #
# execution: parallel
# --------------------------------------------------------------------------- #


def _execute_parallel(
    tasks: Sequence[BenchmarkTask],
    executor_factory: TaskExecutorFactory,
    *,
    max_workers: int,
    task_timeout_seconds: float | None,
    progress: Callable[[TaskExecution], None] | None,
) -> tuple[TaskExecution, ...]:
    """Run tasks with bounded concurrency, resolving results in input order."""
    workers = min(max_workers, len(tasks))
    slots = [_Slot(task=task) for task in tasks]
    local = threading.local()

    def work(slot: _Slot) -> TaskExecution:
        holder = getattr(local, "holder", None)
        if holder is None:
            holder = []
            local.holder = holder
        slot.started_at = monotonic()
        return _execute_one(slot.task, executor_factory, holder)

    # The pool is not used as a context manager: `__exit__` joins every worker,
    # which would re-introduce the very stall the timeout exists to prevent when
    # a task is hung. Shutdown is explicit and non-joining below.
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="benchtask")
    executions: list[TaskExecution] = []
    try:
        futures = [pool.submit(work, slot) for slot in slots]
        # Futures whose task overran its deadline and may still be occupying a
        # worker thread. A worker is only reclaimed if such a future completes.
        stuck: list[Future[TaskExecution]] = []

        for slot, future in zip(slots, futures):
            if slot.started_at is None and _stuck_workers(stuck) >= workers:
                # Every worker is held by a task we already gave up on, and
                # those threads cannot be killed. This task will never start, so
                # record that honestly instead of blocking forever.
                if future.cancel():
                    executions.append(_never_started(slot.task, workers))
                    _notify(progress, executions[-1])
                    continue
            execution = _await(future, slot, task_timeout_seconds)
            if execution.timed_out:
                stuck.append(future)
            executions.append(execution)
            _notify(progress, execution)
    finally:
        # wait=False: a hung executor must not keep the coordinator (or a test
        # session) blocked. cancel_futures reclaims anything still queued.
        pool.shutdown(wait=False, cancel_futures=True)
    return tuple(executions)


def _stuck_workers(stuck: Sequence[Future[TaskExecution]]) -> int:
    """How many worker threads are still held by timed-out tasks.

    A timed-out task whose future has since completed released its worker, so it
    no longer counts. This keeps the "no worker will ever free up" judgement
    accurate rather than pessimistic.
    """
    return sum(1 for future in stuck if not future.done())


def _await(
    future: Future[TaskExecution],
    slot: _Slot,
    task_timeout_seconds: float | None,
) -> TaskExecution:
    """Wait for one task, enforcing its deadline from when it actually started.

    Two properties are load-bearing here:

    * **A finished task is never called timed out.** Its result is collected
      before any deadline arithmetic, and the verdict then depends only on the
      task's own ``elapsed_seconds``. Otherwise a task that finished in 1ms
      would be reported as timed out simply because the coordinator was busy
      waiting on an earlier, slower task -- making the verdict a function of
      thread scheduling.
    * **The clock starts when a worker picks the task up**, not at submission.
      Time spent queued behind other tasks is the runner's latency, not the
      task's, and charging it to the task would spuriously time out later tasks
      in a large batch.
    """
    if task_timeout_seconds is None:
        try:
            return future.result()
        except Exception as exc:  # noqa: BLE001 - failures are returned as data
            return _failed(slot.task, exc, elapsed=0.0)

    while True:
        # Collect first: a completed task is judged on its own elapsed time.
        if future.done():
            try:
                return _reject_if_over_budget(future.result(), task_timeout_seconds)
            except Exception as exc:  # noqa: BLE001 - returned as data
                return _failed(slot.task, exc, elapsed=0.0)

        started_at = slot.started_at
        if started_at is None:
            # Still queued: its deadline has not begun. Poll until it starts.
            wait = _POLL_SECONDS
        else:
            remaining = started_at + task_timeout_seconds - monotonic()
            if remaining <= 0:
                elapsed = monotonic() - started_at
                return TaskExecution(
                    task=slot.task,
                    answer=None,
                    ok=False,
                    error=(
                        f"task exceeded task_timeout_seconds="
                        f"{task_timeout_seconds} after {elapsed:.3f}s; the "
                        f"coordinator stopped waiting, but the worker thread "
                        f"cannot be killed and may still be running"
                    ),
                    elapsed_seconds=elapsed,
                    timed_out=True,
                )
            wait = min(remaining, _POLL_SECONDS)
        try:
            return _reject_if_over_budget(
                future.result(timeout=wait), task_timeout_seconds
            )
        except _FutureTimeout:
            continue
        except Exception as exc:  # noqa: BLE001 - failures are returned as data
            return _failed(slot.task, exc, elapsed=0.0)


# --------------------------------------------------------------------------- #
# executing one task
# --------------------------------------------------------------------------- #


def _execute_one(
    task: BenchmarkTask,
    executor_factory: TaskExecutorFactory,
    holder: list[TaskExecutor],
) -> TaskExecution:
    """Execute one task, converting any failure into a recorded execution.

    ``holder`` caches this thread's executor. Construction failure is isolated
    per task too: a factory that needs credentials should surface a
    missing-configuration error, not abort the batch.
    """
    started = monotonic()
    try:
        if not holder:
            holder.append(executor_factory())
        answer = holder[0](task)
    except BaseException as exc:  # noqa: BLE001 - failures are returned as data
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _failed(task, exc, elapsed=monotonic() - started)
    elapsed = monotonic() - started
    if not isinstance(answer, str):
        # Do not coerce. str(None) == "None" would be graded as a real answer,
        # and a silently stringified object is an invented answer.
        return TaskExecution(
            task=task,
            answer=None,
            ok=False,
            error=(
                f"executor returned {type(answer).__name__}, not str; refusing "
                f"to coerce a non-answer into an answer"
            ),
            elapsed_seconds=elapsed,
            timed_out=False,
        )
    return TaskExecution(
        task=task,
        answer=answer,
        ok=True,
        error="",
        elapsed_seconds=elapsed,
        timed_out=False,
    )


def _failed(task: BenchmarkTask, exc: BaseException, *, elapsed: float) -> TaskExecution:
    return TaskExecution(
        task=task,
        answer=None,
        ok=False,
        error=f"{type(exc).__name__}: {exc}",
        elapsed_seconds=max(elapsed, 0.0),
        timed_out=False,
    )


def _never_started(task: BenchmarkTask, workers: int) -> TaskExecution:
    return TaskExecution(
        task=task,
        answer=None,
        ok=False,
        error=(
            f"task never started: all {workers} worker(s) are held by "
            f"timed-out tasks whose threads cannot be killed, so no worker "
            f"will free up; cancelled rather than waited on"
        ),
        elapsed_seconds=0.0,
        timed_out=False,
    )


def _notify(
    progress: Callable[[TaskExecution], None] | None, execution: TaskExecution
) -> None:
    if progress is None:
        return
    try:
        progress(execution)
    except Exception:  # noqa: BLE001 - observability must not fail a run
        pass


# --------------------------------------------------------------------------- #
# scoring (coordinator thread only)
# --------------------------------------------------------------------------- #


def _score_all(
    benchmark: Benchmark,
    executions: Sequence[TaskExecution],
    grader: str,
) -> tuple[tuple[TaskOutcome, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Score every answered task, sequentially, on the calling thread.

    Scoring is cheap next to execution and a benchmark may be stateful, so it is
    deliberately not parallelised. Three outcomes are kept apart:

    * a graded task contributes a :class:`TaskOutcome` (pass or fail);
    * ``GradingUnavailableError`` means "no measurement" and is recorded as
      unscorable, excluded from the denominator;
    * any other scoring exception is recorded and the run continues, so one
      unknown task id cannot discard 41 valid results.
    """
    outcomes: list[TaskOutcome] = []
    unscorable: list[str] = []
    errors: list[tuple[str, str]] = []
    for execution in executions:
        if not execution.ok:
            continue
        assert execution.answer is not None  # guaranteed by TaskExecution
        try:
            outcomes.append(
                benchmark.score(
                    execution.task.task_id, execution.answer, grader=grader
                )
            )
        except GradingUnavailableError:
            unscorable.append(execution.task.task_id)
        except Exception as exc:  # noqa: BLE001 - one bad task must not abort
            errors.append(
                (execution.task.task_id, f"{type(exc).__name__}: {exc}")
            )
    return tuple(outcomes), tuple(unscorable), tuple(errors)
