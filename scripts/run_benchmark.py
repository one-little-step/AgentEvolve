"""Run a benchmark's tasks through the parallel runner and report the score.

Three modes:

``--replay`` (no model calls)
    "Executes" each task by returning the answer that run already recorded in
    ``tasks/<task>/result.json``. This exercises the real runner -- real fan-out,
    real per-task failure isolation, real scoring, real denominator accounting --
    against real data, with zero network use, and must reproduce the run's known
    pass rate. A task directory with no recorded answer is reported as a
    *failed execution*, not as a wrong answer, which is the whole point: one
    observed run had 10 of 42 task directories with no ``result.json``.

``--execute --harness <name-or-path>`` (real CUGA rollouts)
    Runs an actual CUGA agent per task against the named harness version, with
    causal tracing mandatory. ``--harness`` is required and never defaulted: a
    pass rate that cannot name the harness that produced it is not a
    measurement. Every rollout must write a trace; a task that produces an
    answer but no trace is discarded as a failed execution, because the trace is
    the evidence the analyzer and proxy validator consume. See
    ``agent_evolve.benchmarks.cuga_executor``.

    ``--max-workers > 1`` requires ``--isolation process``, which gives each
    worker its own CUGA subprocess. In-process (threaded) concurrency is refused,
    and the reason is measured rather than assumed: ``CUGA_FOLDER`` is a single
    environment variable shared by every thread and read during ``invoke()``, so
    two concurrent candidates can silently swap workspaces while their traces
    still stamp their own ``harness_version``. See
    ``agent_evolve.benchmarks.cuga_process_pool``.

``--executor module:factory`` (bring your own agent)
    Live mode with no agent-specific knowledge in this script.

No grading material is printed -- only aggregate counts and per-task pass/fail.

Usage::

    uv run python scripts/run_benchmark.py \\
        --dataset datasets/gaia/gaia_l1_validation__baseline__20260813_035541 \\
        --grader expected_regex --replay --max-workers 10

    uv run python scripts/run_benchmark.py --dataset <run> --grader expected_regex \\
        --execute --harness vanilla --limit 2 --max-workers 2 --isolation process

    uv run python scripts/run_benchmark.py --dataset <run> --grader expected_regex \\
        --replay --max-workers 1 --limit 10 --task-timeout 1200
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable, Protocol, Sequence, runtime_checkable

from agent_evolve.benchmarks import (
    BenchmarkRunResult,
    BenchmarkTask,
    GaiaBenchmark,
    TaskExecution,
    TaskExecutor,
    run_benchmark,
)
from agent_evolve.benchmarks.cuga_executor import (
    BUILTIN_HARNESS_NAMES,
    DEFAULT_TRACE_ROOT,
    PROCESS_ISOLATION,
    THREAD_ISOLATION,
    CugaExecutorError,
    HarnessVersion,
    TraceRecorder,
    make_cuga_executor_factory,
    missing_trace_task_ids,
    preflight,
)
from agent_evolve.benchmarks.cuga_process_pool import (
    CugaProcessPool,
    default_knowledge_seed,
)
from agent_evolve.core.run_logging import LogCaptureConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Default per-task budget, matching the observed baseline configuration.
DEFAULT_TASK_TIMEOUT = 1200.0
DEFAULT_MAX_WORKERS = 10

#: Where each worker's private knowledge and policy stores live.
#:
#: Under ``data/`` rather than a temp dir so a run's isolation is inspectable
#: afterwards -- "which store did worker w0003 use" is a question that comes up
#: when a parallel run and a serial run disagree.
DEFAULT_WORKER_ROOT = Path("data/workers")


class MissingRecordedAnswer(RuntimeError):
    """Raised in replay mode when a task never recorded an answer.

    Deliberately an error rather than an empty string: a task that produced no
    answer must land in ``failed_count``, not be graded as an empty (wrong)
    answer. Substituting ``""`` would silently convert a broken harness into a
    low score.
    """


@runtime_checkable
class RecordedAnswerSource(Protocol):
    """The only thing replay needs from a benchmark: what the run answered.

    Deliberately narrower than :class:`GaiaBenchmark` so replay is not tied to
    one benchmark adapter; any benchmark that can report its recorded answers
    can be replayed through the runner.
    """

    def recorded_answer(self, task_id: str) -> str | None: ...


def make_replay_factory(bench: RecordedAnswerSource) -> Callable[[], TaskExecutor]:
    """Build an executor factory that replays each task's recorded answer.

    The benchmark object is read-only here, so a single closure is safe across
    threads; a factory is still returned because that is the runner's contract
    and because a real agent factory must build one stateful agent per thread.
    """

    def factory() -> TaskExecutor:
        def execute(task: BenchmarkTask) -> str:
            answer = bench.recorded_answer(task.task_id)
            if answer is None:
                raise MissingRecordedAnswer(
                    f"task {task.task_id!r} recorded no answer in this run; "
                    f"replay cannot invent one"
                )
            return answer

        return execute

    return factory


def load_executor_factory(spec: str) -> Callable[[], TaskExecutor]:
    """Import a ``module:attribute`` executor factory."""
    if ":" not in spec:
        raise SystemExit(f"--executor must look like 'module:factory'; got {spec!r}")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attr, None)
    if factory is None or not callable(factory):
        raise SystemExit(f"{spec!r} does not name a callable factory")
    return factory  # type: ignore[return-value]  # verified callable at runtime


def _print_trace_report(
    recorder: TraceRecorder, tasks: Sequence[BenchmarkTask]
) -> None:
    """Report evidence, not just score.

    A run that produced answers but no traces has not produced the artifact this
    project actually consumes, so the trace count is printed next to the pass
    rate rather than buried, and any task without one is named.
    """
    records = recorder.records
    print(f"    traces written     : {len(records)}")
    if records:
        roots = sorted({str(record.trace_path.parent) for record in records})
        for root in roots[:3]:
            print(f"      under            : {root}")
        versions = sorted({record.harness_version for record in records})
        print(f"      harness_version  : {', '.join(versions)}")
        print(f"      example trace    : {records[0].trace_path}")
    if tasks:
        missing = missing_trace_task_ids(tasks, recorder)
        if missing:
            print(
                f"    !! {len(missing)} task(s) produced NO trace and yield no "
                f"analyzable evidence:"
            )
            for task_id in missing[:20]:
                print(f"      {task_id}")


def _print_report(
    result: BenchmarkRunResult,
    *,
    verbose: bool,
    recorder: TraceRecorder | None = None,
    tasks: Sequence[BenchmarkTask] = (),
) -> None:
    stats = result.grader_stats
    rate = stats.pass_rate

    print("\n=== result")
    print(f"    grader             : {result.grader_name}")
    print(f"    max_workers        : {result.max_workers}")
    print(f"    task_timeout       : {result.task_timeout_seconds}")
    print(f"    wall seconds       : {result.wall_seconds:.2f}")
    print(f"    tasks attempted    : {result.executed_count}")
    print(f"    answered           : {result.ok_count}")
    print(f"    no answer (failed) : {result.failed_count}")
    print(f"    of which timed out : {result.timeout_count}")
    print(f"    non-answer (gave up): {result.non_answer_count}")
    print(f"    answered, ungraded : {result.ungradable_count}")
    print(f"    unscorable total   : {result.unscorable_count}")
    print(f"    scoring errors     : {len(result.scoring_errors)}")
    print(f"    scored (DENOMINATOR): {stats.evaluated}")
    print(f"    passed             : {stats.passed}")
    print(
        f"    pass rate          : "
        f"{'n/a' if rate is None else f'{stats.passed}/{stats.evaluated} = {rate:.2%}'}"
    )
    print(f"    denominator        : {stats.denominator_label}")
    if stats.is_partial:
        print(
            "    !! PARTIAL DENOMINATOR: some tasks produced no answer or could "
            "not be graded. This pass rate covers only the scored tasks; it is "
            "NOT comparable to a full-denominator run."
        )

    if recorder is not None:
        _print_trace_report(recorder, tasks)

    if result.failed_count:
        print("\n    failed executions (no answer produced):")
        for execution in result.failed_executions:
            kind = "TIMEOUT" if execution.timed_out else "ERROR  "
            print(
                f"      {kind} {execution.task.task_id:<40} "
                f"{execution.elapsed_seconds:6.2f}s  {execution.error[:110]}"
            )

    if result.non_answer_count:
        categories = dict(result.non_answer_categories)
        print(
            "\n    non-answers (agent committed no answer; EXCLUDED from the "
            "denominator, not counted as wrong):"
        )
        for task_id in result.non_answer_task_ids:
            print(f"      {task_id:<40} {categories.get(task_id, '')}")

    if result.ungradable_count:
        print("\n    answered but ungraded (no grading material, excluded):")
        for task_id in result.ungradable_task_ids:
            print(f"      {task_id}")

    for task_id, error in result.scoring_errors:
        print(f"\n    !! scoring error {task_id}: {error}")

    if verbose:
        print("\n    per-task outcomes:")
        for outcome in result.outcomes:
            print(
                f"      {'PASS' if outcome.passed else 'FAIL'} "
                f"{outcome.task_id:<40} score {outcome.score:.1f}"
            )

    print(f"\n    summary: {result.summary}")


def _select(tasks: Sequence[BenchmarkTask], limit: int | None) -> Sequence[BenchmarkTask]:
    if limit is None or limit <= 0:
        return tasks
    return tasks[:limit]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a benchmark's tasks with bounded concurrency and score them."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="benchmark run directory, e.g. datasets/gaia/<run_name>",
    )
    parser.add_argument(
        "--grader",
        required=True,
        help="grader name; always explicit, never defaulted",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            f"bounded concurrency, >= 1 (default: {DEFAULT_MAX_WORKERS} for "
            f"--replay / --executor, but 1 for --execute, because parallel real "
            f"execution requires --isolation process; 1 = inline)"
        ),
    )
    parser.add_argument(
        "--isolation",
        choices=(THREAD_ISOLATION, PROCESS_ISOLATION),
        default=THREAD_ISOLATION,
        help=(
            f"how --execute isolates workers. {THREAD_ISOLATION!r} (default) runs "
            f"rollouts in this process and is safe only at --max-workers 1. "
            f"{PROCESS_ISOLATION!r} gives every worker its own CUGA subprocess, "
            f"with its own environment and its own knowledge and policy stores, "
            f"and is required for --max-workers > 1."
        ),
    )
    parser.add_argument(
        "--worker-root",
        type=Path,
        default=DEFAULT_WORKER_ROOT,
        help=(
            f"where each process-isolated worker's private knowledge and policy "
            f"stores are created (default: {DEFAULT_WORKER_ROOT})"
        ),
    )
    parser.add_argument(
        "--empty-worker-knowledge",
        action="store_true",
        help=(
            "start each process-isolated worker with an EMPTY knowledge store "
            "instead of a copy of the one a serial run uses. Measured to change "
            "the pass rate on tiny5 (3/4 serial vs 0/3 with an empty store, at "
            "one worker as well as four), so results are not comparable to a "
            "serial run. For deliberate no-knowledge experiments only."
        ),
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=None,
        help=(
            f"per-task wall-clock budget in seconds "
            f"(omit for no limit; the observed baseline used {DEFAULT_TASK_TIMEOUT:.0f})"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N tasks (0 or omitted = all)",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help=(
            "execute by returning each task's already-recorded answer; no model "
            "calls. Reproduces the run's known pass rate."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "real CUGA rollouts against --harness, with causal tracing "
            "mandatory. Costs model tokens and wall time."
        ),
    )
    parser.add_argument(
        "--harness",
        default=None,
        help=(
            "REQUIRED with --execute: the harness version to run against. A "
            f"built-in name ({', '.join(BUILTIN_HARNESS_NAMES)}) or a path to a "
            "harness JSON file declaring 'version' plus optional "
            "'instructions'/'skills'/'memory'/'policies'."
        ),
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
        help=f"where causal traces are written (default: {DEFAULT_TRACE_ROOT})",
    )
    parser.add_argument(
        "--allow-unsafe-concurrency",
        action="store_true",
        help=(
            "permit --execute --max-workers > 1 with in-process (threaded) "
            "isolation. Tasks WILL be lost to CUGA's knowledge-engine lock, and "
            "concurrent candidates can silently swap workspaces via the "
            "process-global CUGA_FOLDER. Prefer --isolation process; this exists "
            "only for experiments that knowingly accept corrupt evidence."
        ),
    )
    parser.add_argument(
        "--executor",
        default=None,
        help="live mode: 'module:factory' returning a callable(BenchmarkTask) -> str",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="print each task as its result is resolved (coordinator thread)",
    )
    parser.add_argument(
        "--capture-logs",
        action="store_true",
        help=(
            "capture each process-isolated worker's CUGA stderr to "
            "<--log-root>/workers/<worker_id>.log. That stream is the only place "
            "CUGA reports its routing decisions (is_autonomous_subtask, "
            "'Routing to:'), and it is discarded by default -- so a finished run "
            "cannot say why it routed as it did without a paid re-run. OFF by "
            "default: on, nothing is written and no directory is created."
        ),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help=(
            "where captured worker logs are written. Defaults to "
            "<--trace-root>/logs, because traces and logs describe the same run. "
            "Ignored without --capture-logs."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print per-task pass/fail outcomes",
    )
    return parser


def log_capture_from_args(args: argparse.Namespace) -> LogCaptureConfig:
    """Build a workers-only capture config.

    Only ``workers`` is offered here: this script runs rollouts and grades them,
    with no analyzer, editor or evolution loop, so the other three channels have
    nothing to write and offering them would imply otherwise.
    """
    if not args.capture_logs:
        return LogCaptureConfig(enabled=False, root=None, channels=("workers",))
    root = args.log_root if args.log_root is not None else args.trace_root / "logs"
    return LogCaptureConfig(enabled=True, root=Path(root), channels=("workers",))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    chosen = [
        name
        for name, on in (
            ("--replay", args.replay),
            ("--execute", args.execute),
            ("--executor", bool(args.executor)),
        )
        if on
    ]
    if len(chosen) > 1:
        parser.error(f"{' and '.join(chosen)} are mutually exclusive; pick one")
    if not chosen:
        parser.error(
            "supply one of: --replay (offline, uses recorded answers), "
            "--execute --harness <name-or-path> (real CUGA rollouts), or "
            "--executor module:factory (bring your own agent)"
        )
    if args.execute and not args.harness:
        parser.error(
            "--execute requires --harness: a benchmark run must name the harness "
            "version it executed against, because that label is stamped onto "
            "every trace and is the only way to attribute a result later. There "
            f"is no default. Built-ins: {', '.join(BUILTIN_HARNESS_NAMES)}"
        )
    if args.harness and not args.execute:
        parser.error("--harness only applies to --execute")
    if args.isolation != THREAD_ISOLATION and not args.execute:
        parser.error("--isolation only applies to --execute")

    # Real execution defaults to serial rather than inheriting a default that is
    # known to lose tasks: in-process concurrency is refused (see
    # ConcurrencyUnsupportedError), and process isolation is an explicit choice
    # because it starts one CUGA process per worker.
    if args.max_workers is None:
        args.max_workers = 1 if args.execute else DEFAULT_MAX_WORKERS

    run_dir = args.dataset if args.dataset.is_absolute() else REPO_ROOT / args.dataset
    if not (run_dir / "tasks").is_dir():
        print(f"not a benchmark run directory (no tasks/): {run_dir}")
        return 1

    bench = GaiaBenchmark.from_run_dir(run_dir)
    tasks = _select(bench.load_tasks(), args.limit)

    if args.replay:
        mode = "replay (recorded answers, no model calls)"
    elif args.execute:
        mode = "execute (real CUGA rollouts, traced)"
    else:
        mode = "live (--executor)"

    print(f"dataset      : {run_dir}")
    print(f"run name     : {bench.run_name}")
    print(f"benchmark    : {bench.name}")
    print(f"model        : {bench.config.get('model', '<unknown>')}")
    print(f"graders      : {bench.graders()}")
    print(f"tasks loaded : {len(bench.load_tasks())}  selected: {len(tasks)}")
    print(f"mode         : {mode}")

    coverage = bench.key_coverage()
    missing_dirs = coverage.get("task_dirs_without_record", 0)
    if missing_dirs:
        print(
            f"note         : {missing_dirs} task dir(s) have no result.json and are "
            f"absent from the task list entirely"
        )

    recorder: TraceRecorder | None = None
    pool: CugaProcessPool | None = None
    if args.execute:
        # Everything that can be rejected is rejected before the first billed
        # token: an unresolvable harness, an unsafe worker count, absent model
        # configuration.
        try:
            harness = HarnessVersion.resolve(args.harness)
            preflight(
                harness,
                max_workers=args.max_workers,
                tasks=len(tasks),
                allow_unsafe_concurrency=args.allow_unsafe_concurrency,
                isolation=args.isolation,
            )
        except CugaExecutorError as exc:
            print(f"\ncannot start a real run: {exc}")
            return 2
        recorder = TraceRecorder()
        if args.isolation == PROCESS_ISOLATION:
            worker_root = (
                args.worker_root
                if args.worker_root.is_absolute()
                else REPO_ROOT / args.worker_root
            )
            pool = CugaProcessPool(
                root=worker_root,
                trace_root=args.trace_root,
                task_timeout=args.task_timeout,
                knowledge_seed=(
                    None
                    if args.empty_worker_knowledge
                    else default_knowledge_seed()
                ),
                log_capture=log_capture_from_args(args),
            )
            factory = make_cuga_executor_factory(
                harness, trace_root=args.trace_root, recorder=recorder, worker_pool=pool
            )
        else:
            factory = make_cuga_executor_factory(
                harness, trace_root=args.trace_root, recorder=recorder
            )
        print(f"harness      : {harness.version}  (from {harness.source})")
        print(f"artifacts    : {harness.artifact_summary}")
        print(f"trace root   : {args.trace_root}")
        print(f"isolation    : {args.isolation}")
        if pool is not None:
            print(f"worker root  : {pool.root}")
            print(
                f"worker kb    : "
                + (
                    "EMPTY (not comparable to a serial run)"
                    if pool.knowledge_seed is None
                    else f"seeded from {pool.knowledge_seed}"
                )
            )
    elif args.replay:
        factory = make_replay_factory(bench)
    else:
        factory = load_executor_factory(args.executor)

    def show(execution: TaskExecution) -> None:
        state = "ok     " if execution.ok else ("timeout" if execution.timed_out else "failed ")
        print(
            f"    [{state}] {execution.task.task_id:<40} "
            f"{execution.elapsed_seconds:6.2f}s"
        )

    # A real run is long (~40s/task, 42 tasks) and must be observable, so
    # progress is forced on rather than left to a flag the operator may forget.
    progress = show if (args.progress or args.execute) else None

    print("\nexecuting...")
    try:
        result = run_benchmark(
            bench,
            factory,
            grader=args.grader,
            max_workers=args.max_workers,
            task_timeout_seconds=args.task_timeout,
            tasks=tasks,
            progress=progress,
        )
    finally:
        # Every worker holds an exclusive lock on its knowledge store; leaving one
        # running would block the next run on that store.
        if pool is not None:
            pool.close()
    _print_report(result, verbose=args.verbose, recorder=recorder, tasks=tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
