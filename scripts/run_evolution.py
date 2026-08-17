"""Run the evolution pipeline end to end: rollout -> analyze -> edit -> validate.

One command drives the whole loop through
:mod:`agent_evolve.pipeline`, which owns all component wiring so ``core/`` stays
agent-neutral.

Two modes, one code path
------------------------
``--dry-run``
    The fake stack: ``FakeAdapter`` + ``FakeAnalyzerJudge`` + ``FakeEditor``,
    scored against task contracts. No CUGA process, no model endpoint, no
    network, no dataset. It runs the *same*
    :class:`~agent_evolve.core.orchestrator.SequentialGepaRunner` lifecycle a
    live run does, so it is a rehearsal rather than a separate toy.

live (default)
    Real traced CUGA rollouts against ``--harness``, the real LLM analyzer, and
    the real CUGA editor, scored by ``--grader`` on ``--dataset``.

    ``--max-workers > 1`` requires ``--isolation process``. This is refused, not
    warned about: ``CUGA_FOLDER`` is a single process-global environment
    variable read during ``invoke()``, and two threads that each bound a
    different workspace were observed both reading the second one's while their
    traces still stamped their own ``harness_version``. A threaded parallel run
    therefore looks clean while measuring a harness that never existed.

What the output guarantees
--------------------------
* A pass rate is never printed without its denominator.
* A rollout that produced no answer is excluded from the denominator entirely.
  It is not a wrong answer; scoring it as 0.0 would fabricate an improvement
  delta out of a broken harness.
* Every reported delta carries the measured 16.67 pp noise floor.
* No grading material is printed -- only counts.

What survives the run
---------------------
Nothing, unless ``--export-harness`` is passed. Candidate artifacts live in the
adapter's memory, so a finished run otherwise prints a pass rate and destroys the
harness that earned it: the delta becomes unreproducible and unshippable. With
the flag, every pool candidate plus the champion is written as a harness JSON
file that ``--harness`` accepts directly, so the next run seeds from this one.

Usage::

    # offline rehearsal, no network
    uv run python scripts/run_evolution.py --dry-run --tasks 3 --iterations 1

    # keep what the run produced (champion + every pool candidate)
    uv run python scripts/run_evolution.py --dry-run --tasks 3 --iterations 1 \\
        --export-harness data/harnesses/my-run

    # a real run, serial (the only safe in-process mode)
    uv run python scripts/run_evolution.py \\
        --dataset datasets/gaia/gaia_l1_validation__baseline__20260813_035541 \\
        --grader expected_regex --harness vanilla \\
        --tasks 5 --iterations 1 --max-workers 1 --analyzer-workers 4

    # a real parallel run (process isolation is mandatory)
    uv run python scripts/run_evolution.py \\
        --dataset datasets/gaia/gaia_l1_validation__baseline__20260813_035541 \\
        --grader expected_regex --harness vanilla \\
        --tasks 42 --iterations 3 --max-workers 6 --isolation process \\
        --analyzer-workers 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.benchmarks.cuga_executor import (  # noqa: E402
    BUILTIN_HARNESS_NAMES,
    DEFAULT_TRACE_ROOT,
    PROCESS_ISOLATION,
    THREAD_ISOLATION,
    CugaExecutorError,
)
from agent_evolve.core.evaluation import ScoreTally  # noqa: E402
from agent_evolve.core.run_logging import (  # noqa: E402
    ALL_LOG_CHANNELS,
    LogCaptureConfig,
)
from agent_evolve.pipeline import (  # noqa: E402
    DEFAULT_WORKER_KNOWLEDGE_SEED,
    EvolutionStack,
    build_live_stack,
    build_offline_stack,
    format_delta,
    nothing_accepted_warning,
    nothing_accepted_warning_applies,
)

#: Matches the observed baseline configuration.
DEFAULT_TASK_TIMEOUT = 1200.0
DEFAULT_WORKER_ROOT = Path("data/cuga-workers")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the evolution pipeline: rollout -> analyze -> select -> edit -> "
            "validate -> record."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "run the fake stack offline: no CUGA process, no model endpoint, no "
            "network, no dataset. Exercises the same runner lifecycle."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="benchmark run directory, e.g. datasets/gaia/<run_name> (live runs)",
    )
    parser.add_argument(
        "--grader",
        default=None,
        help=(
            "grader name; always explicit for a live run, never defaulted, "
            "because two graders on the same benchmark disagree"
        ),
    )
    parser.add_argument(
        "--harness",
        default=None,
        help=(
            "REQUIRED for a live run: the harness version rollouts execute "
            f"against. A built-in name ({', '.join(BUILTIN_HARNESS_NAMES)}) or a "
            "path to a harness JSON file. Stamped onto every trace."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=3,
        help="how many tasks to evolve against (default: 3)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="outer iterations, one GEPA attempt each (default: 1)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "rollout concurrency (default: 1). Above 1 requires "
            f"--isolation {PROCESS_ISOLATION}: in-process concurrency is refused "
            "because CUGA_FOLDER is process-global."
        ),
    )
    parser.add_argument(
        "--analyzer-workers",
        type=int,
        default=1,
        help=(
            "analyzer fan-out (default: 1). Threads are safe here: analysis is "
            "pure LLM calls with no CUGA process involved."
        ),
    )
    parser.add_argument(
        "--isolation",
        choices=(THREAD_ISOLATION, PROCESS_ISOLATION),
        default=THREAD_ISOLATION,
        help=(
            f"how rollout workers are isolated. {THREAD_ISOLATION!r} (default) "
            f"runs in this process and is safe only at --max-workers 1. "
            f"{PROCESS_ISOLATION!r} gives each worker its own CUGA subprocess."
        ),
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
        help=f"where causal traces are written (default: {DEFAULT_TRACE_ROOT})",
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
        "--seed-worker-knowledge",
        type=Path,
        default=DEFAULT_WORKER_KNOWLEDGE_SEED,
        help=(
            "seed each worker's knowledge store from this directory. Default is "
            "an EMPTY store: this repo's .cuga/knowledge holds leftover fixtures "
            "unrelated to the benchmark, which would be contamination. Both arms "
            "of any comparison must use the same choice."
        ),
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=DEFAULT_TASK_TIMEOUT,
        help=f"per-task wall-clock budget in seconds (default: {DEFAULT_TASK_TIMEOUT})",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for parent sampling and DPP"
    )
    parser.add_argument(
        "--profile",
        default="research_sequential",
        help="config profile name (default: research_sequential)",
    )
    parser.add_argument(
        "--capture-logs",
        action="store_true",
        help=(
            "capture run logs (worker CUGA stderr, analyzer transcripts, editor "
            "transcripts, pipeline decisions). OFF by default: capture writes "
            "files a measurement run has no use for, and worker stderr is not "
            "cheap. On, nothing about the run is unrecoverable; off, nothing is "
            "written and no directory is created."
        ),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help=(
            "where captured logs are written, one subdirectory per channel. "
            "Defaults to <--trace-root>/logs, because traces and logs describe "
            "the same run and belong next to each other. Ignored without "
            "--capture-logs."
        ),
    )
    parser.add_argument(
        "--log-channels",
        default=",".join(ALL_LOG_CHANNELS),
        help=(
            f"comma-separated channels to capture (default: all -- "
            f"{', '.join(ALL_LOG_CHANNELS)}). Narrow it to skip the expensive "
            f"one: 'workers' is per-rollout CUGA stderr, while an operator "
            f"debugging the editor needs only 'editor,pipeline'."
        ),
    )
    parser.add_argument(
        "--export-harness",
        type=Path,
        default=None,
        help=(
            "persist the evolved harness so the run's result outlives the "
            "process. OFF by default; without it a finished run leaves only a "
            "pass rate on stdout, because candidate artifacts are held in "
            "memory. A path ending in '.json' writes the champion to that one "
            "file; any other path is a directory receiving "
            "'candidate-<id>.json' per pool candidate plus 'champion.json'. "
            "Every file is valid input to --harness, so the next run seeds "
            "directly from this one."
        ),
    )
    parser.add_argument(
        "--allow-unsafe-concurrency",
        action="store_true",
        help=(
            "permit threaded parallel real rollouts anyway. Tasks WILL be lost "
            "to CUGA's knowledge-engine lock and candidates can silently swap "
            "workspaces. Exists only for experiments that knowingly accept "
            "corrupt evidence."
        ),
    )
    return parser


def log_capture_from_args(args: argparse.Namespace) -> LogCaptureConfig:
    """Build the capture config, refusing an unknown channel.

    The refusal comes from :class:`LogCaptureConfig` itself and is deliberate: a
    typo'd channel name that was silently dropped would disable capture for
    exactly the channel the operator asked for, and the run would look captured.
    """
    channels = tuple(
        part.strip() for part in str(args.log_channels).split(",") if part.strip()
    )
    if not args.capture_logs:
        # Not merely disabled: no root either, so nothing downstream can write.
        return LogCaptureConfig(enabled=False, root=None, channels=channels)
    root = args.log_root if args.log_root is not None else args.trace_root / "logs"
    return LogCaptureConfig(enabled=True, root=Path(root), channels=channels)


def _print_header(stack: EvolutionStack) -> None:
    print("=" * 72)
    for line in stack.header_lines:
        print(line)
    print("=" * 72)


def _print_tally(label: str, tally: ScoreTally) -> None:
    print(f"{label:<16}: {tally.summary}")


def _build_live(
    args: argparse.Namespace, log_capture: LogCaptureConfig
) -> EvolutionStack | int:
    """Build the live stack, or return an exit code explaining why not."""
    from agent_evolve.benchmarks.cuga_executor import HarnessVersion
    from agent_evolve.benchmarks.gaia import GaiaBenchmark

    if args.dataset is None:
        print(
            "a live run requires --dataset <benchmark run directory>. Use "
            "--dry-run for the offline fake stack."
        )
        return 2
    if not args.grader:
        print(
            "a live run requires --grader: the grader is never silently chosen, "
            "because two graders on the same benchmark disagree on the same "
            "answers."
        )
        return 2
    if not args.harness:
        print(
            "a live run requires --harness: the harness version is stamped onto "
            "every trace and is the only way to attribute a result later. "
            f"Built-ins: {', '.join(BUILTIN_HARNESS_NAMES)}"
        )
        return 2

    run_dir = args.dataset if args.dataset.is_absolute() else REPO_ROOT / args.dataset
    if not (run_dir / "tasks").is_dir():
        print(f"not a benchmark run directory (no tasks/): {run_dir}")
        return 2

    try:
        benchmark = GaiaBenchmark.from_run_dir(run_dir)
        harness = HarnessVersion.resolve(args.harness)
        return build_live_stack(
            benchmark=benchmark,
            grader=args.grader,
            harness=harness,
            task_limit=args.tasks,
            max_workers=args.max_workers,
            analyzer_workers=args.analyzer_workers,
            isolation=args.isolation,
            worker_root=(
                args.worker_root
                if args.worker_root.is_absolute()
                else REPO_ROOT / args.worker_root
            ),
            knowledge_seed=args.seed_worker_knowledge,
            trace_root=args.trace_root,
            task_timeout_seconds=args.task_timeout,
            seed=args.seed,
            profile=args.profile,
            allow_unsafe_concurrency=args.allow_unsafe_concurrency,
            log_capture=log_capture,
        )
    except CugaExecutorError as exc:
        # Every refusal here is a measured one: an unsafe worker count, an
        # unresolvable harness, absent model configuration.
        print(f"cannot start a live run: {exc}")
        return 2
    except (ValueError, OSError) as exc:
        print(f"cannot start a live run: {type(exc).__name__}: {exc}")
        return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.tasks < 1:
        print("--tasks must be >= 1")
        return 2
    if args.iterations < 1:
        print("--iterations must be >= 1")
        return 2

    try:
        log_capture = log_capture_from_args(args)
    except ValueError as exc:
        print(f"cannot start: {exc}")
        return 2

    if args.dry_run:
        if args.dataset is not None:
            print(
                "note: --dry-run ignores --dataset; the fake stack reads no "
                "dataset and makes no model call"
            )
        stack: EvolutionStack = build_offline_stack(
            task_count=args.tasks,
            analyzer_workers=args.analyzer_workers,
            seed=args.seed,
            profile=args.profile,
            log_capture=log_capture,
        )
    else:
        built = _build_live(args, log_capture)
        if isinstance(built, int):
            return built
        stack = built

    try:
        _print_header(stack)

        print("\nmeasuring the base before any edit...")
        before = stack.measure(stack.base_version, prefix="before")
        _print_tally("base", before)

        print("\nevolving...")
        summaries = stack.run_iterations(args.iterations)
        for summary in summaries:
            print(f"  {summary.line}")

        if nothing_accepted_warning_applies(
            len(stack.tasks), any(s.accepted for s in summaries)
        ):
            # Surfaced loudly because a silently inert run is worse than a loud
            # one. The text names causes an operator can check; it deliberately
            # no longer claims a multi-task arithmetic floor, which was true of
            # the pre-fix weighted_net_gain and is false now.
            print(nothing_accepted_warning(len(stack.tasks)))

        champion = stack.champion_version()
        print(f"\nmeasuring the champion ({champion})...")
        after = stack.measure(champion, prefix="after")
        _print_tally("champion", after)

        if args.export_harness is not None:
            # After measurement, so the exported provenance carries scores the
            # champion actually earned rather than an empty tensor.
            written = stack.export_pool(args.export_harness)
            print(f"\nexported {len(written)} harness file(s):")
            for path in written:
                print(f"  {path}")
            print(
                f"note   : re-run against the champion with --harness "
                f"{written[-1]}"
            )

        print()
        for line in format_delta(before, after):
            print(line)

        if champion == stack.base_version:
            print(
                "note   : the champion is the base, so the two numbers are two "
                "measurements of the same version -- the delta is pure run-to-run "
                "variance, not evolution"
            )
        print(
            f"note   : {stack.candidate_count()} candidate(s) in the pool; with "
            f"no RHO seeder there is one lineage, so cross-candidate entropy and "
            f"DPP diversity contributed nothing to selection"
        )
    finally:
        stack.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
