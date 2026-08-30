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
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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
from agent_evolve.cuga_wrapper import ALLOW_RESPONSE_CACHE_ENV  # noqa: E402
from agent_evolve.core.run_logging import (  # noqa: E402
    ALL_LOG_CHANNELS,
    LogCaptureConfig,
)
from agent_evolve.pipeline import (  # noqa: E402
    DEFAULT_WORKER_KNOWLEDGE_SEED,
    EvolutionStack,
    IterationSummary,
    build_live_stack,
    build_offline_stack,
    build_rho_hooks,
    format_delta,
    nothing_accepted_warning,
    nothing_accepted_warning_applies,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent_evolve.core.rho.rounds import RoundConfig

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
        "--max-rollouts-per-worker",
        type=int,
        default=None,
        help=(
            "replace a CUGA worker process after this many rollouts (default: "
            "25). Bounds the dominant memory-growth term: a worker reuses one "
            "CugaWrapper and the SDK's per-invocation state is never released "
            "between calls, so an unbounded worker grows monotonically for the "
            "whole run. Lower it if RAM is tight; raise it to amortise worker "
            "startup over more rollouts."
        ),
    )
    parser.add_argument(
        "--cleanup-on-exit",
        action="store_true",
        help=(
            "at end of run, kill orphaned Playwright browsers and prune "
            "cuga_workspace/ directories older than one hour. Off by default "
            "because it terminates processes and deletes directories; the run "
            "always REPORTS what it would reclaim either way."
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
    parser.add_argument(
        "--allow-response-cache",
        action="store_true",
        help=(
            "permit the upstream gateway's response cache. OFF by default "
            "because a cached repeat returns ONE observation N times: verified "
            "live, four identical requests shared a single response id and "
            "identical text, so a G-group or R-repeat measures zero variance "
            "while every counter still reports N rollouts. Enable only to "
            "reproduce a pre-fix run or to cut spend on a run whose rollout "
            "diversity does not matter."
        ),
    )
    _add_budget_arguments(parser)
    _add_tuning_arguments(parser)
    _add_rho_arguments(parser)
    return parser


def _add_budget_arguments(parser: argparse.ArgumentParser) -> None:
    """Add spend caps.

    Every :class:`BudgetLimits` field defaults to ``None`` (unlimited), and
    nothing set them before, so any run was an uncapped run: a genetic loop
    could issue rollouts and editor calls until the dataset ran out. These flags
    are the only way to bound a run's cost. Omitted flags stay ``None``, so an
    existing invocation behaves exactly as before.
    """
    group = parser.add_argument_group(
        "budgets (spend caps; every default is UNLIMITED)",
        "A cap is checked before the work is issued, so it refuses rather than "
        "truncating a half-finished attempt. Unset means no limit.",
    )
    group.add_argument(
        "--max-rollouts",
        type=int,
        default=None,
        help="hard cap on total rollouts across the whole run (default: unlimited)",
    )
    group.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="hard cap on edit attempts across the whole run (default: unlimited)",
    )
    group.add_argument(
        "--max-accepted-edits",
        type=int,
        default=None,
        help="stop accepting edits after this many (default: unlimited)",
    )
    group.add_argument(
        "--max-editor-calls",
        type=int,
        default=None,
        help="hard cap on editor-agent invocations (default: unlimited)",
    )
    group.add_argument(
        "--max-judge-verdicts",
        type=int,
        default=None,
        help="hard cap on analyzer/judge calls (default: unlimited)",
    )
    group.add_argument(
        "--max-model-tokens",
        type=int,
        default=None,
        help="hard cap on model tokens across the run (default: unlimited)",
    )
    group.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help="hard cap on run wall-clock seconds (default: unlimited)",
    )
    group.add_argument(
        "--max-pool-candidates",
        type=int,
        default=None,
        help=(
            "cap the persistent pool size. NOTE: RHO retains all N candidates by "
            "design; setting this can refuse a retention the design requires "
            "(default: unlimited)"
        ),
    )
    group.add_argument(
        "--max-history-records",
        type=int,
        default=None,
        help="cap edit-memory history records supplied to the editor (default: unlimited)",
    )
    group.add_argument(
        "--max-rag-context-tokens",
        type=int,
        default=None,
        help="cap retrieved-context tokens handed to the editor (default: unlimited)",
    )
    group.add_argument(
        "--edit-max-retries",
        type=int,
        default=3,
        help="retries per edit attempt before it is abandoned (default: 3)",
    )


def _add_tuning_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the algorithm knobs that ``ResolvedConfig`` already accepted.

    These were reachable only from Python. Each one maps 1:1 onto a
    ``resolve_profile`` override, and omitting a flag leaves the profile default
    untouched, so nothing changes for an existing invocation.
    """
    group = parser.add_argument_group(
        "algorithm tuning (selection, entropy, clustering, champion weights)",
        "Every flag here overrides one ResolvedConfig field. Unset means the "
        "--profile default stands.",
    )
    # --- DPP coreset / issue selection ---
    group.add_argument("--dpp-max-items", type=int, default=None,
                       help="max items considered by the DPP (default: 100)")
    group.add_argument("--dpp-theta", type=float, default=None,
                       help="DPP quality/diversity tradeoff in [0,1] (default: 0.7)")
    group.add_argument("--dpp-score-floor", type=float, default=None,
                       help="minimum normalized quality in [0,1] (default: 0.1)")
    group.add_argument("--dpp-min-gain", type=float, default=None,
                       help="stop greedy MAP below this marginal gain (default: 1e-12)")
    # --- entropy-guided selection ---
    group.add_argument("--entropy-refresh-mode",
                       choices=("outer_iteration", "accepted_edits", "pool_growth"),
                       default=None,
                       help="when entropy is recomputed (default: outer_iteration)")
    group.add_argument("--entropy-score-floor", type=float, default=None,
                       help="minimum score for an entropy cell in [0,1]")
    group.add_argument("--entropy-recombination-score-threshold", type=float,
                       default=None, help="score above which recombination is allowed")
    group.add_argument("--entropy-frontier-weight", type=float, default=None,
                       help="weight on frontier novelty in [0,1]")
    group.add_argument("--entropy-min-comparable-candidates", type=int, default=None,
                       help=(
                           "candidates a cell needs before entropy may drive "
                           "selection (default: 3). With --rho-candidates below "
                           "this, cross-candidate entropy stays inert"
                       ))
    group.add_argument("--entropy-min-rollouts-per-candidate", type=int, default=None,
                       help=(
                           "rollouts per candidate before its cell counts as "
                           "comparable (default: 2; this is the R that "
                           "--rho-candidate-rollouts must meet)"
                       ))
    # --- mechanism clustering ---
    group.add_argument("--cluster-similarity-threshold", type=float, default=None,
                       help="cosine threshold for one mechanism cluster in [0,1]")
    group.add_argument("--max-clusters-per-task", type=int, default=None,
                       help="cap mechanism clusters retained per task")
    # --- validation probes ---
    group.add_argument("--generalization-probe-mode",
                       choices=("deferred", "enabled"), default=None,
                       help=(
                           "cluster-level generalization probes. 'deferred' "
                           "(default) records them without spending rollouts"
                       ))
    group.add_argument("--probe-budget-fraction", type=float, default=None,
                       help="fraction of rollouts reserved for probes (default: 0.15)")
    # --- champion aggregate weights (reported, not used for ranking) ---
    # SV-2: ranking is a pairwise comparison restricted to the cells both entries
    # measured, so none of these four weights can change which candidate wins. They
    # still parameterise the aggregate recorded in the manifest, which is why they
    # remain configurable -- but the help text must not imply they select anything.
    #
    # gamma and delta additionally weight terms that were never implemented
    # (`stability = 1.0`, `regression_risk = 0.0`; SV-5). Their old help strings
    # named them "worst-case" and "novelty", which described a specification rather
    # than the code.
    group.add_argument("--champion-alpha", type=float, default=None,
                       help="reported aggregate weight: mean score (default: 0.55; "
                            "does not affect selection)")
    group.add_argument("--champion-beta", type=float, default=None,
                       help="reported aggregate weight: coverage (default: 0.20; "
                            "does not affect selection -- use "
                            "--champion-min-coverage-fraction to act on coverage)")
    group.add_argument("--champion-gamma", type=float, default=None,
                       help="reported aggregate weight: reserved, term is currently "
                            "the constant 1.0 (default: 0.15; does not affect "
                            "selection)")
    group.add_argument("--champion-delta", type=float, default=None,
                       help="reported aggregate weight: reserved, term is currently "
                            "the constant 0.0 (default: 0.10; does not affect "
                            "selection)")
    group.add_argument("--champion-min-coverage-fraction", type=float, default=None,
                       help=(
                           "minimum task coverage before a candidate may be "
                           "champion (default: 0.0). Raise this to stop a "
                           "candidate winning on one lucky task. Unlike the "
                           "weights above, this is enforced"
                       ))
    # Ablation switch, not a feature toggle. Absent (the default) the RHO paper's
    # S_j > 0 acceptance gate is ACTIVE; passing this disables it so a run can
    # measure what the pairwise judge contributes. `store_true` with
    # default=None keeps it out of `overrides` unless explicitly passed.
    group.add_argument("--experimental-candidate-promotion",
                       action="store_true", default=None,
                       help=(
                           "ABLATION: disable the RHO pairwise acceptance gate "
                           "(S_j > 0) and rank candidates by the grader "
                           "aggregate alone. Default (absent) is paper "
                           "behaviour: a candidate may only become champion if "
                           "the symmetric preference judge prefers it to the "
                           "incumbent"
                       ))
    _add_ablation_arguments(parser)


def _add_ablation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add per-feature ablation switches.

    ``--profile`` sets these five gates as a bundle; an ablation study needs to
    move exactly one. Each flag is a tri-state: unset leaves the profile's value
    alone, so these compose with ``--profile`` instead of replacing it.
    """
    group = parser.add_argument_group(
        "ablations (override individual feature gates set by --profile)",
        "Unset leaves the profile default. Use these to isolate one mechanism.",
    )
    for flag, dest, what in (
        ("causal-blame", "use_causal_blame",
         "causal blame graphs for choosing the artifact to edit"),
        ("edit-memory", "use_edit_memory",
         "edit memory (past attempts shown to the editor)"),
        ("focused-validation", "use_focused_validation",
         "focused validation (origin cases, worked sets, regression probes)"),
        ("entropy-selection", "use_entropy_selection",
         "entropy-guided parent selection"),
        ("parallel-execution", "parallel_execution",
         "parallel batch execution (snapshots, write leases, coordinator)"),
        ("positivity-judge", "use_positivity_judge",
         "Judge-2 positivity analysis (strengths evidence for complementary parents)"),
    ):
        group.add_argument(
            f"--enable-{flag}", dest=dest, action="store_true", default=None,
            help=f"force ON: {what}",
        )
        group.add_argument(
            f"--disable-{flag}", dest=dest, action="store_false", default=None,
            help=f"force OFF: {what}",
        )


def _add_rho_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the RHO stage flags.

    Grouped at the end rather than interleaved with the existing concurrency
    flags: ``--max-workers``, ``--analyzer-workers`` and ``--isolation`` describe
    one decision together, and splitting them made ``--help`` read as if
    ``--isolation`` were an RHO option. Every flag here is inert at the default
    ``--mode genetic``, so an existing invocation is unchanged.
    """
    group = parser.add_argument_group(
        "RHO stage",
        "retrospective harness optimization; inert unless --mode selects it",
    )
    group.add_argument(
        "--mode",
        choices=("rho", "genetic", "rho-genetic"),
        default="genetic",
        help=(
            "which phases run per outer iteration. 'genetic' (default) is the "
            "existing mutation/crossover loop, unchanged. 'rho' runs the "
            "retrospective harness-optimization round. 'rho-genetic' alternates "
            "[RHO round -> genetic iterations]."
        ),
    )
    group.add_argument(
        "--rho-rounds",
        type=int,
        default=1,
        help="how many RHO rounds to run (default: 1)",
    )
    group.add_argument(
        "--rho-history",
        type=Path,
        default=None,
        help=(
            "trace root holding the historical corpus. Omitted means COLD START: "
            "difficulty judging is skipped and the coreset is chosen without "
            "difficulty weighting, so the run proves plumbing, not the method."
        ),
    )
    group.add_argument(
        "--rho-coreset-size",
        type=int,
        default=10,
        help="k: how many historical tasks to diagnose and re-solve (default: 10)",
    )
    group.add_argument(
        "--rho-group-rollouts",
        type=int,
        default=3,
        help="G: baseline rollouts per coreset task (default: 3)",
    )
    group.add_argument(
        "--rho-candidates",
        type=int,
        default=3,
        help=(
            "N: independent candidate proposals per round (default: 3). Each is "
            "its own workspace-agent invocation; ALL survivors are retained."
        ),
    )
    group.add_argument(
        "--rho-candidate-rollouts",
        type=int,
        default=2,
        help=(
            "R: rollouts per candidate per coreset task (default: 2). 2 is the "
            "minimum that clears the cross-candidate entropy evidence floor "
            "(min_rollouts_per_candidate=2), so low-evidence cells are never "
            "silently skipped. 1 halves cost but leaves every candidate mean "
            "resting on a single stochastic rollout."
        ),
    )
    group.add_argument(
        "--rho-selector",
        choices=("dpp", "difficulty_rank", "random"),
        default="dpp",
        help="coreset selection strategy (default: dpp)",
    )
    group.add_argument(
        "--rho-group-workers",
        type=int,
        default=4,
        help="concurrently admitted task groups (default: 4)",
    )
    group.add_argument(
        "--rho-rollout-workers",
        type=int,
        default=3,
        help="concurrent rollouts within one group (default: 3)",
    )
    group.add_argument(
        "--rho-proposal-temperature",
        type=float,
        default=None,
        help=(
            "ABLATION ONLY. Unset by default: candidate diversity comes from N "
            "independent agent invocations, not sampling. 0.0 is rejected by the "
            "endpoint and is refused here."
        ),
    )
    group.add_argument(
        "--rho-summary-cache",
        type=Path,
        default=None,
        help="cache dir for trajectory comprehension, keyed by trace content hash",
    )
    group.add_argument(
        "--rho-difficulty-cache",
        type=Path,
        default=None,
        help="cache dir for difficulty/fingerprint verdicts",
    )
    group.add_argument(
        "--rho-embedding-cache",
        type=Path,
        default=None,
        help="cache dir for fingerprint embeddings",
    )
    group.add_argument(
        "--genetic-iterations-per-round",
        type=int,
        default=1,
        help="genetic iterations after each RHO round in rho-genetic (default: 1)",
    )


def resolve_rho_config(args: argparse.Namespace) -> "RoundConfig":
    """Validate RHO configuration before anything expensive is constructed.

    Every refusal here is credential-independent on purpose. A run configured
    with an impossible concurrency, or with threads against a process-global
    ``CUGA_FOLDER``, is wrong whether or not a model is reachable, and reporting
    "no model configured" first would send an operator to fix the wrong thing --
    after paying for a CUGA wrapper. Same ordering rationale as
    :func:`agent_evolve.pipeline.require_safe_rollout_concurrency`.

    Nothing is ever clamped. A ``--max-workers`` larger than the two-level
    structure can produce is a configuration error: silently lowering it would
    make the run measure a concurrency the operator never asked for.
    """
    # Imported inside the function so this module keeps importing while the RHO
    # round machinery is absent, and so --help costs nothing.
    from agent_evolve.core.rho.scheduler import ConcurrencyPlan

    if (
        args.rho_proposal_temperature is not None
        and args.rho_proposal_temperature == 0.0
    ):
        raise SystemExit(
            "--rho-proposal-temperature 0.0 is rejected by the endpoint "
            "('temperature does not support 0.0'); omit the flag instead"
        )

    # Checked before the plan is built: an unsafe isolation choice is unsafe at
    # every worker count that exceeds 1, whatever the two-level split is.
    if (
        not args.dry_run
        and not args.allow_unsafe_concurrency
        and args.isolation != PROCESS_ISOLATION
        and max(args.max_workers, args.rho_rollout_workers) > 1
    ):
        raise SystemExit(
            f"RHO rollout concurrency requires --isolation {PROCESS_ISOLATION}. "
            f"CUGA_FOLDER is a process-global environment variable read during "
            f"invoke(): two threads binding different candidate workspaces were "
            f"observed both reading the second one's, while each trace still "
            f"stamped its own harness_version -- the run would look clean while "
            f"measuring a harness that never existed. Use --isolation "
            f"{PROCESS_ISOLATION}, or --max-workers 1 --rho-rollout-workers 1."
        )

    try:
        plan = ConcurrencyPlan.validated(
            group_workers=args.rho_group_workers,
            rollout_workers=args.rho_rollout_workers,
            global_cap=args.max_workers,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid RHO concurrency: {exc}") from exc

    from agent_evolve.core.rho.rounds import RoundConfig

    try:
        return RoundConfig(
            mode=args.mode,
            rounds=args.rho_rounds,
            coreset_size=args.rho_coreset_size,
            group_rollouts=args.rho_group_rollouts,
            candidates=args.rho_candidates,
            candidate_rollouts=args.rho_candidate_rollouts,
            selector=args.rho_selector,
            genetic_iterations_per_round=args.genetic_iterations_per_round,
            concurrency=plan,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid RHO configuration: {exc}") from exc


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
    # Sampling provenance: a cached rollout is one observation reported N times,
    # so a reader comparing two runs must be able to see which regime produced
    # the numbers without reconstructing the command line.
    cached = os.getenv(ALLOW_RESPONSE_CACHE_ENV)
    if cached and cached.strip().lower() in {"1", "true", "yes", "on"}:
        print(
            "response cache : ALLOWED -- repeated identical prompts may return "
            "one cached observation N times; rollout variance is not measured"
        )
    else:
        print("response cache : disabled (each rollout is an independent sample)")
    print("=" * 72)


def _print_tally(label: str, tally: ScoreTally) -> None:
    print(f"{label:<16}: {tally.summary}")
    # S4-10: an unscorable must always say why. Silent unscorables are how an
    # unwired judge or a dead endpoint masquerades as "no data".
    for task_id in tally.unscorable_task_ids:
        reason = tally.unscorable_reasons.get(task_id) or "reason not recorded"
        print(f"{'':<16}  unscorable {task_id}: {reason}")


def resolve_config_overrides(args: argparse.Namespace) -> dict:
    """Turn the budget / tuning / ablation flags into ``resolve_profile`` kwargs.

    Only flags the user actually passed appear in the result. An unset flag is
    ``None`` and is omitted, so the ``--profile`` default stands and an existing
    invocation resolves to exactly the config it did before.

    ``--edit-max-retries`` is the one budget field with a real default (3), so it
    is always sent; sending ``None`` there would violate its type.
    """
    from agent_evolve.core.config import BudgetLimits, FeatureGates

    overrides: dict = {}

    scalar_fields = (
        "dpp_max_items", "dpp_theta", "dpp_score_floor", "dpp_min_gain",
        "entropy_refresh_mode", "entropy_score_floor",
        "entropy_recombination_score_threshold", "entropy_frontier_weight",
        "entropy_min_comparable_candidates", "entropy_min_rollouts_per_candidate",
        "cluster_similarity_threshold", "max_clusters_per_task",
        "generalization_probe_mode", "probe_budget_fraction",
        "champion_alpha", "champion_beta", "champion_gamma", "champion_delta",
        "champion_min_coverage_fraction",
        "experimental_candidate_promotion",
    )
    for name in scalar_fields:
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value

    budget_fields = (
        "max_attempts", "max_accepted_edits", "max_model_tokens", "max_rollouts",
        "max_judge_verdicts", "max_editor_calls", "max_wall_seconds",
        "max_pool_candidates", "max_history_records", "max_rag_context_tokens",
    )
    budget_kwargs = {
        name: getattr(args, name)
        for name in budget_fields
        if getattr(args, name, None) is not None
    }
    retries = getattr(args, "edit_max_retries", None)
    # Only send retries when the user actually moved it. Sending the default
    # would emit a ``budgets`` override on a run that passed no budget flag,
    # which makes "no flags" indistinguishable from "explicitly default" in the
    # manifest.
    if retries is not None and retries != 3:
        budget_kwargs["edit_max_retries"] = retries
    if budget_kwargs:
        overrides["budgets"] = BudgetLimits(**budget_kwargs)

    # Ablations are tri-state: only a gate the user moved is overridden, so these
    # compose with --profile rather than replacing the whole bundle.
    gate_fields = (
        "use_causal_blame", "use_edit_memory", "use_focused_validation",
        "use_entropy_selection", "parallel_execution", "use_positivity_judge",
    )
    moved = {
        name: getattr(args, name)
        for name in gate_fields
        if getattr(args, name, None) is not None
    }
    if moved:
        from agent_evolve.core.config import PROFILE_GATES

        base = PROFILE_GATES.get(args.profile)
        if base is None:
            raise SystemExit(f"error: unknown profile {args.profile!r}")
        overrides["features"] = FeatureGates(**{**base, **moved})

    return overrides


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
            max_rollouts_per_worker=args.max_rollouts_per_worker,
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
            config_overrides=resolve_config_overrides(args),
        )
    except CugaExecutorError as exc:
        # Every refusal here is a measured one: an unsafe worker count, an
        # unresolvable harness, absent model configuration.
        print(f"cannot start a live run: {exc}")
        return 2
    except (ValueError, OSError) as exc:
        print(f"cannot start a live run: {type(exc).__name__}: {exc}")
        return 2


def _run_rho_preflight(args: argparse.Namespace) -> "RoundConfig | int":
    """Resolve and report the RHO configuration, or return an exit code.

    Every refusal here is credential-independent and happens before the dataset,
    grader, harness and model checks: an impossible concurrency is wrong whatever
    the credentials are.

    Returns the validated :class:`RoundConfig` on success. It is *returned*
    rather than reported-and-discarded because the caller now executes the round
    with it -- silently running the genetic loop under ``--mode rho`` would
    attribute genetic results to RHO, which is the one failure this reporting
    exists to prevent.
    """
    try:
        config = resolve_rho_config(args)
    except SystemExit as exc:
        print(f"cannot start: {exc}")
        return 2
    except ImportError as exc:
        # The round machinery is absent; the invariant above was still checked.
        print(
            f"--mode {args.mode} validated its concurrency but the RHO round "
            f"machinery is unavailable ({exc}); no RHO phase can run"
        )
        return 2

    print(
        f"RHO     : mode={config.mode} rounds={config.rounds} "
        f"k={config.coreset_size} G={config.group_rollouts} "
        f"N={config.candidates} R={config.candidate_rollouts} "
        f"selector={config.selector}"
    )
    print(
        f"RHO cost: {config.rollouts_per_round} rollout(s) per round under a "
        f"global cap of {config.concurrency.global_cap} "
        f"({config.concurrency.group_workers} groups x "
        f"{config.concurrency.rollout_workers} rollouts)"
    )
    if args.rho_history is None:
        print(
            "RHO      : COLD START (no --rho-history): no historical trace "
            "corpus was supplied, so difficulty judging and coreset selection "
            "have nothing to rank and the RHO phases are skipped. This proves "
            "plumbing, not the method."
        )
    return config


def _offline_preference_judge() -> object:
    """The deterministic preference judge used by ``--dry-run``.

    Imported at call time so a live run never loads the examples package. Shares
    the offline RHO judge, so a dry run exercises the same acceptance-gate and
    retirement logic a live run does -- just with a verdict that is a pure
    function of the traces rather than a model call.
    """
    from examples.fake_rho_components import OfflinePreferenceJudge

    return OfflinePreferenceJudge()


def _rho_components_for(args: argparse.Namespace) -> dict[str, object]:
    """The five RHO components this run should use.

    ``--dry-run`` gets deterministic offline ones. That is not a convenience: the
    flag documents "no CUGA process, no model endpoint, no network", and the real
    comprehender and difficulty judge call ``litellm`` while the real diagnoser,
    optimizer and preference judge each construct a CUGA agent. A dry run that
    built them would make a real network call while claiming not to -- and would
    then report every failed call as an unobserved result, so the round would
    degrade to "summaries unavailable" and read like a data problem rather than a
    wiring one.

    A live run gets ``{}``, so :func:`build_rho_hooks` constructs the real
    adapters itself. Returning explicit fakes for a live run would be the same
    mistake in the other direction: a fabricated round reported as a real one.
    """
    if not args.dry_run:
        return {}
    from examples.fake_rho_components import offline_rho_components

    return offline_rho_components()


def _run_rho_rounds(
    stack: EvolutionStack, args: argparse.Namespace, config: "RoundConfig"
) -> None:
    """Execute the RHO rounds against ``stack`` and report each one.

    Nothing is inferred from a passing configuration: every line printed here
    describes work that actually happened, and a phase that produced nothing
    says so in ``notes``.
    """
    from agent_evolve.core.entropy import EntropyTracker
    from agent_evolve.core.rho.rounds import run_rounds

    history_root = args.rho_history
    if history_root is not None and not Path(history_root).is_absolute():
        history_root = REPO_ROOT / history_root

    hooks = build_rho_hooks(
        stack,
        history_root=history_root,
        summary_cache_root=args.rho_summary_cache,
        difficulty_cache_root=args.rho_difficulty_cache,
        embedding_cache_root=args.rho_embedding_cache,
        proposal_temperature=args.rho_proposal_temperature,
        **_rho_components_for(args),  # type: ignore[arg-type]
    )
    # A tracker of its own rather than the runner's: the RHO cells are keyed by
    # ``rho_cluster_id``, and mixing them into the genetic tracker's fixed
    # cluster would make two different mechanisms share one cell.
    tracker = EntropyTracker()

    print(f"\nrunning {config.rounds} RHO round(s)...")
    summaries = run_rounds(config, hooks, tracker=tracker)
    for summary in summaries:
        print(f"  {summary.line()}")
        print(
            f"    rollouts={summary.rollouts_spent} "
            f"failures={summary.rollout_failures} "
            f"diagnoses_observed={summary.diagnoses_observed} "
            f"preferences={summary.preferences_available} available / "
            f"{summary.preferences_unavailable} unavailable"
        )
        if summary.preferences_available:
            print(f"    mean preference={summary.preference_mean:+.3f}")
            # Per-candidate S_j and its gate consequence. The aggregate mean
            # cannot show this: a positive mean can hide individually gated
            # candidates, which is exactly what a reader needs to see.
            for item in summary.evidence:
                if not item.decided:
                    print(
                        f"      cand[{item.candidate_index}] S_j=n/a "
                        f"(no verdict) -> INELIGIBLE"
                    )
                    continue
                verdict = "eligible" if item.mean_preference > 0.0 else "GATED"
                print(
                    f"      cand[{item.candidate_index}] "
                    f"S_j={item.mean_preference:+.3f} -> {verdict}"
                )
        if summary.collapsed:
            # Never silent: a collapse to one candidate means the pairwise judge
            # compared a harness against itself.
            print(
                f"    warning: candidates COLLAPSED to "
                f"{summary.candidates_distinct} of "
                f"{summary.candidates_requested} requested; a single surviving "
                f"candidate is compared against itself"
            )
        for index, reason in summary.discarded:
            print(f"    discarded candidate {index}: {reason}")
        for note in summary.notes:
            print(f"    note: {note}")
        if summary.genetic_iterations:
            print(
                f"    genetic: {summary.genetic_iterations} iteration(s) over "
                f"the {len(summary.coreset_ids)} coreset task(s)"
            )
    hits = summaries[-1].cache_hits if summaries else {}
    if hits:
        print(
            "  cache hits: "
            + " ".join(f"{name}={count}" for name, count in sorted(hits.items()))
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Set before any CUGA import and before the process pool forks: rollout
    # workers are separate processes that build their own wrapper, so the
    # environment is the only channel that reaches them.
    if args.allow_response_cache:
        os.environ[ALLOW_RESPONSE_CACHE_ENV] = "1"
    else:
        os.environ.pop(ALLOW_RESPONSE_CACHE_ENV, None)

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

    rho_config: "RoundConfig | None" = None
    if args.mode != "genetic":
        # Before the dataset, grader, harness and model checks on purpose: an
        # impossible concurrency or an unsafe isolation choice is wrong whatever
        # the credentials are, and reporting a missing dataset first would send
        # an operator to fix the wrong thing.
        resolved = _run_rho_preflight(args)
        if isinstance(resolved, int):
            return resolved
        rho_config = resolved

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
            config_overrides=resolve_config_overrides(args),
            # The deterministic judge, so a dry run rehearses generational
            # retirement and pairwise resolution rather than skipping them. A
            # judge-free dry run would report "base won" for reasons that have
            # nothing to do with the harness under test.
            preference_judge=_offline_preference_judge(),
        )
    else:
        built = _build_live(args, log_capture)
        if isinstance(built, int):
            return built
        stack = built

    try:
        # Set before the header so the "no RHO seeder" caveat matches the run
        # about to happen rather than the run the builder assumed.
        stack.mode = args.mode
        _print_header(stack)

        print("\nmeasuring the base before any edit...")
        before = stack.measure(stack.base_version, prefix="before")
        _print_tally("base", before)

        summaries: tuple[IterationSummary, ...] = ()
        if rho_config is not None:
            # The RHO round owns the genetic phase in ``rho-genetic`` -- it hands
            # it the coreset tasks only, because after a RHO round the
            # (task, mechanism) cells exist only there. Running the outer genetic
            # loop as well would spend a second, differently-scoped budget and
            # attribute both to one number.
            _run_rho_rounds(stack, args, rho_config)
        else:
            print("\nevolving...")
            summaries = stack.run_iterations(args.iterations)
            for summary in summaries:
                print(f"  {summary.line}")

        if summaries and nothing_accepted_warning_applies(
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
            f"note   : {stack.candidate_count()} candidate(s) in the pool"
            + (
                ""
                if rho_config is not None
                else "; with no RHO seeder there is one lineage, so "
                "cross-candidate entropy and DPP diversity contributed nothing "
                "to selection"
            )
        )
    finally:
        stack.close()
        # Out-of-heap leaks: orphaned Playwright browsers and per-invocation
        # workspace scratch. Neither is a Python object, so closing agents and
        # recycling workers cannot reclaim them.
        #
        # Always REPORTS, only acts under --cleanup-on-exit: this kills processes
        # and deletes directories, which must be an explicit choice. Best-effort
        # throughout -- a cleanup failure must never turn a completed multi-hour
        # measurement into a non-zero exit.
        try:
            from agent_evolve.benchmarks.cleanup import run_cleanup

            report = run_cleanup(dry_run=not args.cleanup_on_exit)
            if report.found_browsers or report.removed_dirs:
                verb = "reclaimed" if args.cleanup_on_exit else "reclaimable"
                print(
                    f"  cleanup ({verb}): "
                    f"{len(report.found_browsers)} browser process(es), "
                    f"{report.removed_dirs} workspace dir(s), "
                    f"{report.reclaimed_bytes / 1e9:.2f} GB"
                )
                if not args.cleanup_on_exit:
                    print("    re-run with --cleanup-on-exit to reclaim")
            for problem in report.errors:
                print(f"    cleanup warning: {problem}")
        except Exception as exc:  # noqa: BLE001 - cleanup must not fail a run
            print(f"  cleanup skipped: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
