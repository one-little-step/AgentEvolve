"""Decoupled benchmark abstraction for AgentEvolve.

The evolution core consumes benchmarks only through
:class:`~agent_evolve.benchmarks.base.Benchmark`. No benchmark-specific JSON
schema, success measure, or grader name appears in the core.

Public surface::

    from agent_evolve.benchmarks import GaiaBenchmark, compute_run_statistics

    bench = GaiaBenchmark.from_run_dir("datasets/gaia/<run_name>")
    tasks = bench.load_tasks()                      # safe to expose
    stats = compute_run_statistics(bench, bench.observations())
    outcome = bench.score(tasks[0].task_id, "answer", grader="expected_regex")

Whole-benchmark execution (baseline runs and evolution rollouts alike) goes
through :func:`~agent_evolve.benchmarks.runner.run_benchmark`, which fans tasks
out with bounded concurrency and reports a pass rate that always carries its own
denominator::

    result = run_benchmark(bench, my_agent_factory, grader="expected_regex",
                           max_workers=10, task_timeout_seconds=1200)
    print(result.summary)

Ground-truth material never reaches a task-facing object: see
:data:`~agent_evolve.benchmarks.base.GRADING_KEY_DENYLIST`.
"""

from __future__ import annotations

from .base import (
    GRADING_KEY_DENYLIST,
    Benchmark,
    BenchmarkError,
    BenchmarkGrading,
    BenchmarkTask,
    GraderDelta,
    GraderStats,
    GradingUnavailableError,
    LeakageError,
    RunComparison,
    RunObservations,
    RunStatistics,
    TaskOutcome,
    UnknownGraderError,
    UnknownTaskError,
    compare_runs,
    compute_run_statistics,
    outcomes_disagree,
)
from .gaia import (
    GAIA_RESULT_KEYS,
    GRADER_EXPECTED_REGEX,
    GRADER_RECORDED_LLM_VERDICT,
    GaiaBenchmark,
    GaiaGrading,
    discover_gaia_runs,
)
from .runner import (
    BenchmarkRunResult,
    TaskExecution,
    TaskExecutor,
    TaskExecutorFactory,
    run_benchmark,
)

__all__ = [
    "GAIA_RESULT_KEYS",
    "GRADER_EXPECTED_REGEX",
    "GRADER_RECORDED_LLM_VERDICT",
    "GRADING_KEY_DENYLIST",
    "Benchmark",
    "BenchmarkError",
    "BenchmarkGrading",
    "BenchmarkRunResult",
    "BenchmarkTask",
    "GaiaBenchmark",
    "GaiaGrading",
    "GraderDelta",
    "GraderStats",
    "GradingUnavailableError",
    "LeakageError",
    "RunComparison",
    "RunObservations",
    "RunStatistics",
    "TaskExecution",
    "TaskExecutor",
    "TaskExecutorFactory",
    "TaskOutcome",
    "UnknownGraderError",
    "UnknownTaskError",
    "compare_runs",
    "compute_run_statistics",
    "discover_gaia_runs",
    "outcomes_disagree",
    "run_benchmark",
]
