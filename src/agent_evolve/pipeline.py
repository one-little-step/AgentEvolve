"""Composition root: assembles a runnable evolution pipeline from real parts.

Nothing here is new capability. Every component is built and tested elsewhere;
this module is the one place allowed to know about all of them at once, so
``core/`` stays agent-neutral (it may not import ``cuga`` or
``agent_evolve.adapters``) while a real run still gets real components.

Two stacks, one shape
---------------------
:func:`build_offline_stack`
    ``FakeAdapter`` + ``FakeAnalyzerJudge`` + ``FakeEditor`` + a task-contract
    scorer. No CUGA process, no model endpoint, no network. This is what
    ``--dry-run`` and the test suite use, and it exercises the same
    :class:`~agent_evolve.core.orchestrator.SequentialGepaRunner` code path a
    live run does.

:func:`build_live_stack`
    ``CugaAdapter`` + ``CugaTrajectoryAnalyzer`` + ``CugaEditorAgent`` + a
    benchmark-driven scorer, with rollouts executed through
    :func:`~agent_evolve.benchmarks.runner.run_benchmark`.

Both return an :class:`EvolutionStack` with the same API, so the CLI has one
code path and a dry run is a genuine rehearsal of a live one rather than a
separate toy.

Constraints this module enforces, all of them measured
-----------------------------------------------------
* **Real parallel rollouts require process isolation.** ``CUGA_FOLDER`` is a
  single process-global environment variable read during ``invoke()``. Two
  threads that each bound a different workspace were observed both reading the
  second one's, while each trace still stamped its own ``harness_version`` --
  so a threaded run looks clean while measuring a harness that never existed.
  :class:`CugaRolloutRunner` therefore refuses ``max_workers > 1`` without a
  worker pool. Analyzer fan-out is exempt: it is pure LLM calls with no CUGA
  process involved.
* **A failed rollout is not a wrong answer.** The benchmark runner reports it as
  ``ok=False`` with no answer; this module maps that to a traceless
  :class:`~agent_evolve.core.evaluation.RolloutOutcome`, and the core marks it
  unscorable. It never reaches a denominator.
* **Knowledge-store parity.** Every arm of a comparison must use an identical
  knowledge store, so the choice is explicit and printed. The default is an
  EMPTY worker store: this repo's ``.cuga/knowledge`` holds two leftover
  fixtures (``favorite-color.md``, ``project-clearance-code.md``) that are
  irrelevant to Gaia and would be contamination.
* **No temperature is ever passed.** The endpoint rejects any non-default value.
* **There is one candidate.** No RHO seeder exists, so cross-candidate entropy
  (which needs >= 3 comparable candidates per cell) and DPP diversity (which
  needs alternatives) are inert. The pipeline runs correctly at N=1 and would
  work unchanged at N>1; the count is printed so the inertness is visible
  rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from agent_evolve.adapters.cuga_adapter import CugaAdapter
from agent_evolve.benchmarks.base import Benchmark, BenchmarkTask
from agent_evolve.benchmarks.cuga_executor import (
    DEFAULT_TRACE_ROOT,
    PROCESS_ISOLATION,
    THREAD_ISOLATION,
    HarnessVersion,
    TraceRecorder,
    make_cuga_executor_factory,
    preflight,
)
from agent_evolve.benchmarks.runner import run_benchmark
from agent_evolve.core.analyzer import FakeAnalyzerJudge
from agent_evolve.core.clustering import LexicalEmbedder
from agent_evolve.core.config import ResolvedConfig, resolve_profile
from agent_evolve.core.contracts import (
    EvolutionCandidate,
    EvolutionTask,
    ExecutionTrace,
)
from agent_evolve.core.evaluation import (
    BenchmarkScorer,
    ContractScorer,
    RolloutOutcome,
    ScoreTally,
    Scorer,
)
from agent_evolve.core.fake_editor import FakeEditor
from agent_evolve.core.memory import EditMemory
from agent_evolve.core.orchestrator import SequentialGepaRunner
from agent_evolve.core.pool import PersistentPool
from agent_evolve.core.storage import StorageBackend

__all__ = [
    "DEFAULT_WORKER_KNOWLEDGE_SEED",
    "NOISE_FLOOR_PP",
    "CugaRolloutRunner",
    "EvolutionStack",
    "IterationSummary",
    "build_live_stack",
    "build_offline_stack",
    "describe_knowledge_choice",
    "format_delta",
    "require_safe_rollout_concurrency",
]


#: Two identical baseline runs on the same 42-task Gaia set with the same
#: harness and the same grader scored 10/42 and 17/42. Any reported delta
#: smaller than this is indistinguishable from run-to-run noise, and every
#: delta this pipeline prints carries the caveat.
NOISE_FLOOR_PP = 16.67

#: What a process-isolated worker's knowledge store starts from. ``None`` means
#: EMPTY, on purpose: seeding from ``.cuga/knowledge`` would copy two leftover
#: test fixtures that have nothing to do with Gaia. An operator who wants
#: parity with a serial run passes a path explicitly.
DEFAULT_WORKER_KNOWLEDGE_SEED: Path | None = None

#: The mechanism cluster every score cell is keyed by. One fixed cluster keeps
#: cells comparable across candidates; task-local semantic clustering is a
#: separate, unbuilt stage.
DEFAULT_MECHANISM_CLUSTER = "mechanism-default"

_OFFLINE_ROLLOUT_ISOLATION = "in-process (fake adapter)"


def describe_knowledge_choice(seed: Path | None) -> str:
    """One line naming the knowledge store, for the run header.

    An empty store measurably changes the pass rate, so which one was used is
    part of the measurement, not an implementation detail.
    """
    if seed is None:
        return (
            "EMPTY per worker (not comparable to a serial run that reads "
            ".cuga/knowledge)"
        )
    return f"seeded from {seed}"


def format_delta(before: ScoreTally, after: ScoreTally) -> tuple[str, ...]:
    """Report a before/after delta, always with its denominators and the caveat.

    Returns lines rather than printing so the caller owns output. A delta is
    refused outright when either side scored nothing: there is no delta between
    a number and the absence of one.
    """
    lines = [
        f"before : {before.summary}",
        f"after  : {after.summary}",
    ]
    if before.grader_name != after.grader_name:
        lines.append(
            f"delta  : NOT COMPARABLE -- graded by {before.grader_name!r} then "
            f"{after.grader_name!r}; two graders on the same answers disagree"
        )
        return tuple(lines)
    if before.pass_rate is None or after.pass_rate is None:
        lines.append(
            "delta  : NOT COMPUTABLE -- one side scored zero rollouts, so there "
            "is no rate to compare"
        )
        return tuple(lines)
    delta_pp = (after.pass_rate - before.pass_rate) * 100.0
    verdict = (
        "WITHIN NOISE"
        if abs(delta_pp) < NOISE_FLOOR_PP
        else "above the noise floor, but a single run is still one sample"
    )
    lines.append(f"delta  : {delta_pp:+.2f} pp ({verdict})")
    lines.append(
        f"caveat : the measured noise floor is {NOISE_FLOOR_PP:.2f} pp -- two "
        f"identical baseline runs scored 10/42 and 17/42 on expected_regex. A "
        f"delta below that magnitude carries no information."
    )
    if before.is_partial or after.is_partial:
        lines.append(
            "caveat : at least one side is partial (some rollouts produced no "
            "measurement), so the two denominators differ"
        )
    return tuple(lines)


def require_safe_rollout_concurrency(
    harness: HarnessVersion,
    *,
    max_workers: int,
    isolation: str,
    allow_unsafe_concurrency: bool = False,
) -> None:
    """Refuse an unsafe rollout concurrency, before anything expensive is built.

    Split out of :func:`preflight` and called first because this refusal must be
    **credential-independent**. Ordering matters: a run configured with
    ``--max-workers 4 --isolation thread`` is unsafe whether or not a model is
    configured, and reporting "no model configured" first would send an operator
    to fix the wrong thing and then hit the real refusal after building a CUGA
    wrapper.

    The reason is measured, not precautionary: ``CUGA_FOLDER`` is a single
    process-global environment variable read during ``invoke()``. Two threads
    that each bound a different workspace were observed both reading the second
    one's, while each trace still stamped its own ``harness_version`` -- so the
    run would look clean while measuring a harness that never existed.
    """
    if max_workers <= 1:
        return
    if isolation == PROCESS_ISOLATION:
        return
    preflight(
        harness,
        max_workers=max_workers,
        # A positive count, because this call exists only to reach the
        # concurrency refusal; task selection is validated by the caller.
        tasks=1,
        allow_unsafe_concurrency=allow_unsafe_concurrency,
        isolation=isolation,
    )


# --------------------------------------------------------------------------- #
# Real rollout execution
# --------------------------------------------------------------------------- #
class CugaRolloutRunner:
    """Executes a task batch as real, traced CUGA rollouts.

    Implements :class:`~agent_evolve.core.evaluation.RolloutBatch` on top of
    :func:`~agent_evolve.benchmarks.runner.run_benchmark`, so the evolution loop
    inherits the runner's input-ordered results, per-task timeouts and
    failure-as-data behaviour without the core knowing about benchmarks.

    Refuses ``max_workers > 1`` without a worker pool. This is not caution: two
    threads binding two workspaces were directly observed reading each other's
    ``CUGA_FOLDER``, and the trace cannot detect it because ``harness_version``
    is copied from config. A run that looks clean while measuring a harness that
    never existed is worse than a run that refuses to start.
    """

    def __init__(
        self,
        *,
        harness: HarnessVersion,
        benchmark: Benchmark,
        max_workers: int = 1,
        worker_pool: object | None = None,
        trace_root: Path | str = DEFAULT_TRACE_ROOT,
        task_timeout_seconds: float | None = None,
        recorder: TraceRecorder | None = None,
        allow_unsafe_concurrency: bool = False,
        run_preflight: bool = True,
    ) -> None:
        self.harness = harness
        self.benchmark = benchmark
        self.max_workers = int(max_workers)
        self.worker_pool = worker_pool
        self.trace_root = Path(trace_root)
        self.task_timeout_seconds = task_timeout_seconds
        self.recorder = recorder if recorder is not None else TraceRecorder()
        self.isolation = (
            PROCESS_ISOLATION if worker_pool is not None else THREAD_ISOLATION
        )
        # The concurrency refusal comes first and needs no credentials, so an
        # unsafe worker count is reported as an unsafe worker count rather than
        # as a missing model.
        require_safe_rollout_concurrency(
            harness,
            max_workers=self.max_workers,
            isolation=self.isolation,
            allow_unsafe_concurrency=allow_unsafe_concurrency,
        )
        # Everything else that can be rejected is rejected before the first
        # billed token: absent model configuration, an empty task set.
        if run_preflight:
            preflight(
                harness,
                max_workers=self.max_workers,
                tasks=max(1, len(benchmark.load_tasks())),
                allow_unsafe_concurrency=allow_unsafe_concurrency,
                isolation=self.isolation,
            )

    def run_rollouts(
        self, version: str, tasks: Sequence[EvolutionTask], *, prefix: str
    ) -> tuple[RolloutOutcome, ...]:
        """Roll every task out once, returning a traceless outcome for failures.

        The grader passed to ``run_benchmark`` is irrelevant here -- the core
        scores with its own :class:`Scorer` so one definition of the number
        exists -- but the parameter is mandatory, so the benchmark's first
        declared grader is used and its outcomes are discarded.
        """
        if not tasks:
            return ()
        by_id = {task.task_id: task for task in tasks}
        bench_tasks = tuple(
            BenchmarkTask(task_id=task.task_id, question=task.input_text)
            for task in tasks
        )
        factory = make_cuga_executor_factory(
            self.harness,
            trace_root=self.trace_root,
            recorder=self.recorder,
            worker_pool=self.worker_pool,  # type: ignore[arg-type]
        )
        result = run_benchmark(
            self.benchmark,
            factory,
            grader=self.benchmark.graders()[0],
            max_workers=self.max_workers,
            task_timeout_seconds=self.task_timeout_seconds,
            tasks=bench_tasks,
        )
        outcomes: list[RolloutOutcome] = []
        for execution in result.executions:
            task = by_id[execution.task.task_id]
            if not execution.ok or execution.answer is None:
                outcomes.append(
                    RolloutOutcome(
                        task=task,
                        trace=None,
                        error=execution.error or "no answer was produced",
                    )
                )
                continue
            trace_path = self.recorder.trace_path(task.task_id)
            outcomes.append(
                RolloutOutcome(
                    task=task,
                    trace=ExecutionTrace(
                        trace_id=f"{prefix}-{task.task_id}",
                        candidate_id=version,
                        task_id=task.task_id,
                        events=_events_from_trace_dir(trace_path),
                        final_output=execution.answer,
                        status="success",
                    ),
                )
            )
        return tuple(outcomes)

    def close(self) -> None:
        """Release the worker pool. Every worker holds a knowledge-store lock."""
        closer = getattr(self.worker_pool, "close", None)
        if closer is not None:
            closer()


def _events_from_trace_dir(trace_path: Path | None) -> tuple:
    """Load a persisted trace's events, or return none.

    Reuses ``CugaAdapter._rich_events`` rather than re-parsing: the trace format
    has exactly one reader, and payload blobs (which may hold raw prompts and
    expected answers) are never dereferenced by it.
    """
    if trace_path is None:
        return ()
    directory = trace_path if trace_path.is_dir() else trace_path.parent
    try:
        events = CugaAdapter._rich_events(directory)
    except Exception:  # noqa: BLE001 - a missing/corrupt trace is not fatal here
        return ()
    return events or ()


# --------------------------------------------------------------------------- #
# The stack
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class IterationSummary:
    """What one outer iteration did. Counts only; no grading material."""

    iteration: int
    attempts: int
    accepted: int
    rejected: int
    no_issue: int
    pool_size: int
    unscorable_probes: int
    analysis_failures: int

    @property
    def line(self) -> str:
        return (
            f"iteration {self.iteration}: attempts={self.attempts} "
            f"accepted={self.accepted} rejected={self.rejected} "
            f"no_issue={self.no_issue} pool={self.pool_size} "
            f"unscorable_probes={self.unscorable_probes} "
            f"analysis_failures={self.analysis_failures}"
        )


@dataclass(slots=True)
class EvolutionStack:
    """An assembled, ready-to-run pipeline.

    Holds every component so a caller can inspect what it actually got -- which
    scorer, how many analyzer workers, which knowledge store -- instead of
    trusting a flag it passed in.
    """

    runner: SequentialGepaRunner
    adapter: object
    pool: PersistentPool
    tasks: tuple[EvolutionTask, ...]
    scorer: Scorer
    base_version: str
    rollout_isolation: str
    knowledge_seed: Path | None
    uses_real_agent: bool
    rollout_workers: int = 1
    trace_root: Path | None = None
    _closers: tuple[Callable[[], None], ...] = ()

    # -- inspection ------------------------------------------------------- #

    @property
    def grader_name(self) -> str:
        return self.scorer.grader_name

    @property
    def analyzer_workers(self) -> int:
        return self.runner.analyzer_workers

    def candidate_count(self) -> int:
        return len(self.pool)

    def pool_size(self) -> int:
        return len(self.pool)

    @property
    def header_lines(self) -> tuple[str, ...]:
        """Every choice that can change the reported number, named up front."""
        return (
            f"mode            : {'live CUGA' if self.uses_real_agent else 'dry run (fake stack, offline)'}",
            f"adapter         : {getattr(self.adapter, 'adapter_name', type(self.adapter).__name__)}",
            f"analyzer        : {type(self.runner.analyzer_judge).__name__}",
            f"editor          : {type(self.runner.editor).__name__}",
            f"grader          : {self.grader_name}",
            f"tasks           : {len(self.tasks)}",
            f"rollout workers : {self.rollout_workers}",
            f"rollout isolation: {self.rollout_isolation}",
            f"analyzer workers: {self.analyzer_workers}",
            f"knowledge store : {describe_knowledge_choice(self.knowledge_seed)}",
            f"trace root      : {self.trace_root if self.trace_root else '<none>'}",
            (
                f"candidates      : {self.candidate_count()} (base only -- no RHO "
                f"seeder exists, so cross-candidate entropy and DPP diversity are "
                f"inert)"
            ),
        )

    # -- execution -------------------------------------------------------- #

    def measure(self, version: str, *, prefix: str = "measure") -> ScoreTally:
        """Score one version over this stack's task set."""
        return self.runner.measure(version, self.tasks, prefix=prefix)

    def run_iterations(self, iterations: int) -> tuple[IterationSummary, ...]:
        """Run ``iterations`` outer iterations, one GEPA attempt each."""
        if isinstance(iterations, bool) or not isinstance(iterations, int):
            raise ValueError("iterations must be a positive integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        summaries: list[IterationSummary] = []
        for index in range(1, iterations + 1):
            probes_before = self.runner.unscorable_probe_count
            failures_before = len(self.runner.analysis_failures)
            outcome = self.runner.run_attempt(self.tasks)
            pending = outcome.status.value == "pending"
            summaries.append(
                IterationSummary(
                    iteration=index,
                    attempts=1,
                    accepted=1 if outcome.accepted else 0,
                    rejected=0 if (outcome.accepted or pending) else 1,
                    no_issue=1 if pending else 0,
                    pool_size=len(self.pool),
                    unscorable_probes=(
                        self.runner.unscorable_probe_count - probes_before
                    ),
                    analysis_failures=(
                        len(self.runner.analysis_failures) - failures_before
                    ),
                )
            )
        return tuple(summaries)

    def champion_version(self) -> str:
        """The best candidate's version, or the base's when none was accepted."""
        try:
            champion = self.pool.select_champion(config=self.runner.config)
        except ValueError:
            return self.base_version
        return self.pool.get(champion.candidate_id).version

    def close(self) -> None:
        for closer in self._closers:
            closer()


# --------------------------------------------------------------------------- #
# Offline stack
# --------------------------------------------------------------------------- #
def _offline_tasks(count: int, token: str) -> tuple[EvolutionTask, ...]:
    """A deterministic offline task set the fake editor can actually repair."""
    return tuple(
        EvolutionTask(
            task_id=f"task-{index}",
            input_text=f"produce result {index}",
            expected_contract={"expected_substring": token},
        )
        for index in range(1, count + 1)
    )


def build_offline_stack(
    *,
    task_count: int = 3,
    tasks: Sequence[EvolutionTask] | None = None,
    task_token: str = "graphrag-retrieval",
    adapter: object | None = None,
    analyzer: object | None = None,
    analyzer_factory: Callable[[], object] | None = None,
    analyzer_workers: int = 1,
    editor: object | None = None,
    benchmark: Benchmark | None = None,
    grader: str | None = None,
    storage: StorageBackend | None = None,
    seed: int = 0,
    profile: str = "research_sequential",
) -> EvolutionStack:
    """Assemble the fake stack: no CUGA, no model endpoint, no network.

    Same runner, same lifecycle, same scoring path as a live run -- only the
    adapter, analyzer and editor are deterministic fakes. That is deliberate:
    a dry run that took a different code path would not rehearse anything.

    A ``benchmark`` may be supplied to exercise benchmark-driven scoring
    offline; without one, task contracts are scored, which is what the existing
    offline suite does.
    """
    from examples.fake_adapter import FakeAdapter

    resolved_adapter = adapter if adapter is not None else FakeAdapter()
    resolved_tasks = (
        tuple(tasks) if tasks is not None else _offline_tasks(task_count, task_token)
    )

    if analyzer_factory is not None:
        probe = analyzer_factory()
        resolved_analyzer: object = analyzer if analyzer is not None else probe
    else:
        resolved_analyzer = analyzer if analyzer is not None else FakeAnalyzerJudge()

    scorer: Scorer
    if benchmark is not None:
        if grader is None:
            raise ValueError(
                "a benchmark requires an explicit grader: the grader is never "
                "silently chosen, because two graders on the same benchmark "
                "disagree"
            )
        scorer = BenchmarkScorer(benchmark=benchmark, grader=grader)
    else:
        scorer = ContractScorer()

    base_version = "base-v0"
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(
        EvolutionCandidate(
            candidate_id="base",
            version=base_version,
            artifact_hashes={
                d.artifact_id: d.version_hash
                for d in resolved_adapter.artifact_inventory(base_version)  # type: ignore[attr-defined]
            },
        )
    )

    config = resolve_profile(
        profile, seed=seed, max_analyzer_workers=max(1, int(analyzer_workers))
    )
    runner = SequentialGepaRunner(
        adapter=resolved_adapter,  # type: ignore[arg-type]
        pool=pool,
        analyzer_judge=resolved_analyzer,  # type: ignore[arg-type]
        editor=editor if editor is not None else FakeEditor(),  # type: ignore[arg-type]
        embedder=LexicalEmbedder(dim=32),
        storage=storage,
        config=config,
        mechanism_cluster_id=DEFAULT_MECHANISM_CLUSTER,
        seed=seed,
        scorer=scorer,
        analyzer_factory=analyzer_factory,  # type: ignore[arg-type]
    )
    return EvolutionStack(
        runner=runner,
        adapter=resolved_adapter,
        pool=pool,
        tasks=resolved_tasks,
        scorer=scorer,
        base_version=base_version,
        rollout_isolation=_OFFLINE_ROLLOUT_ISOLATION,
        knowledge_seed=None,
        uses_real_agent=False,
        rollout_workers=1,
        trace_root=None,
    )


# --------------------------------------------------------------------------- #
# Live stack
# --------------------------------------------------------------------------- #
def _tasks_from_benchmark(
    benchmark: Benchmark, limit: int | None
) -> tuple[EvolutionTask, ...]:
    """Convert benchmark tasks into evolution tasks, carrying no grading material.

    ``expected_contract`` is left empty on purpose. The benchmark's grader owns
    the answer key; copying it onto the task would put it in front of the
    analyzer and the editor, and a "diagnosis" that compares an answer to the
    key is not causal reasoning.
    """
    loaded = benchmark.load_tasks()
    selected = loaded if limit is None else loaded[:limit]
    return tuple(
        EvolutionTask(
            task_id=task.task_id, input_text=task.question, expected_contract={}
        )
        for task in selected
    )


def build_live_stack(
    *,
    benchmark: Benchmark,
    grader: str,
    harness: HarnessVersion,
    task_limit: int | None = None,
    max_workers: int = 1,
    analyzer_workers: int = 1,
    isolation: str = THREAD_ISOLATION,
    worker_root: Path | str = Path("data/cuga-workers"),
    knowledge_seed: Path | None = DEFAULT_WORKER_KNOWLEDGE_SEED,
    trace_root: Path | str = DEFAULT_TRACE_ROOT,
    task_timeout_seconds: float | None = 1200.0,
    storage: StorageBackend | None = None,
    seed: int = 0,
    profile: str = "research_sequential",
    allow_unsafe_concurrency: bool = False,
) -> EvolutionStack:
    """Assemble the live stack: real CUGA rollouts, real analyzer, real editor.

    Imports of CUGA-backed components are deferred to call time so importing
    this module -- which the offline tests do -- never requires the SDK or a
    model endpoint.

    ``isolation`` must be ``process`` for ``max_workers > 1``; the refusal comes
    from :class:`CugaRolloutRunner`, which is where the reason is documented.
    """
    from agent_evolve.adapters.cuga_analyzer import CugaTrajectoryAnalyzer
    from agent_evolve.adapters.cuga_editor import CugaEditorAgent
    from agent_evolve.benchmarks.cuga_process_pool import CugaProcessPool
    from agent_evolve.core.trace import PayloadLevel
    from agent_evolve.cuga_wrapper import CugaWrapper, RuntimeSettings, TraceConfig

    trace_root = Path(trace_root)
    # Refused first, before any CUGA process, wrapper or model lookup: an unsafe
    # worker count is unsafe regardless of credentials, and diagnosing it as a
    # missing model would send an operator to fix the wrong thing.
    require_safe_rollout_concurrency(
        harness,
        max_workers=max_workers,
        isolation=isolation,
        allow_unsafe_concurrency=allow_unsafe_concurrency,
    )
    scorer = BenchmarkScorer(benchmark=benchmark, grader=grader)
    tasks = _tasks_from_benchmark(benchmark, task_limit)
    if not tasks:
        raise ValueError("no tasks selected; nothing to evolve against")

    wrapper = CugaWrapper.from_cuga(
        RuntimeSettings.from_env(),
        trace_config=TraceConfig(
            enabled=True,
            output_root=trace_root,
            payload_level=PayloadLevel.RAW_OPT_IN,
            allow_raw_payloads=True,
            capture_node_payloads=True,
        ),
    )
    adapter = CugaAdapter(wrapper)
    base_version = "base"
    adapter.register_candidate(base_version, _harness_artifacts(harness))

    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(
        EvolutionCandidate(
            candidate_id="base",
            version=base_version,
            artifact_hashes={
                d.artifact_id: d.version_hash
                for d in adapter.artifact_inventory(base_version)
            },
        )
    )

    worker_pool: CugaProcessPool | None = None
    if isolation == PROCESS_ISOLATION:
        worker_pool = CugaProcessPool(
            root=Path(worker_root),
            trace_root=trace_root,
            task_timeout=task_timeout_seconds,
            knowledge_seed=knowledge_seed,
        )

    rollout_batch = CugaRolloutRunner(
        harness=harness,
        benchmark=benchmark,
        max_workers=max_workers,
        worker_pool=worker_pool,
        trace_root=trace_root,
        task_timeout_seconds=task_timeout_seconds,
        allow_unsafe_concurrency=allow_unsafe_concurrency,
    )

    config = resolve_profile(
        profile, seed=seed, max_analyzer_workers=max(1, int(analyzer_workers))
    )
    # No temperature is ever passed: the endpoint rejects any non-default value.
    analyzer_factory = CugaTrajectoryAnalyzer.factory()
    runner = SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        # The report-based analyzer is adapted by the runner's own shim; the
        # static mismatch here is the whole reason the shim exists.
        analyzer_judge=analyzer_factory(),  # type: ignore[arg-type]
        editor=CugaEditorAgent(adapter=adapter, memory=EditMemory()),
        embedder=LexicalEmbedder(dim=32),
        storage=storage,
        config=config,
        mechanism_cluster_id=DEFAULT_MECHANISM_CLUSTER,
        seed=seed,
        scorer=scorer,
        rollout_batch=rollout_batch,
        analyzer_factory=analyzer_factory,
    )
    return EvolutionStack(
        runner=runner,
        adapter=adapter,
        pool=pool,
        tasks=tasks,
        scorer=scorer,
        base_version=base_version,
        rollout_isolation=rollout_batch.isolation,
        knowledge_seed=knowledge_seed,
        uses_real_agent=True,
        rollout_workers=max_workers,
        trace_root=trace_root,
        _closers=(rollout_batch.close,),
    )


def _harness_artifacts(harness: HarnessVersion) -> dict[str, str]:
    """Map a harness version onto the adapter's artifact ids.

    The base must own at least one writable artifact, or every issue is dropped
    for lack of attribution and the loop can never act. A harness with no
    injected artifacts therefore gets one empty, editable skill slot rather than
    an empty inventory that would fail silently.
    """
    artifacts: dict[str, str] = {}
    if harness.instructions:
        artifacts["instructions"] = harness.instructions
    for group in ("skills", "policies", "memory"):
        for name, body in getattr(harness, group).items():
            artifacts[f"{group}/{name}"] = body
    if not artifacts:
        artifacts["skills/generated-evolved"] = ""
    return artifacts
