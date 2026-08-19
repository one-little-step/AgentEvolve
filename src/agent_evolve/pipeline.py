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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

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
from agent_evolve.core.clustering import LexicalEmbedder, MechanismEmbedder
from agent_evolve.core.rho.cache import JsonDiskCache
from agent_evolve.core.config import ResolvedConfig, resolve_profile
from agent_evolve.core.contracts import (
    ArtifactEdit,
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
from agent_evolve.core.pool import PersistentPool, ScoreProvenance
from agent_evolve.core.run_logging import (
    LogCaptureConfig,
    RunLogSink,
    build_sinks,
)
from agent_evolve.core.storage import StorageBackend

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent_evolve.core.rho.history import HistoryLoadReport
    from agent_evolve.core.rho.rounds import RhoHooks

__all__ = [
    "CANDIDATE_FILENAME_PREFIX",
    "CHAMPION_FILENAME",
    "DEFAULT_WORKER_KNOWLEDGE_SEED",
    "EXPORT_FORMAT",
    "NOISE_FLOOR_PP",
    "CugaRolloutRunner",
    "EvolutionStack",
    "IterationSummary",
    "RhoBinding",
    "build_live_stack",
    "build_offline_stack",
    "build_rho_hooks",
    "describe_knowledge_choice",
    "export_harness",
    "format_delta",
    "harness_payload",
    "harness_version_name",
    "nothing_accepted_warning",
    "nothing_accepted_warning_applies",
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
    #: Iterations that did nothing because a spend cap was already reached.
    #: Separate from ``no_issue``: "found nothing to fix" and "was not allowed to
    #: try" have opposite meanings, and reporting a budget stop as no_issue sends
    #: the reader to the analyzer to debug a cap they set themselves.
    budget_exhausted: int = 0

    @property
    def line(self) -> str:
        base = (
            f"iteration {self.iteration}: attempts={self.attempts} "
            f"accepted={self.accepted} rejected={self.rejected} "
            f"no_issue={self.no_issue} pool={self.pool_size} "
            f"unscorable_probes={self.unscorable_probes} "
            f"analysis_failures={self.analysis_failures}"
        )
        if self.budget_exhausted:
            base += " BUDGET EXHAUSTED (no attempt issued)"
        return base


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
    #: Every sink this stack owns, by channel. Held so :meth:`close` can flush
    #: them: a stream still open at exit loses its final lines.
    log_sinks: Mapping[str, RunLogSink] = field(default_factory=dict)
    #: Which phases the run selected, or ``""`` for the plain genetic loop. Read
    #: only by the header: the "no RHO seeder exists" caveat is true of a genetic
    #: run and false of a RHO one, and printing it under ``--mode rho`` would
    #: tell an operator to discount a real cross-candidate result as inert.
    mode: str = "genetic"
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
            f"log capture     : {self._describe_capture()}",
            f"candidates      : {self.candidate_count()}{self._describe_seeding()}",
        )

    def _describe_seeding(self) -> str:
        """Whether cross-candidate machinery can do anything on this run.

        Under ``--mode genetic`` the pool starts and stays at one lineage, so
        cross-candidate entropy (which needs >= 3 comparable candidates per cell)
        and DPP diversity (which needs alternatives) are genuinely inert, and
        saying so keeps an operator from reading a selection decision into them.
        Under a RHO mode a seeder runs and produces N candidates, so the same
        sentence would be a false reassurance about a real result.
        """
        if self.mode == "genetic":
            return (
                " (base only -- no RHO seeder runs in this mode, so "
                "cross-candidate entropy and DPP diversity are inert)"
            )
        return (
            f" before the first round (the RHO seeder adds N per round; "
            f"ALL are retained)"
        )

    def _describe_capture(self) -> str:
        """Name the capture choice: it decides what survives the run.

        Printed for the same reason the knowledge store is: "were the analyzer
        transcripts kept?" is a question asked after the run, when the answer can
        no longer be changed.
        """
        capture = getattr(self.runner.config, "log_capture", None)
        if capture is None or not capture.enabled:
            return "OFF (nothing written; a later 'why did it route that way?' needs a re-run)"
        return f"{capture.root} channels={','.join(capture.channels)}"

    # -- execution -------------------------------------------------------- #

    def measure(self, version: str, *, prefix: str = "measure") -> ScoreTally:
        """Score one version over this stack's task set."""
        tally = self.runner.measure(version, self.tasks, prefix=prefix)
        self._record(
            f"{prefix}__{version}",
            {
                "event": "measured",
                "version": version,
                "prefix": prefix,
                "grader_name": tally.grader_name,
                "passed": tally.passed,
                "evaluated": tally.evaluated,
                "attempted": tally.attempted,
                "unscorable": tally.unscorable,
                "unscorable_task_ids": list(tally.unscorable_task_ids),
                "pass_rate": tally.pass_rate,
            },
        )
        return tally

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
            self._record(
                f"iteration-{index}",
                {
                    "event": "iteration_start",
                    "iteration": index,
                    "pool_size": len(self.pool),
                    "tasks": [t.task_id for t in self.tasks],
                },
            )
            outcome = self.runner.run_attempt(self.tasks)
            pending = outcome.status.value == "pending"
            # A budget stop is a planned non-attempt, so it must not be counted
            # as an attempt or as "no issue found".
            capped = pending and "budget exhausted" in (outcome.reason or "")
            summary = IterationSummary(
                iteration=index,
                attempts=0 if capped else 1,
                accepted=1 if outcome.accepted else 0,
                rejected=0 if (outcome.accepted or pending) else 1,
                no_issue=1 if (pending and not capped) else 0,
                budget_exhausted=1 if capped else 0,
                pool_size=len(self.pool),
                unscorable_probes=(
                    self.runner.unscorable_probe_count - probes_before
                ),
                analysis_failures=(
                    len(self.runner.analysis_failures) - failures_before
                ),
            )
            summaries.append(summary)
            # The attempt's own identity travels with the counts: an accepted or
            # rejected edit is only attributable through its attempt and issue,
            # and ``reason`` is the only record of *why* a rejection happened.
            self._record(
                f"iteration-{index}",
                {
                    "event": "iteration_end",
                    "iteration": index,
                    "accepted": summary.accepted,
                    "rejected": summary.rejected,
                    "no_issue": summary.no_issue,
                    "pool_size": summary.pool_size,
                    "unscorable_probes": summary.unscorable_probes,
                    "analysis_failures": summary.analysis_failures,
                    "attempt_id": outcome.attempt_id,
                    "issue_id": outcome.issue_id,
                    "parent_candidate_id": outcome.parent_candidate_id,
                    "result_candidate_id": outcome.result_candidate_id,
                    "status": outcome.status.value,
                    "weighted_net_gain": outcome.weighted_net_gain,
                    "reason": outcome.reason,
                    "artifact_ids": list(outcome.artifact_ids),
                    "fallback_reason": outcome.fallback_reason,
                },
            )
        return tuple(summaries)

    def champion_version(self) -> str:
        """The best candidate's version, or the base's when none was accepted."""
        try:
            champion = self.pool.select_champion(config=self.runner.config)
        except ValueError:
            return self.base_version
        return self.pool.get(champion.candidate_id).version

    def export_pool(self, path: Path) -> tuple[Path, ...]:
        """Persist every pool candidate as a re-runnable harness file.

        Two shapes, chosen by the target's suffix:

        ``*.json``
            One file, the champion only, ready to hand straight to ``--harness``.
        anything else
            A directory: ``candidate-<id>.json`` per pool member plus
            ``champion.json``. Every candidate is written because a sibling
            proposal cost real rollouts to produce and is exactly what an RHO
            seeded run starts from; only the champion would throw the frontier
            away.

        Returns the files written, champion last, so a caller can report them.
        """
        target = Path(path)
        try:
            champion_id: str | None = self.pool.select_champion(
                config=self.runner.config
            ).candidate_id
        except ValueError:
            # No candidate carries comparable evidence yet, so selection has no
            # opinion. The base is still exported and still named the champion:
            # it is what the next run would execute against.
            champion_id = None

        def provenance_for(entry: object, *, is_champion: bool) -> dict[str, object]:
            record = _entry_provenance(entry, is_champion=is_champion)
            record["source_base_version"] = self.base_version
            record["grader_name"] = self.grader_name
            record["task_ids"] = [t.task_id for t in self.tasks]
            return record

        champion_entry = (
            self.pool.get(champion_id) if champion_id is not None else self.pool.base
        )

        if target.suffix == ".json":
            return (
                export_harness(
                    self.adapter,
                    version=champion_entry.version,
                    candidate_id=champion_entry.candidate_id,
                    path=target,
                    provenance=provenance_for(champion_entry, is_champion=True),
                ),
            )

        written: list[Path] = []
        for entry in self.pool.all_entries():
            is_champion = entry.candidate_id == champion_entry.candidate_id
            written.append(
                export_harness(
                    self.adapter,
                    version=entry.version,
                    candidate_id=entry.candidate_id,
                    path=(
                        target
                        / f"{CANDIDATE_FILENAME_PREFIX}"
                        f"{_safe_filename_part(entry.candidate_id)}.json"
                    ),
                    provenance=provenance_for(entry, is_champion=is_champion),
                )
            )
        written.append(
            export_harness(
                self.adapter,
                version=champion_entry.version,
                candidate_id=champion_entry.candidate_id,
                path=target / CHAMPION_FILENAME,
                provenance=provenance_for(champion_entry, is_champion=True),
            )
        )
        return tuple(written)

    def _record(self, name: str, record: Mapping[str, object]) -> None:
        """Best-effort pipeline record. Never raises and never changes a number.

        Swallows every error for the same reason the adapters do: a logging
        failure must not discard rollouts that have already been paid for.
        """
        sink = self.log_sinks.get("pipeline")
        if sink is None:
            return
        try:
            sink.write_record(name, record)
        except Exception:  # noqa: BLE001 - capture is an observer, never a gate
            pass

    def close(self) -> None:
        for closer in self._closers:
            closer()
        # After the components: a sink closed first would drop records written
        # during a component's own teardown.
        for sink in self.log_sinks.values():
            sink.close()


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


def _override_kwargs(overrides: Mapping[str, object] | None) -> dict:
    """Sanitize caller overrides before they are splatted into resolve_profile.

    ``environ`` is a positional parameter of ``resolve_profile``, not a config
    field, so an override carrying that key would bind to it and silently
    replace the environment mapping. Every other key is validated by
    ``resolve_profile`` itself against ``_VALID_OVERRIDES``.
    """
    return {k: v for k, v in dict(overrides or {}).items() if k != "environ"}


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
    log_capture: LogCaptureConfig | None = None,
    config_overrides: Mapping[str, object] | None = None,
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

    capture = log_capture if log_capture is not None else LogCaptureConfig()
    config = resolve_profile(
        profile,
        seed=seed,
        max_analyzer_workers=max(1, int(analyzer_workers)),
        log_capture=capture,
        # Overrides land last so an explicit flag beats the profile. Keys are
        # validated by resolve_profile against _VALID_OVERRIDES; ``environ`` is
        # filtered out because it is positional there, not a config field.
        **_override_kwargs(config_overrides),
    )
    # Every channel, active or not, so a fake analyzer or editor that grows a
    # sink later needs no change here and ``close()`` has one thing to close.
    sinks = build_sinks(capture)
    resolved_editor = editor if editor is not None else FakeEditor()
    # The runner writes attempt history; the editor's history tools read it.
    # They must share one object or the editor reads an empty store (SV-6).
    # An injected editor may already own a memory, so adopt that one rather than
    # creating a second; otherwise build one bound to this run's storage.
    editor_memory = getattr(resolved_editor, "memory", None)
    edit_memory = (
        editor_memory
        if isinstance(editor_memory, EditMemory)
        else EditMemory(storage=storage)
    )
    runner = SequentialGepaRunner(
        adapter=resolved_adapter,  # type: ignore[arg-type]
        pool=pool,
        analyzer_judge=resolved_analyzer,  # type: ignore[arg-type]
        editor=resolved_editor,  # type: ignore[arg-type]
        embedder=LexicalEmbedder(dim=32),
        storage=storage,
        config=config,
        mechanism_cluster_id=DEFAULT_MECHANISM_CLUSTER,
        seed=seed,
        scorer=scorer,
        analyzer_factory=analyzer_factory,  # type: ignore[arg-type]
        edit_memory=edit_memory,
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
        log_sinks=sinks,
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
    log_capture: LogCaptureConfig | None = None,
    max_rollouts_per_worker: int | None = None,
    config_overrides: Mapping[str, object] | None = None,
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
    from agent_evolve.benchmarks import cuga_process_pool
    from agent_evolve.core.trace import PayloadLevel
    from agent_evolve.cuga_wrapper import CugaWrapper, RuntimeSettings, TraceConfig

    trace_root = Path(trace_root)
    capture = log_capture if log_capture is not None else LogCaptureConfig()
    # One sink per channel, built before the components that write to them so
    # the analyzer factory, the editor and the pool all receive the same config.
    sinks = build_sinks(capture)
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

    worker_pool: object | None = None
    if isolation == PROCESS_ISOLATION:
        # Resolved through the module rather than imported by name so the pool a
        # test substitutes is the pool the composition root builds.
        worker_pool = cuga_process_pool.CugaProcessPool(
            root=Path(worker_root),
            trace_root=trace_root,
            task_timeout=task_timeout_seconds,
            knowledge_seed=knowledge_seed,
            log_capture=capture,
            # Bounds the dominant memory-growth term: a worker is replaced after
            # this many rollouts instead of accumulating SDK state for the whole
            # run (2026-08-19: ~90 GB, machine killed). None keeps the module
            # default rather than hardcoding a second one here.
            **(
                {"max_rollouts_per_worker": max_rollouts_per_worker}
                if max_rollouts_per_worker is not None
                else {}
            ),
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
        profile,
        seed=seed,
        max_analyzer_workers=max(1, int(analyzer_workers)),
        log_capture=capture,
        # Overrides land last so an explicit flag beats the profile. Keys are
        # validated by resolve_profile against _VALID_OVERRIDES; ``environ`` is
        # filtered out because it is positional there, not a config field.
        **_override_kwargs(config_overrides),
    )
    # No temperature is ever passed: the endpoint rejects any non-default value.
    # The sink goes through the factory, not onto the single instance: the runner
    # rebuilds one analyzer per worker thread, and an instance-only sink would
    # capture nothing from a fanned-out analysis.
    analyzer_factory = CugaTrajectoryAnalyzer.factory(log_sink=sinks["analyzer"])
    # One EditMemory, shared by the runner (which writes every attempt into it)
    # and the editor (whose history tools read it). Two instances would give the
    # editor an empty store to read from, which is the SV-6 failure mode.
    edit_memory = EditMemory(storage=storage)
    runner = SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        # The report-based analyzer is adapted by the runner's own shim; the
        # static mismatch here is the whole reason the shim exists.
        analyzer_judge=analyzer_factory(),  # type: ignore[arg-type]
        editor=CugaEditorAgent(
            adapter=adapter, memory=edit_memory, log_sink=sinks["editor"]
        ),
        embedder=LexicalEmbedder(dim=32),
        storage=storage,
        config=config,
        mechanism_cluster_id=DEFAULT_MECHANISM_CLUSTER,
        seed=seed,
        scorer=scorer,
        rollout_batch=rollout_batch,
        analyzer_factory=analyzer_factory,
        edit_memory=edit_memory,
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
        log_sinks=sinks,
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


# --------------------------------------------------------------------------- #
# RHO round wiring
#
# ``core/rho/rounds.py`` may not import ``cuga``, ``litellm``, or
# ``agent_evolve.adapters``; every model call, agent invocation and rollout
# therefore arrives as an injected callable on ``RhoHooks``. This section is the
# one place allowed to bind those callables to live adapters, which is the whole
# reason ``RhoHooks`` exists rather than a set of direct imports inside the core.
# --------------------------------------------------------------------------- #

#: What produced a RHO cell's diagnosis, recorded on every score provenance so a
#: later reader can tell a RHO-round cell from a genetic-attempt cell.
RHO_ANALYZER_MODEL_ID = "cuga-rho-group-diagnoser"

#: What ranked a RHO candidate. The *score* in the cell comes from the grader,
#: not from this judge; the judge decides reported ordering only.
RHO_JUDGE_MODEL_ID = "cuga-preference-judge"


@dataclass(slots=True)
class RhoBinding:
    """Mutable state the RHO hooks share, kept out of the closures' scope.

    Held as an object rather than as closure cells so a caller can read the
    counters after a round: "how many rollouts produced no measurement?" is a
    question asked after the run, when the answer can no longer be recovered.
    """

    #: version -> (task_id, cluster_id) -> list of scores awaiting pool commit.
    #: Buffered because ``run_round`` scores a candidate in phase 8 but commits
    #: it in phase 10, and ``PersistentPool.record_score`` refuses a candidate
    #: that is not in the pool yet.
    pending_scores: dict[str, dict[tuple[str, str], list[float]]] = field(
        default_factory=dict
    )
    #: Rollouts whose trace carried no measurement. Counted, never scored as
    #: zero into the pool: a probe with no evidence is not a failing probe.
    unscorable: int = 0
    #: Monotonic candidate registration counter, so two proposals never share a
    #: workspace or a pool candidate id.
    registrations: int = 0
    #: Every candidate version committed to the pool, in commit order.
    committed_versions: tuple[str, ...] = ()


def build_rho_hooks(
    stack: EvolutionStack,
    *,
    history_root: Path | None = None,
    comprehender: object | None = None,
    difficulty_judge: object | None = None,
    diagnoser: object | None = None,
    optimizer: object | None = None,
    preference_judge: object | None = None,
    summary_cache_root: Path | None = None,
    difficulty_cache_root: Path | None = None,
    embedding_cache_root: Path | None = None,
    proposal_temperature: float | None = None,
    expected_answer_for: Callable[[str], str | None] | None = None,
    binding: RhoBinding | None = None,
) -> "RhoHooks":
    """Bind every :class:`RhoHooks` field to a live adapter over ``stack``.

    All five RHO components are injectable and default to the real adapters,
    constructed lazily so importing this module never requires the CUGA SDK.
    An offline test injects duck-typed fakes; the shapes each one must satisfy
    are documented on ``RhoHooks`` itself.

    ``history_root`` is optional: its absence is a COLD START, which
    :func:`~agent_evolve.core.rho.history.load_history` reports as data rather
    than as an error.

    Notes on the two hooks whose contract is not a straight delegation:

    ``rollout`` adapts the batch-shaped ``run_rollouts`` to the round's
    per-index call and **never raises** -- a raised exception would discard a
    whole group's evidence for one broken rollout. It also resolves the round's
    fixed ``BASE_VERSION`` ("base") to whatever this stack actually named its
    incumbent, because the offline stack calls it ``base-v0`` and rolling out a
    version the adapter never registered would fail silently.

    ``score`` returns the grader's number but records into the pool only when
    the rollout was *scorable*. An unscorable rollout still reaches the entropy
    tracker as ``0.0`` because the hook signature carries no "unmeasured"
    channel; the count is kept on :class:`RhoBinding` and logged rather than
    hidden.

    ``contamination_literals`` is deliberately left empty. The scan is
    observational, and the only way to populate it is to hand this function an
    answer key -- which would then travel into a manifest.
    """
    from agent_evolve.core.rho.cache import JsonDiskCache
    from agent_evolve.core.rho.history import HistoryLoadReport, load_history
    from agent_evolve.core.rho.rounds import BASE_VERSION, RhoHooks, rho_cluster_id

    state = binding if binding is not None else RhoBinding()
    adapter = stack.adapter
    pool = stack.pool
    scorer = stack.scorer

    summary_cache = JsonDiskCache(summary_cache_root)
    difficulty_cache = JsonDiskCache(difficulty_cache_root)
    embedding_cache = JsonDiskCache(embedding_cache_root)

    if comprehender is None:
        from agent_evolve.adapters.cuga_rho_comprehender import RhoComprehender

        comprehender = RhoComprehender(cache=summary_cache)
    elif hasattr(comprehender, "cache") and summary_cache_root is not None:
        comprehender.cache = summary_cache  # type: ignore[attr-defined]

    if difficulty_judge is None:
        from agent_evolve.adapters.cuga_rho_judge import RhoDifficultyJudge

        difficulty_judge = RhoDifficultyJudge(cache=difficulty_cache)
    elif hasattr(difficulty_judge, "cache") and difficulty_cache_root is not None:
        difficulty_judge.cache = difficulty_cache  # type: ignore[attr-defined]

    if diagnoser is None:
        from agent_evolve.adapters.cuga_rho_diagnoser import RhoGroupDiagnoser

        diagnoser = RhoGroupDiagnoser()

    if optimizer is None:
        from agent_evolve.adapters.cuga_rho_optimizer import RhoOptimizer

        # Never 0.0: the endpoint rejects it and RhoOptimizer raises. Unset by
        # default -- diversity comes from N independent invocations, not from
        # sampling.
        optimizer = RhoOptimizer(temperature=proposal_temperature)
    elif proposal_temperature is not None and hasattr(optimizer, "temperature"):
        optimizer.temperature = proposal_temperature  # type: ignore[attr-defined]

    if preference_judge is None:
        from agent_evolve.adapters.cuga_preference_judge import PreferenceJudge

        preference_judge = PreferenceJudge()

    tasks_by_id = {task.task_id: task for task in stack.tasks}

    # -- phase 1 ---------------------------------------------------------- #
    def load_history_hook() -> "HistoryLoadReport":
        if history_root is None:
            return HistoryLoadReport()
        return load_history(Path(history_root))

    # -- phase 3 ---------------------------------------------------------- #
    def judge_hook(record: object, summary_text: str) -> object:
        expected = (
            expected_answer_for(getattr(record, "task_id", ""))
            if expected_answer_for is not None
            else None
        )
        judge_call = getattr(difficulty_judge, "judge")
        return judge_call(record, summary_text, expected_answer=expected)

    # -- phase 4 ---------------------------------------------------------- #
    def task_for(task_id: str) -> EvolutionTask | None:
        return tasks_by_id.get(task_id)

    # -- phases 5 and 8 --------------------------------------------------- #
    def resolve_version(version: str) -> str:
        """Map the round's fixed base label onto this stack's incumbent."""
        return stack.base_version if version == BASE_VERSION else version

    def rollout(
        version: str, task: EvolutionTask, index: int
    ) -> RolloutOutcome:
        resolved = resolve_version(version)
        prefix = f"rho-{version}-{index}"
        try:
            outcomes = stack.runner._execute_rollouts(
                resolved, (task,), prefix=prefix
            )
        except Exception as exc:  # noqa: BLE001 - a failure is data, never a raise
            return RolloutOutcome(
                task=task, trace=None, error=f"{type(exc).__name__}: {exc}"
            )
        if not outcomes:
            return RolloutOutcome(
                task=task, trace=None, error="the rollout batch returned nothing"
            )
        return _restamped(outcomes[0], resolved)

    # -- phase 7 ---------------------------------------------------------- #
    def base_artifacts() -> Mapping[str, str]:
        version = stack.base_version
        ids = tuple(
            d.artifact_id
            for d in adapter.artifact_inventory(version)  # type: ignore[attr-defined]
        )
        return dict(adapter.read_artifacts(version, ids))  # type: ignore[attr-defined]

    def register_candidate(proposed: object) -> str:
        """Materialize a proposal as a rollout-able version.

        Two paths, one contract. ``CugaAdapter`` exposes ``register_candidate``,
        which is the seeding seam RHO was built for and which validates every
        artifact id against a CUGA harness slot. An adapter without it (the
        offline fake) goes through the neutral contract instead, so a dry run
        rehearses this same code path rather than a toy.
        """
        state.registrations += 1
        artifacts = dict(getattr(proposed, "artifacts", {}) or {})
        index = int(getattr(proposed, "candidate_index", state.registrations))
        attempt_id = f"rho-{state.registrations:03d}-c{index}"

        seeder = getattr(adapter, "register_candidate", None)
        if seeder is not None:
            version = f"rho-cand-{state.registrations:03d}-c{index}"
            seeder(version, artifacts)
            return version

        base = dict(base_artifacts())
        workspace = adapter.materialize_candidate(  # type: ignore[attr-defined]
            stack.base_version, attempt_id
        )
        edits = [
            ArtifactEdit(
                artifact_id=artifact_id,
                operation="replace" if artifact_id in base else "create",
                payload={"content": content},
            )
            for artifact_id, content in sorted(artifacts.items())
        ]
        if edits:
            adapter.apply_structured_edits(workspace, edits)  # type: ignore[attr-defined]
        return workspace.version

    # -- optional: grader scores ------------------------------------------ #
    def pool_id_for(version: str) -> str:
        """Resolve a *version* to the pool's *candidate id*.

        The two are not the same and conflating them is silent: the offline
        stack's base is version ``base-v0`` under candidate id ``base``, so
        writing evidence keyed by version would raise for the base and, worse,
        would leave the incumbent's cells empty in any pool that tolerated it.
        """
        return pool.base_id if version == stack.base_version else version

    def score(task: EvolutionTask, trace: ExecutionTrace) -> float:
        result = scorer.score_rollout(task, trace)
        value = float(result.score)
        version = trace.candidate_id or stack.base_version
        if not result.scorable:
            # No measurement is not a zero. It is counted and excluded from the
            # pool tensor; the entropy tracker still sees ``value`` because the
            # hook signature has nowhere to say "unmeasured".
            state.unscorable += 1
            return value
        cell_key = (task.task_id, rho_cluster_id(task.task_id))
        if version == stack.base_version:
            # The base is already in the pool, so its evidence lands directly.
            # Without base cells the incumbent's champion coverage is zero and a
            # candidate would win selection on coverage alone.
            _record_pool_score(pool, pool_id_for(version), cell_key, value)
        else:
            state.pending_scores.setdefault(version, {}).setdefault(
                cell_key, []
            ).append(value)
        return value

    # -- phase 10 --------------------------------------------------------- #
    def commit(evidence: object) -> None:
        version = str(getattr(evidence, "version", ""))
        artifacts = dict(getattr(evidence, "artifacts", {}) or {})
        try:
            hashes = {
                d.artifact_id: d.version_hash
                for d in adapter.artifact_inventory(version)  # type: ignore[attr-defined]
            }
        except Exception:  # noqa: BLE001 - an adapter without this version
            hashes = {
                artifact_id: f"sha256:{_sha256_text(content)}"
                for artifact_id, content in artifacts.items()
            }
        pool.add_candidate(
            EvolutionCandidate(
                candidate_id=version,
                version=version,
                artifact_hashes=hashes,
                parent_ids=(pool.base_id,),
                ancestor_ids=(pool.base_id,),
            ),
            origin_attempt_ids=(version,),
        )
        state.committed_versions = state.committed_versions + (version,)
        # Carry the pairwise verdict across the commit boundary (SV-4). Before
        # this, `mean_preference` was computed, printed, and dropped here: the
        # pool had no field for it and `select_champion` could not receive it, so
        # the paper's acceptance signal was bought on every round and discarded.
        available = int(getattr(evidence, "preferences_available", 0) or 0)
        pool.record_preference(
            version,
            float(getattr(evidence, "mean_preference", 0.0) or 0.0),
            available=available,
            unavailable=int(getattr(evidence, "preferences_unavailable", 0) or 0),
        )
        for cell_key, values in state.pending_scores.pop(version, {}).items():
            for value in values:
                _record_pool_score(pool, version, cell_key, value)

    # -- optional: the genetic phase, coreset tasks only ------------------ #
    def run_genetic(tasks: Sequence[EvolutionTask], iterations: int) -> None:
        """Run the existing genetic loop over ``tasks`` only.

        The task set is narrowed rather than the loop reimplemented, so the
        genetic phase is byte-for-byte the loop that produced the measured
        baseline. It is restored in ``finally``: leaving the stack narrowed
        would silently shrink the final champion measurement to the coreset.
        """
        if iterations < 1:
            return
        original = stack.tasks
        stack.tasks = tuple(tasks)
        try:
            stack.run_iterations(iterations)
        finally:
            stack.tasks = original

    def cache_hits() -> Mapping[str, int]:
        return {
            "summary": int(getattr(summary_cache, "hits", 0)),
            "difficulty": int(getattr(difficulty_cache, "hits", 0)),
            # Reported for completeness; nothing reads it yet because
            # ``run_round`` selects the coreset without an embedder.
            "embedding": int(getattr(embedding_cache, "hits", 0)),
        }

    return RhoHooks(
        load_history=load_history_hook,
        comprehend=comprehender.comprehend,  # type: ignore[union-attr]
        judge=judge_hook,
        task_for=task_for,
        rollout=rollout,
        diagnose=diagnoser.diagnose,  # type: ignore[union-attr]
        base_artifacts=base_artifacts,
        propose=optimizer.propose,  # type: ignore[union-attr]
        register_candidate=register_candidate,
        # Symmetric on purpose: position bias is the dominant systematic error
        # of an LLM preference judge, and it is selection-critical here.
        compare=preference_judge.compare_symmetric,  # type: ignore[union-attr]
        commit=commit,
        pool_size=stack.pool_size,
        score=score,
        run_genetic=run_genetic,
        cache_hits=cache_hits,
        # The DPP diversity term. Without this the coreset degrades to a plain
        # difficulty ranking, which silently discards half the selection design.
        embedder=(
            _CachingEmbedder(stack.runner.embedder, embedding_cache)
            if stack.runner.embedder is not None
            else None
        ),
        # Never populated: the only way to fill it is to hand this function an
        # answer key, which would then travel into a manifest.
        contamination_literals=(),
    )


class _CachingEmbedder:
    """Wrap a :class:`MechanismEmbedder` with a content-hash disk cache.

    Fingerprint text is stable across rounds, so re-embedding it every round is
    pure waste. ``EmbeddingProviderUnavailable`` is deliberately NOT caught --
    ``select_coreset`` catches that sentinel itself to trigger its documented
    quality-only fallback, and swallowing it here would make a degraded run
    indistinguishable from a healthy one.
    """

    def __init__(self, inner: "MechanismEmbedder", cache: "JsonDiskCache") -> None:
        self._inner = inner
        self._cache = cache
        self.dim = int(inner.dim)

    def embed(self, text: str) -> tuple[float, ...]:
        key = f"embed:{self.dim}:{_sha256_text(text)}"
        cached = self._cache.get(key)
        if cached is not None:
            vector = cached.get("vector")
            if isinstance(vector, list) and len(vector) == self.dim:
                return tuple(float(v) for v in vector)
        vector_out = self._inner.embed(text)
        # JsonDiskCache stores dicts only, so the vector must be wrapped.
        self._cache.put(key, {"vector": list(vector_out)})
        return vector_out


def _sha256_text(content: str) -> str:
    from hashlib import sha256

    return sha256(content.encode("utf-8")).hexdigest()


def _restamped(outcome: RolloutOutcome, version: str) -> RolloutOutcome:
    """Force the trace's ``candidate_id`` to the version that was rolled out.

    Not cosmetic. An adapter is free to stamp its own per-attempt workspace name
    (``FakeAdapter`` returns ``base-v0+rho-...``), and the RHO round keys every
    downstream decision on ``trace.candidate_id``: the score hook uses it to
    decide which pool entry to credit, so a workspace-shaped id would file the
    base's own evidence under a candidate that is not in the pool, and the
    entropy cell for the incumbent would stay empty no matter how many rollouts
    were spent.
    """
    trace = outcome.trace
    if trace is None or trace.candidate_id == version:
        return outcome
    return RolloutOutcome(
        task=outcome.task,
        trace=ExecutionTrace(
            trace_id=trace.trace_id,
            candidate_id=version,
            task_id=trace.task_id,
            events=trace.events,
            final_output=trace.final_output,
            status=trace.status,
            checkpoint_ids=trace.checkpoint_ids,
        ),
    )


def _record_pool_score(
    pool: PersistentPool,
    candidate_id: str,
    cell_key: tuple[str, str],
    value: float,
) -> None:
    """Append one measured score to a pool cell with RHO provenance.

    ``rollout_seq`` must be the cell's next slot or ``ScoreCell.add`` refuses
    the write, so it is read from the cell rather than tracked separately.
    """
    task_id, cluster_id = cell_key
    cell = pool.get(candidate_id).cell(task_id, cluster_id)
    pool.record_score(
        candidate_id,
        max(0.0, min(1.0, float(value))),
        ScoreProvenance(
            task_id=task_id,
            mechanism_cluster_id=cluster_id,
            trace_id=f"rho:{candidate_id}:{task_id}:{cell.rollout_count}",
            rollout_seq=cell.rollout_count,
            analyzer_model_id=RHO_ANALYZER_MODEL_ID,
            judge_model_id=RHO_JUDGE_MODEL_ID,
            # A RHO round performs no causal blame attribution, so claiming any
            # blame confidence would be inventing evidence. Zero is the honest
            # value; neither field participates in the weighted cell score.
            blame_confidence=0.0,
            blame_stability=0.0,
        ),
    )


# --------------------------------------------------------------------------- #
# Exporting an evolved harness
# --------------------------------------------------------------------------- #
#: File written for the selected champion inside an export directory. Fixed so
#: an operator can wire the next run's ``--harness`` without re-deriving
#: selection from the per-candidate files.
CHAMPION_FILENAME = "champion.json"

#: Prefix for the per-candidate files. Every pool member is exported, not only
#: the winner: with RHO seeding the frontier is what the next run seeds from,
#: and a sibling discarded here cost real rollouts to produce.
CANDIDATE_FILENAME_PREFIX = "candidate-"

#: Marks an exported harness so a later reader can tell an evolved artifact set
#: from a hand-written one. ``HarnessVersion.from_path`` reads only its named
#: keys via ``raw.get`` and ignores the rest, so this and ``provenance`` travel
#: inside the harness file without stopping ``--harness`` from loading it.
EXPORT_FORMAT = "agent-evolve-harness-v1"


def harness_version_name(candidate_id: str) -> str:
    """The ``version`` string an exported candidate declares.

    ``HarnessVersion.from_path`` refuses to guess a version from the filename,
    and should: the version is stamped onto every trace the next run writes and
    is the only way to attribute a later result back to this candidate. Renaming
    the file therefore cannot change what the harness claims to be.
    """
    text = str(candidate_id).strip()
    if not text:
        raise ValueError(
            "cannot export a harness for an unnamed candidate: the version is "
            "stamped onto every trace and will not be guessed"
        )
    return f"evolved-{text}"


def _safe_filename_part(text: str) -> str:
    """Flatten a candidate id into one path segment.

    Candidate versions contain ``:`` and ``/`` (``base:att-1``,
    ``skills/x``), either of which would silently write outside the export
    directory or fail to open.
    """
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in text)


def harness_payload(
    adapter: object,
    *,
    version: str,
    candidate_id: str,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Invert :func:`_harness_artifacts` back into a ``--harness`` JSON payload.

    The exact inverse of the forward mapping: an artifact id the adapter holds
    becomes the harness key it came from. Artifacts are read through the neutral
    adapter contract (``artifact_inventory`` + ``read_artifacts``) so the offline
    rehearsal exercises this same path.

    An artifact id with no CUGA harness slot cannot be expressed in a file
    ``--harness`` loads. Such an id is never quietly dropped and never
    reinterpreted as a skill -- inventing a slot would ship a harness that is not
    the one the run measured -- so it is preserved verbatim under
    ``provenance.unexported_artifacts`` instead, where it is recoverable but
    cannot be mistaken for something the agent loaded.
    """
    artifact_ids = tuple(
        d.artifact_id
        for d in adapter.artifact_inventory(version)  # type: ignore[attr-defined]
    )
    artifacts = adapter.read_artifacts(version, artifact_ids)  # type: ignore[attr-defined]

    payload: dict[str, object] = {"version": harness_version_name(candidate_id)}
    groups: dict[str, dict[str, str]] = {}
    unexported: dict[str, str] = {}
    for artifact_id in sorted(artifacts):
        content = artifacts[artifact_id]
        try:
            key, member = CugaAdapter._harness_slot(artifact_id)
        except ValueError:
            unexported[artifact_id] = content
            continue
        if member is None:
            payload[key] = content
        else:
            groups.setdefault(key, {})[member] = content
    payload.update(groups)
    payload["export_format"] = EXPORT_FORMAT
    record: dict[str, object] = {
        "candidate_id": candidate_id,
        "candidate_version": version,
        **dict(provenance or {}),
    }
    if unexported:
        record["unexported_artifacts"] = unexported
    payload["provenance"] = record
    return payload


def export_harness(
    adapter: object,
    *,
    version: str,
    candidate_id: str,
    path: Path,
    provenance: Mapping[str, object] | None = None,
) -> Path:
    """Write one candidate's artifact set as a loadable harness JSON file.

    Without this, a finished run left nothing behind but a pass rate on stdout:
    the adapter holds candidate artifacts in memory only, so the improved
    harness died with the process and its delta was unreproducible.
    """
    payload = harness_payload(
        adapter, version=version, candidate_id=candidate_id, provenance=provenance
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def nothing_accepted_warning(task_count: int) -> str:
    """The diagnostic for a run that accepted nothing.

    An earlier version of this text told the operator that a run above one task
    could not accept an edit regardless of edit quality, because
    ``weighted_net_gain`` charged -1.0 for every *passing* regression probe. That
    defect is fixed -- a passing probe now costs exactly nothing, and only a
    failing one is charged ``1 - score`` (pinned by
    ``tests/test_editor.py::test_passing_regression_probes_are_free_at_every_probe_count``)
    -- so the claim is false and the ``--tasks 1`` advice sent operators away
    from the real cause.

    The warning stays loud, because a silently inert run is still worse than a
    loud one; it now lists causes that can actually be checked.
    """
    return (
        f"\nwarning: nothing was accepted across {task_count} tasks. This is a "
        "real outcome, not an arithmetic floor -- acceptance is reachable at any "
        "task count. Check, in order:\n"
        "  - no issue was attributed: analysis found no blamable artifact, so no "
        "attempt was made (look for no_issue=1 in the iteration lines above)\n"
        "  - the editor declined or produced no valid plan (an unauthorized "
        "write or an empty edit set)\n"
        "  - validation rejected the edit: a genuine regression, a failed origin "
        "or worked probe, or a protected-floor violation\n"
        "  - the retry budget was exhausted before a plan validated\n"
        "Re-run with --capture-logs to get the analyzer, editor and pipeline "
        "records that name which of these happened."
    )


def nothing_accepted_warning_applies(
    task_count: int, accepted_any: bool
) -> bool:
    """Whether the run should print :func:`nothing_accepted_warning`."""
    return task_count >= 1 and not accepted_any


def _entry_provenance(entry: object, *, is_champion: bool) -> dict[str, object]:
    """Lineage and scores for one pool entry, for the exported file's record."""
    candidate = getattr(entry, "candidate")
    scored = [
        cell.mean
        for cell in getattr(entry, "score_tensor", {}).values()
        if cell.rollout_count
    ]
    return {
        "is_champion": is_champion,
        "is_base": bool(getattr(entry, "is_base", False)),
        "parent_ids": list(getattr(candidate, "parent_ids", ())),
        "ancestor_ids": list(getattr(candidate, "ancestor_ids", ())),
        "attempt_ids": list(getattr(candidate, "attempt_ids", ())),
        "origin_attempt_ids": list(getattr(entry, "origin_attempt_ids", ())),
        "scored_cells": len(scored),
        "mean_score": (sum(scored) / len(scored)) if scored else None,
    }

