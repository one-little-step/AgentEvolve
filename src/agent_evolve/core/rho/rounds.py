"""RHO round configuration, phase sequencing, mode dispatch, and execution.

One outer loop; the mode selects which phases run per outer iteration:

  rho           the 10 RHO phases, repeated ``rounds`` times
  genetic       the existing mutation/crossover loop, unchanged
  rho-genetic   [RHO round -> genetic iterations] x ``rounds``

Pool, export, run logging, budgets, and the adapter are shared across all three.

**Agent neutrality.** This module lives in ``core/`` and therefore may not import
``cuga``, ``litellm``, or anything from ``agent_evolve.adapters``. Every model
call, agent invocation, and rollout arrives as an injected callable on
:class:`RhoHooks`. Everything the round does with those callables -- ordering,
budget accounting, the ``observed``/``available`` gates, entropy promotion, pool
commit -- is deterministic Python that a test can drive entirely offline.

**All N candidates are retained.** The paper takes best-of-N and discards the
rest; we keep every proposal. That is the point: N distinct harness hypotheses
become the parents whose disagreement the genetic stage exploits. Preference rank
determines reported ordering and champion selection only, never survival.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping, Sequence

from agent_evolve.core.contamination import scan_artifacts
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace
from agent_evolve.core.entropy import EntropyTracker
from agent_evolve.core.evaluation import RolloutOutcome
from agent_evolve.core.clustering import MechanismEmbedder
from agent_evolve.core.rho.coreset import candidates_from_verdicts, select_coreset
from agent_evolve.core.rho.history import HistoricalRecord, HistoryLoadReport
from agent_evolve.core.rho.scheduler import ConcurrencyPlan, run_groups

__all__ = [
    "BASE_VERSION",
    "CandidateEvidence",
    "PHASES",
    "RhoHooks",
    "RhoMode",
    "RoundConfig",
    "RoundSummary",
    "phases_for",
    "rho_cluster_id",
    "run_round",
    "run_rounds",
]

RhoMode = Literal["rho", "genetic", "rho-genetic"]

#: The pool/version id the round rolls the incumbent harness out under.
BASE_VERSION = "base"

_RHO_PHASES: tuple[str, ...] = (
    "history_load",
    "trajectory_comprehension",
    "difficulty_fingerprint",
    "coreset_selection",
    "group_rollouts",
    "group_diagnosis",
    "candidate_proposal",
    "candidate_rollouts",
    "preference_judging",
    "pool_commit",
)

PHASES: dict[str, tuple[str, ...]] = {
    "rho": _RHO_PHASES,
    "genetic": ("genetic_iterations",),
    "rho-genetic": _RHO_PHASES + ("genetic_iterations",),
}


def phases_for(mode: str) -> tuple[str, ...]:
    """Return the ordered phase sequence for ``mode``."""
    try:
        return PHASES[mode]
    except KeyError:
        raise ValueError(
            f"unknown mode: {mode!r}; expected one of {sorted(PHASES)}"
        ) from None


def rho_cluster_id(task_id: str) -> str:
    """The task-local mechanism cluster a RHO round records evidence into.

    Deliberately derived from the task id alone, not from diagnosis text. Base
    rollouts happen in phase 5, before any diagnosis exists (phase 6), so a
    diagnosis-derived cluster id would file base evidence in a different cell
    from candidate evidence and ``min_comparable_candidates`` could never be met.
    A task-local cell keeps base and all N candidates comparable, which is the
    whole basis of cross-candidate variance.
    """
    return f"rho-task:{task_id}"


# ---------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RoundConfig:
    """Validated RHO round configuration.

    Defaults follow the paper: k=10, G=3, N=3, plus R=2.
    """

    mode: RhoMode
    rounds: int
    coreset_size: int
    group_rollouts: int
    candidates: int
    concurrency: ConcurrencyPlan
    #: R: rollouts per candidate per coreset task. Default 2 so each candidate's
    #: cell clears ``EntropyTracker.min_rollouts_per_candidate=2`` naturally,
    #: without weakening the evidence floor. See the Task 11 note: the floor is
    #: met by spending a second rollout, not by deleting the guard.
    candidate_rollouts: int = 2
    selector: str = "dpp"
    genetic_iterations_per_round: int = 0

    def __post_init__(self) -> None:
        if self.mode not in PHASES:
            raise ValueError(
                f"unknown mode: {self.mode!r}; expected one of {sorted(PHASES)}"
            )
        for name in (
            "rounds",
            "coreset_size",
            "group_rollouts",
            "candidates",
            "candidate_rollouts",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.genetic_iterations_per_round < 0:
            raise ValueError("genetic_iterations_per_round must be >= 0")

    @property
    def rollouts_per_round(self) -> int:
        """``k*G`` baseline plus ``k*N*R`` candidate rollouts.

        At k=10, G=3, N=3, R=2 that is 30 + 60 = 90 rollouts per round. R is the
        price of clearing the entropy evidence floor honestly rather than by
        deleting the guard.
        """
        return self.coreset_size * (
            self.group_rollouts + self.candidates * self.candidate_rollouts
        )

    @property
    def phases(self) -> tuple[str, ...]:
        return phases_for(self.mode)


# ---------------------------------------------------------------------- #
# Injected adapter boundary
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RhoHooks:
    """Every adapter the round needs, injected.

    ``core/`` may not import ``adapters/``, so each field is an ordinary
    callable whose argument types are either core contracts or duck-typed
    adapter results. The duck-typed returns need only the attributes named below:

    ``load_history()`` -> :class:`HistoryLoadReport`.

    ``comprehend(record)`` -> object with ``task_id``, ``observed``,
    ``embedding_text``.

    ``judge(record, summary_text)`` -> object with ``task_id``, ``difficulty``,
    ``abstract_fingerprint``, ``observed``.

    ``task_for(task_id)`` -> :class:`EvolutionTask` or ``None`` when the coreset
    named a task the current benchmark does not carry.

    ``rollout(version, task, index)`` -> :class:`RolloutOutcome`. Must not raise;
    a failure is ``RolloutOutcome(trace=None, error=...)``. ``index`` lets a
    caller vary a seed or a trace path per rollout in a group.

    ``diagnose(task_id, task_input, traces)`` -> object with ``task_id`` and
    ``observed``.

    ``base_artifacts()`` -> the incumbent harness's complete artifact mapping.

    ``propose(base_artifacts, diagnoses, n)`` -> object with ``candidates``,
    ``requested``, ``discarded``; each candidate has ``candidate_index``,
    ``artifacts``, ``observed``.

    ``register_candidate(proposed)`` -> the version/candidate id the rollout hook
    will accept for that proposal.

    ``compare(task, baseline_trace, candidate_trace)`` -> object with ``score``
    and ``available``. Prefer a symmetric comparison at the call site: position
    bias is not this module's business but it is selection-critical.

    ``commit(evidence)`` -> append the candidate to the persistent pool. Called
    for **every** candidate; never conditioned on rank.

    Optional hooks:

    ``pool_size()`` -> reported pool size after commit.

    ``score(task, trace)`` -> a grader score in ``[0, 1]``. Supplied only when
    the caller wants entropy cells populated; ``None`` means no tracker writes.

    ``run_genetic(tasks, iterations)`` -> run the existing genetic loop. In
    ``rho-genetic`` it receives the **coreset tasks only**: cross-candidate
    variance needs a populated ``(task, mechanism)`` cell, and after a RHO round
    cells exist only for the coreset. Off the coreset variance is *undefined*,
    not low.

    ``cache_hits()`` -> Interface A cache counters for the summary.
    ``embedder`` -> a :class:`MechanismEmbedder` for the DPP diversity term.
    When absent, ``select_coreset`` degrades to quality-only ordering and says
    so in ``fallback_reason`` -- the coreset is then difficulty-ranked, with no
    diversity pressure at all.

    ``contamination_literals`` -> answer-key literals for the post-hoc scan.
    Observational only: the scan reports artifact ids and confidences, blocks
    nothing, and mutates nothing. Never persist the literals themselves.
    """

    load_history: Callable[[], HistoryLoadReport] | None = None
    comprehend: Callable[[HistoricalRecord], object] | None = None
    judge: Callable[[HistoricalRecord, str], object] | None = None
    task_for: Callable[[str], EvolutionTask | None] | None = None
    rollout: Callable[[str, EvolutionTask, int], RolloutOutcome] | None = None
    diagnose: (
        Callable[[str, str, Sequence[ExecutionTrace]], object] | None
    ) = None
    base_artifacts: Callable[[], Mapping[str, str]] | None = None
    propose: (
        Callable[[Mapping[str, str], Sequence[object], int], object] | None
    ) = None
    register_candidate: Callable[[object], str] | None = None
    compare: (
        Callable[[EvolutionTask, ExecutionTrace, ExecutionTrace], object] | None
    ) = None
    commit: Callable[["CandidateEvidence"], None] | None = None
    pool_size: Callable[[], int] | None = None
    score: Callable[[EvolutionTask, ExecutionTrace], float] | None = None
    run_genetic: Callable[[Sequence[EvolutionTask], int], None] | None = None
    cache_hits: Callable[[], Mapping[str, int]] | None = None
    embedder: MechanismEmbedder | None = None
    contamination_literals: tuple[str, ...] = ()

    def require(self, name: str) -> Callable[..., object]:
        """Return hook ``name``, naming it in the error when it is absent."""
        hook = getattr(self, name, None)
        if hook is None:
            raise ValueError(
                f"RhoHooks.{name} is required for this phase but was not supplied"
            )
        return hook


# ---------------------------------------------------------------------- #
# Results
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """One proposed candidate plus the evidence gathered about it.

    Handed to ``commit`` for every candidate. ``mean_preference`` averages only
    *available* verdicts; an unavailable verdict is absent evidence and is never
    folded in as a tie.
    """

    candidate_index: int
    version: str
    artifacts: Mapping[str, str]
    rollouts: int = 0
    rollout_failures: int = 0
    mean_preference: float = 0.0
    preferences_available: int = 0
    preferences_unavailable: int = 0
    task_scores: Mapping[str, float] = field(default_factory=dict)

    @property
    def decided(self) -> bool:
        """True when at least one available verdict backs ``mean_preference``."""
        return self.preferences_available > 0


@dataclass(frozen=True, slots=True)
class RoundSummary:
    """One round's reportable outcome."""

    round_index: int
    selection_method: str
    coreset_ids: tuple[str, ...]
    candidates_requested: int
    candidates_distinct: int
    discarded: tuple[tuple[int, str], ...]
    cache_hits: Mapping[str, int]
    pool_size: int
    mode: str = "rho"
    phases_run: tuple[str, ...] = ()
    rollouts_spent: int = 0
    rollout_failures: int = 0
    diagnoses_observed: int = 0
    preference_mean: float = 0.0
    preferences_available: int = 0
    preferences_unavailable: int = 0
    evidence: tuple[CandidateEvidence, ...] = ()
    contamination: tuple[tuple[int, str, str], ...] = ()
    genetic_iterations: int = 0
    notes: tuple[str, ...] = ()

    @property
    def collapsed(self) -> bool:
        """True when fewer distinct candidates survived than were requested.

        Surfaced because a collapse to one candidate means the pairwise judge is
        comparing a harness against itself, which must never be silent.
        """
        return self.candidates_distinct < self.candidates_requested

    def line(self) -> str:
        return (
            f"round {self.round_index}: {len(self.coreset_ids)} coreset tasks "
            f"({self.selection_method}), candidates "
            f"{self.candidates_distinct} of {self.candidates_requested} distinct, "
            f"pool {self.pool_size}"
        )


# ---------------------------------------------------------------------- #
# Execution
# ---------------------------------------------------------------------- #
def run_rounds(
    config: RoundConfig,
    hooks: RhoHooks,
    *,
    tracker: EntropyTracker | None = None,
) -> tuple[RoundSummary, ...]:
    """Run ``config.rounds`` outer iterations, one summary each."""
    return tuple(
        run_round(config, hooks, round_index=index, tracker=tracker)
        for index in range(1, config.rounds + 1)
    )


def run_round(
    config: RoundConfig,
    hooks: RhoHooks,
    *,
    round_index: int = 1,
    tracker: EntropyTracker | None = None,
) -> RoundSummary:
    """Run one outer iteration: the phases ``config.mode`` selects.

    Never raises for adapter-level failure -- a failed rollout, an unobserved
    diagnosis, and an unavailable preference verdict are all data. It *does*
    raise ``ValueError`` when a hook a selected phase needs was not injected,
    because that is a wiring error and discovering it after 90 rollouts would be
    worse than discovering it now.
    """
    phases = config.phases
    notes: list[str] = []

    if "history_load" not in phases:
        # genetic mode: nothing RHO-shaped runs at all.
        iterations = _run_genetic(hooks, (), config.genetic_iterations_per_round)
        return RoundSummary(
            round_index=round_index,
            selection_method="",
            coreset_ids=(),
            candidates_requested=0,
            candidates_distinct=0,
            discarded=(),
            cache_hits=_cache_hits(hooks),
            pool_size=_pool_size(hooks),
            mode=config.mode,
            phases_run=phases,
            genetic_iterations=iterations,
        )

    # -- phase 1: history load ----------------------------------------- #
    report = hooks.require("load_history")()
    assert isinstance(report, HistoryLoadReport)
    if report.rejected:
        notes.append(f"{len(report.rejected)} historical traces rejected")
    if report.is_cold_start:
        notes.append(
            "cold start: no usable historical traces, RHO phases skipped this round"
        )
        return RoundSummary(
            round_index=round_index,
            selection_method="",
            coreset_ids=(),
            candidates_requested=0,
            candidates_distinct=0,
            discarded=(),
            cache_hits=_cache_hits(hooks),
            pool_size=_pool_size(hooks),
            mode=config.mode,
            phases_run=phases,
            notes=tuple(notes),
        )

    # -- phase 2: trajectory comprehension ----------------------------- #
    comprehend = hooks.require("comprehend")
    summaries: dict[str, str] = {}
    unobserved_summary: set[str] = set()
    for record in report.records:
        summary = comprehend(record)
        text = str(getattr(summary, "embedding_text", "") or "")
        if getattr(summary, "observed", False):
            summaries[record.task_id] = text
        else:
            unobserved_summary.add(record.task_id)
    if unobserved_summary:
        notes.append(f"{len(unobserved_summary)} trajectory summaries unavailable")

    # -- phase 3: difficulty + fingerprint ----------------------------- #
    judge = hooks.require("judge")
    verdicts = []
    for record in report.records:
        if record.task_id in unobserved_summary:
            # No summary means the judge has nothing abstract to reason over.
            # Absent evidence, not an easy task: it must not compete.
            continue
        verdicts.append(judge(record, summaries.get(record.task_id, "")))

    # -- phase 4: coreset selection ------------------------------------ #
    coreset = select_coreset(
        candidates_from_verdicts(verdicts, summaries),
        config.coreset_size,
        selector=config.selector,
        embedder=hooks.embedder,
    )
    if coreset.fallback_reason:
        notes.append(f"coreset fallback: {coreset.fallback_reason}")

    task_for = hooks.require("task_for")
    tasks: list[EvolutionTask] = []
    for task_id in coreset.selected_ids:
        task = task_for(task_id)
        if task is None:
            notes.append(f"coreset task {task_id} is absent from the benchmark")
            continue
        tasks.append(task)  # type: ignore[arg-type]

    if not tasks:
        notes.append("no coreset task resolved to a runnable task")
        return RoundSummary(
            round_index=round_index,
            selection_method=coreset.selection_method,
            coreset_ids=(),
            candidates_requested=0,
            candidates_distinct=0,
            discarded=(),
            cache_hits=_cache_hits(hooks),
            pool_size=_pool_size(hooks),
            mode=config.mode,
            phases_run=phases,
            notes=tuple(notes),
        )

    coreset_ids = tuple(task.task_id for task in tasks)

    # -- phase 5: group rollouts (k x G on the incumbent) -------------- #
    base_groups, base_failures = _rollout_grid(
        hooks, BASE_VERSION, tasks, config.group_rollouts, config.concurrency
    )
    rollouts_spent = len(tasks) * config.group_rollouts
    rollout_failures = base_failures
    _record_scores(hooks, tracker, BASE_VERSION, tasks, base_groups)

    # -- phase 6: group diagnosis (1 per task with usable traces) ------ #
    diagnose = hooks.require("diagnose")
    diagnoses: list[object] = []
    for task in tasks:
        traces = base_groups.get(task.task_id, ())
        if not traces:
            notes.append(f"task {task.task_id} produced no usable base trace")
            continue
        diagnoses.append(diagnose(task.task_id, task.input_text, traces))
    # The observed gate: an unobserved diagnosis is absent evidence and must not
    # be handed to the optimizer as if it were a finding.
    observed_diagnoses = tuple(
        d for d in diagnoses if getattr(d, "observed", False)
    )
    if len(observed_diagnoses) < len(diagnoses):
        notes.append(
            f"{len(diagnoses) - len(observed_diagnoses)} diagnoses unobserved"
        )

    # -- phase 7: candidate proposal (N independent invocations) ------- #
    base_artifacts = hooks.require("base_artifacts")()
    proposal = hooks.require("propose")(
        base_artifacts, observed_diagnoses, config.candidates
    )
    proposed = tuple(
        c
        for c in getattr(proposal, "candidates", ())
        if getattr(c, "observed", False)
    )
    discarded = tuple(getattr(proposal, "discarded", ()) or ())
    requested = int(getattr(proposal, "requested", config.candidates) or 0)

    register = hooks.require("register_candidate")
    commit = hooks.require("commit")
    compare = hooks.require("compare")

    evidence: list[CandidateEvidence] = []
    available_scores: list[float] = []
    unavailable_total = 0

    for candidate in proposed:
        version = str(register(candidate))

        # -- phase 8: candidate rollouts (k x R for this candidate) ---- #
        cand_groups, cand_failures = _rollout_grid(
            hooks, version, tasks, config.candidate_rollouts, config.concurrency
        )
        rollouts_spent += len(tasks) * config.candidate_rollouts
        rollout_failures += cand_failures
        task_scores = _record_scores(hooks, tracker, version, tasks, cand_groups)

        # -- phase 9: preference judging (one verdict per task) -------- #
        # N x k invocations, not N x k x R: the judge ranks a candidate against
        # the baseline on a task, not one rollout against another.
        cand_available: list[float] = []
        cand_unavailable = 0
        for task in tasks:
            base_traces = base_groups.get(task.task_id, ())
            cand_traces = cand_groups.get(task.task_id, ())
            if not base_traces or not cand_traces:
                cand_unavailable += 1
                continue
            verdict = compare(task, base_traces[0], cand_traces[0])
            if getattr(verdict, "available", False):
                cand_available.append(float(getattr(verdict, "score", 0.0) or 0.0))
            else:
                cand_unavailable += 1
        mean = sum(cand_available) / len(cand_available) if cand_available else 0.0
        available_scores.extend(cand_available)
        unavailable_total += cand_unavailable

        evidence.append(
            CandidateEvidence(
                candidate_index=int(getattr(candidate, "candidate_index", 0)),
                version=version,
                artifacts=dict(getattr(candidate, "artifacts", {}) or {}),
                rollouts=len(tasks) * config.candidate_rollouts,
                rollout_failures=cand_failures,
                mean_preference=mean,
                preferences_available=len(cand_available),
                preferences_unavailable=cand_unavailable,
                task_scores=task_scores,
            )
        )

    # -- phase 10: pool commit (ALL N, never best-of-N) ---------------- #
    # Rank orders the report and picks a champion; it never decides survival.
    for item in sorted(evidence, key=lambda e: e.candidate_index):
        commit(item)

    contamination = _scan_contamination(hooks, evidence)
    if contamination:
        notes.append(
            f"contamination scan flagged {len(contamination)} artifact(s); "
            "observational only, nothing was blocked"
        )

    # -- genetic phase (rho-genetic only), coreset tasks only ---------- #
    genetic_iterations = 0
    if "genetic_iterations" in phases:
        genetic_iterations = _run_genetic(
            hooks, tasks, config.genetic_iterations_per_round
        )

    return RoundSummary(
        round_index=round_index,
        selection_method=coreset.selection_method,
        coreset_ids=coreset_ids,
        candidates_requested=requested,
        candidates_distinct=len(evidence),
        discarded=discarded,
        cache_hits=_cache_hits(hooks),
        pool_size=_pool_size(hooks),
        mode=config.mode,
        phases_run=phases,
        rollouts_spent=rollouts_spent,
        rollout_failures=rollout_failures,
        diagnoses_observed=len(observed_diagnoses),
        preference_mean=(
            sum(available_scores) / len(available_scores)
            if available_scores
            else 0.0
        ),
        preferences_available=len(available_scores),
        preferences_unavailable=unavailable_total,
        evidence=tuple(evidence),
        contamination=contamination,
        genetic_iterations=genetic_iterations,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------- #
# Internals
# ---------------------------------------------------------------------- #
def _rollout_grid(
    hooks: RhoHooks,
    version: str,
    tasks: Sequence[EvolutionTask],
    repeats: int,
    plan: ConcurrencyPlan,
) -> tuple[dict[str, tuple[ExecutionTrace, ...]], int]:
    """Roll ``version`` out ``repeats`` times per task, group-major.

    Returns per-task usable traces plus the failure count. A failed rollout is
    dropped from its group and counted; its siblings survive, because one broken
    rollout must not discard a group's remaining evidence.
    """
    rollout = hooks.require("rollout")
    by_id = {task.task_id: task for task in tasks}
    groups = [
        (task.task_id, tuple(range(repeats))) for task in tasks
    ]

    def run_one(group_id: str, index: int) -> RolloutOutcome:
        return rollout(version, by_id[group_id], index)  # type: ignore[return-value]

    results = run_groups(groups, run_one, plan)

    traces: dict[str, tuple[ExecutionTrace, ...]] = {}
    failures = 0
    for result in results:
        usable: list[ExecutionTrace] = []
        for outcome in result.outcomes:
            trace = getattr(outcome, "trace", None)
            if trace is None:
                failures += 1
                continue
            usable.append(trace)
        failures += result.failures
        traces[result.group_id] = tuple(usable)
    return traces, failures


def _record_scores(
    hooks: RhoHooks,
    tracker: EntropyTracker | None,
    version: str,
    tasks: Sequence[EvolutionTask],
    groups: Mapping[str, Sequence[ExecutionTrace]],
) -> dict[str, float]:
    """Score every usable trace and populate the task-local entropy cell.

    ``mark_comparable`` is the wiring gotcha: ``EntropyTracker`` counts only
    candidates promoted through it, so recording scores alone leaves entropy
    ``None`` no matter how many rollouts were spent. Promotion happens once the
    candidate actually has ``min_rollouts_per_candidate`` scores in the cell --
    promoting a thinner candidate would defeat the floor it exists to enforce.
    """
    if hooks.score is None:
        return {}
    by_id = {task.task_id: task for task in tasks}
    means: dict[str, float] = {}
    for task_id, traces in groups.items():
        task = by_id.get(task_id)
        if task is None or not traces:
            continue
        values = [float(hooks.score(task, trace)) for trace in traces]
        means[task_id] = sum(values) / len(values)
        if tracker is None:
            continue
        cluster = rho_cluster_id(task_id)
        for value in values:
            tracker.record_score(task_id, cluster, version, value)
        if len(values) >= tracker.min_rollouts_per_candidate:
            tracker.mark_comparable(task_id, cluster, version)
    return means


def _scan_contamination(
    hooks: RhoHooks, evidence: Sequence[CandidateEvidence]
) -> tuple[tuple[int, str, str], ...]:
    """Post-hoc, observational GT-literal scan over proposed artifacts."""
    if not hooks.contamination_literals:
        return ()
    hits: list[tuple[int, str, str]] = []
    for item in evidence:
        for hit in scan_artifacts(
            dict(item.artifacts), hooks.contamination_literals
        ):
            # The literal itself is deliberately not carried: reporting it would
            # write an answer key into a log.
            hits.append((item.candidate_index, hit.artifact_id, hit.confidence))
    return tuple(hits)


def _run_genetic(
    hooks: RhoHooks, tasks: Sequence[EvolutionTask], iterations: int
) -> int:
    if iterations < 1:
        return 0
    hooks.require("run_genetic")(tasks, iterations)
    return iterations


def _cache_hits(hooks: RhoHooks) -> Mapping[str, int]:
    if hooks.cache_hits is None:
        return {}
    return dict(hooks.cache_hits())


def _pool_size(hooks: RhoHooks) -> int:
    if hooks.pool_size is None:
        return 0
    return int(hooks.pool_size())
