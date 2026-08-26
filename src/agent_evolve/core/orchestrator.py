"""Top-level evolution orchestrator and profile definitions.

The orchestrator ties together every core module:

* :class:`PersistentPool` for candidate tracking
* :class:`EditMemory` for attempt history + retry budget
* :class:`ClusterRegistry` for task-local mechanism clustering
* :class:`EntropyTracker` for cross-candidate entropy
* :class:`HierarchicalDPPSelector` for issue selection
* :class:`AnalyzerJudge` + :class:`Editor` for blame + edits
* :class:`SnapshotLeaseManager` + :class:`BatchCoordinator` for parallel mode

Profiles
--------
* ``minimal``: persistent pool + outcome Pareto + RHO diagnosis editor,
  sequential execution. No causal blame, no entropy, no parallelism.
* ``research_sequential``: + causal blame, edit memory, focused validation,
  sequential attempts.
* ``research_parallel``: + snapshot/lease batch execution.
* ``full_ablation``: all individual feature controls exposed.

The orchestrator runs one *outer iteration* at a time. The caller decides
how many iterations to run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Callable, Iterable, Mapping, Sequence

from agent_evolve.core.analyzer import (
    AnalyzerJudge,
    FakeAnalyzerJudge,
    PositivityJudge,
    ReportAnalyzerJudge,
    as_legacy_analyzer,
    contract_score,
    is_report_analyzer,
)
from agent_evolve.core.parallel_analysis import (
    AnalysisOutcome,
    ParallelAnalysisRunner,
)
from agent_evolve.core.blame import (
    BlameGraph,
    BlameNode,
    CausalAnalysis,
    CausalFinding,
    abstained_analysis,
    analysis_from_finding,
    empty_analysis,
    is_placeholder_mechanism,
    unanalyzed_analysis,
)
from agent_evolve.core.clustering import (
    ClusterRegistry,
    LexicalEmbedder,
    MechanismClusterer,
    MechanismEmbedder,)
from agent_evolve.core.config import BudgetLimits, BudgetUsage, ResolvedConfig
from agent_evolve.core.correlation import (
    CorrelationContext,
    correlation_scope,
)
from agent_evolve.core.contracts import (
    ArtifactEdit,
    CandidateWorkspace,
    EvolutionAdapter,
    EvolutionCandidate,
    EvolutionTask,
    ExecutionTrace,
)
from agent_evolve.core.editor import (
    ParentContext,
    AcceptanceDecision,
    Editor,
    EditorRequest,
    EditorResponse,
    FocusedValidationReport,
    ProtectedFloor,
    ValidationKind,
    ValidationPlanner,
    ValidationResult,
    build_attempt,
    decide_acceptance,
    lineage_of,
    record_attempt,
    repair_once_then_classify,
)
from agent_evolve.core.entropy import (
    EntropyTracker,
    HierarchicalDPPSelector,
    Issue,
)
from agent_evolve.core.evaluation import (
    ContractScorer,
    ObservedRollout,
    RolloutBatch,
    RolloutOutcome,
    RolloutScore,
    ScoreTally,
    Scorer,
    tally_scores,
)
from agent_evolve.core.evidence import rollout_group_report
from agent_evolve.core.retirement import decide_retirement
from agent_evolve.core.fake_editor import FakeEditor
from agent_evolve.core.issues import (
    DEFAULT_SCORE_FLOOR as TARGET_SCORE_FLOOR,
    DEFAULT_THETA as TARGET_THETA,
    HierarchicalDPPSelector as TargetIssueSelector,
    Issue as TargetIssue,
    IssueSelectionReport as TargetIssueSelectionReport,
    build_issue as build_target_issue,
)
from agent_evolve.core.mechanism_index import IndexEntry, SignedMechanismIndex
from agent_evolve.core.memory import (
    AttemptStatus,
    EditAttempt,
    EditMemory,
    make_attempt_id,
)
from agent_evolve.core.pool import (
    ChampionReport,
    PersistentPool,
    PoolEntry,
    ScoreProvenance,
)
from agent_evolve.core.storage import StorageBackend
from agent_evolve.core.parallel import (
    BatchCoordinator,
    PoolSnapshot,
    SnapshotLeaseManager,
    WorkerResult,
    snapshot_pool,
)


# ---------------------------------------------------------------------- #
# Profile
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Profile:
    """Feature flags for one evolution run."""

    name: str
    use_causal_blame: bool = False
    use_edit_memory: bool = False
    use_focused_validation: bool = False
    use_entropy_selection: bool = False
    use_parallel_batch: bool = False
    base_rollout_group_size: int = 3
    candidate_initial_rollouts: int = 1
    max_attempts_per_issue: int = 3
    net_gain_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.base_rollout_group_size < 1:
            raise ValueError("base_rollout_group_size must be >= 1")
        if self.candidate_initial_rollouts < 1:
            raise ValueError("candidate_initial_rollouts must be >= 1")
        if self.max_attempts_per_issue < 1:
            raise ValueError("max_attempts_per_issue must be >= 1")


MINIMAL = Profile(name="minimal")
RESEARCH_SEQUENTIAL = Profile(
    name="research_sequential",
    use_causal_blame=True,
    use_edit_memory=True,
    use_focused_validation=True,
    use_entropy_selection=False,
    use_parallel_batch=False,
)
RESEARCH_PARALLEL = Profile(
    name="research_parallel",
    use_causal_blame=True,
    use_edit_memory=True,
    use_focused_validation=True,
    use_entropy_selection=True,
    use_parallel_batch=True,
)
FULL_ABLATION = Profile(
    name="full_ablation",
    use_causal_blame=True,
    use_edit_memory=True,
    use_focused_validation=True,
    use_entropy_selection=True,
    use_parallel_batch=True,
)


# ---------------------------------------------------------------------- #
# Iteration result
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class IterationResult:
    """Summary of one outer iteration."""

    iteration: int
    attempts: tuple[EditAttempt, ...]
    accepted: tuple[str, ...]  # attempt_ids
    rejected: tuple[str, ...]
    regressions: tuple[str, ...]
    pool_size: int
    pareto_frontier: tuple[str, ...]


# ---------------------------------------------------------------------- #
# Orchestrator
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class Orchestrator:
    """Drives one RHO-Parallel-GEPA evolution run.

    Construction sets up all the core modules. The caller then invokes
    :meth:`run_iteration` once per outer iteration, passing the task coreset
    for that iteration.
    """

    adapter: EvolutionAdapter
    analyzer_judge: AnalyzerJudge
    editor: Editor
    pool: PersistentPool
    profile: Profile = field(default_factory=lambda: MINIMAL)
    edit_memory: EditMemory = field(default_factory=EditMemory)
    cluster_registry: ClusterRegistry = field(
        default_factory=lambda: ClusterRegistry(
            embedder_factory=lambda: LexicalEmbedder(),
        )
    )
    entropy: EntropyTracker = field(default_factory=EntropyTracker)
    selector: HierarchicalDPPSelector = field(default_factory=HierarchicalDPPSelector)
    protected_floors: tuple[ProtectedFloor, ...] = ()
    #: Run-scoped spend caps. ``None`` means unlimited, which is the historical
    #: behaviour: nothing enforced these before, so a run had no reachable
    #: ceiling on attempts or accepted edits.
    budgets: BudgetLimits | None = None
    #: Run-scoped spend ledger, accumulating ACROSS outer iterations on purpose:
    #: a cap means "for this run", so a per-iteration counter would let N
    #: iterations each spend the full cap.
    _budget: BudgetUsage = field(
        default_factory=BudgetUsage, init=False, repr=False, compare=False
    )
    _iteration: int = 0
    _attempt_seq: int = 0
    #: Every mechanism string that reached clustering, in record order. Kept so a
    #: caller (and the test suite) can assert that a profile without an analyzer
    #: never emits something that looks like a semantic mechanism.
    _observed_mechanisms: list[str] = field(default_factory=list)
    #: Mechanism strings carried by the synthesized base-failure issues.
    _issue_mechanisms: list[str] = field(default_factory=list)
    #: Blamed actor tuples carried by those issues, for the same audit purpose.
    _issue_blame_actors: list[tuple[str, ...]] = field(default_factory=list)
    #: The best-evidence analysis retained per ``(task_id, cluster_id)`` cell from
    #: this iteration's base rollouts, keyed exactly as the score tensor is.
    #: Issue synthesis reads it instead of inventing a stand-in analysis.
    _base_analyses: dict[tuple[str, str], CausalAnalysis] = field(
        default_factory=dict
    )
    _resolved_analyzer: AnalyzerJudge | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # ------------------------------------------------------------------ #
    # Analyzer protocol resolution
    # ------------------------------------------------------------------ #
    @property
    def resolved_analyzer(self) -> AnalyzerJudge:
        """``analyzer_judge`` adapted to the legacy ``(task, trace)`` call sites.

        A legacy analyzer is returned by identity, so existing behaviour is
        byte-identical. A report-based analyzer is wrapped once in a
        :class:`~agent_evolve.core.analyzer.ReportAnalyzerShim`; the wrapper is
        cached because it may hold a stateful analyzer.
        """
        resolved = self._resolved_analyzer
        if resolved is None:
            resolved = as_legacy_analyzer(self.analyzer_judge)
            self._resolved_analyzer = resolved
        return resolved

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def initialize_base(self, base_candidate: EvolutionCandidate) -> PoolEntry:
        """Add the base harness to the pool."""
        if self.pool.base_id if hasattr(self.pool, "_base_id") and self.pool._base_id else None:
            raise ValueError("base already initialized")
        return self.pool.add_base(base_candidate)

    def _next_attempt_id(self) -> str:
        aid = make_attempt_id(self._iteration, self._attempt_seq)
        self._attempt_seq += 1
        return aid

    # ------------------------------------------------------------------ #
    # Rollout + analysis
    # ------------------------------------------------------------------ #
    def _rollout(
        self,
        workspace: CandidateWorkspace,
        task: EvolutionTask,
        rollout_id: str,
    ) -> tuple[ExecutionTrace, CausalAnalysis]:
        result = self.adapter.run_full_rollout(workspace, task, rollout_id)
        trace = self.adapter.capture_trace(result)
        if self.profile.use_causal_blame:
            analysis = self.resolved_analyzer.analyze(task, trace)
        else:
            # Minimal profile: there is no analyzer, so the rollout is *scored*
            # against the task contract and left *undiagnosed*. The mechanism is
            # the reserved UNANALYZED_MECHANISM sentinel rather than a
            # task-derived template: a template like f"failed-to-match-{task_id}"
            # yields one identical string per task, which collapses mechanism
            # clustering to one cluster per task and makes cross-candidate
            # entropy and DPP diversity degenerate -- while still looking like a
            # real mechanism to every downstream consumer.
            score = contract_score(task, trace)
            if score >= 1.0:
                analysis = empty_analysis()
            else:
                # The blamed actor is directly observed (the first actor that
                # acted); only the mechanism is unknown.
                actor = next(
                    (e.actor_id for e in trace.events if e.actor_id),
                    "unknown",
                )
                analysis = unanalyzed_analysis(score=score, actor_id=actor)
        return trace, analysis

    def _record_score(
        self,
        entry: PoolEntry,
        task: EvolutionTask,
        analysis: CausalAnalysis,
        cluster_id: str,
        trace_id: str,
        rollout_seq: int | None = None,
    ) -> None:
        # If rollout_seq is None, use the cell's current rollout_count as
        # the next slot index. This makes the orchestrator robust across
        # multiple iterations on the same cell.
        if rollout_seq is None:
            cell = entry.cell(task.task_id, cluster_id)
            rollout_seq = cell.rollout_count
        self._observed_mechanisms.append(analysis.mechanism)
        prov = ScoreProvenance(
            task_id=task.task_id,
            mechanism_cluster_id=cluster_id,
            trace_id=trace_id,
            rollout_seq=rollout_seq,
            analyzer_model_id=analysis.analyzer_model_id,
            judge_model_id=analysis.judge_model_id,
            blame_confidence=min(1.0, analysis.blame_graph.total_blame()),
            blame_stability=1.0,  # Single-call default; ablations vary this.
            artifact_versions=dict(entry.candidate.artifact_hashes),
        )
        self.pool.record_score(entry.candidate_id, analysis.score, prov)

    # ------------------------------------------------------------------ #
    # One attempt
    # ------------------------------------------------------------------ #
    def _run_attempt(
        self,
        parent_entry: PoolEntry,
        task: EvolutionTask,
        analysis: CausalAnalysis,
        issue_id: str,
        write_set: tuple[str, ...],
    ) -> tuple[EditAttempt, AcceptanceDecision]:
        # 1. Materialize a candidate workspace from the parent.
        attempt_id = self._next_attempt_id()
        workspace = self.adapter.materialize_candidate(
            parent_entry.version, attempt_id
        )

        # 2. Read current artifact contents for the editor.
        current = self.adapter.read_artifacts(parent_entry.version, write_set)

        # 3. Ask the editor for an edit.
        request = EditorRequest(
            base_workspace=workspace,
            task=task,
            analysis=analysis,
            issue_id=issue_id,
            write_set=write_set,
            current_artifacts=dict(current),
            history_refs=tuple(
                a.attempt_id for a in self.edit_memory.for_issue(issue_id)
            ) if self.profile.use_edit_memory else (),
        )
        response = self.editor.propose_edit(request)

        # 4. Apply the edits via the adapter.
        self.adapter.apply_structured_edits(workspace, response.edits)

        # 5. Run focused validation if enabled.
        if self.profile.use_focused_validation:
            report = self._validate(workspace, task, response)
        else:
            # Minimal: just re-roll the origin task once.
            trace, post_analysis = self._rollout(
                workspace, task, f"{attempt_id}-validate"
            )
            passed = post_analysis.score >= 0.5
            report = FocusedValidationReport(
                origin=(ValidationResult(
                    kind=ValidationKind.ORIGIN,
                    task_id=task.task_id,
                    score=post_analysis.score,
                    trace_id=trace.trace_id,
                    passed=passed,
                ),),
                worked=(),
                regression=(),
            )

        # 6. Decide acceptance.
        decision = decide_acceptance(
            report,
            protected_floors=self.protected_floors,
            net_gain_threshold=self.profile.net_gain_threshold,
        )

        # 7. Build the attempt record.
        attempt = build_attempt(
            attempt_id=attempt_id,
            candidate_id=workspace.version,
            issue_id=issue_id,
            response=response,
            evidence_refs=(),
            history_refs=request.history_refs,
            report=report,
            decision=decision,
        )

        # 8. If accepted, add the candidate to the pool.
        if decision.accepted:
            new_candidate = EvolutionCandidate(
                candidate_id=workspace.version,
                version=workspace.version,
                artifact_hashes={
                    d.artifact_id: d.version_hash
                    for d in self.adapter.artifact_inventory(workspace.version)
                },
                parent_ids=(parent_entry.candidate_id,),
                ancestor_ids=parent_entry.candidate.ancestor_ids + (parent_entry.candidate_id,),
                attempt_ids=(attempt_id,),
            )
            self.pool.add_candidate(new_candidate, origin_attempt_ids=(attempt_id,))
            # Record the post-edit score.
            for r in report.all_results:
                cluster_id = issue_id.split(":", 1)[1] if ":" in issue_id else "c0"
                entry = self.pool.get(new_candidate.candidate_id)
                self._record_score(
                    entry,
                    EvolutionTask(
                        task_id=r.task_id,
                        input_text="",
                        expected_contract={},
                    ),
                    # We don't have a fresh analysis per probe; reuse the
                    # origin analysis score for the origin probe, 1.0 for
                    # worked (assumption: passing), 0.0 for failed regression.
                    CausalAnalysis(
                        mechanism=analysis.mechanism,
                        severity=analysis.severity,
                        score=r.score,
                        blame_graph=analysis.blame_graph,
                        analyzer_model_id=analysis.analyzer_model_id,
                        judge_model_id=analysis.judge_model_id,
                    ),
                    cluster_id,
                    r.trace_id,
                )

        # 9. Record in edit memory (if enabled).
        if self.profile.use_edit_memory:
            record_attempt(self.edit_memory, attempt, workspace)

        return attempt, decision

    def _validate(
        self,
        workspace: CandidateWorkspace,
        origin_task: EvolutionTask,
        response: EditorResponse,
    ) -> FocusedValidationReport:
        planner = ValidationPlanner(origin_task=origin_task)
        probes = planner.build_probes()
        results: list[ValidationResult] = []
        for probe in probes:
            trace, analysis = self._rollout(
                workspace, probe.task, f"{workspace.attempt_id}-{probe.kind.value}"
            )
            results.append(ValidationResult(
                kind=probe.kind,
                task_id=probe.task.task_id,
                score=analysis.score,
                trace_id=trace.trace_id,
                passed=analysis.score >= 0.5,
            ))
        # All probes are ORIGIN in the minimal validation planner.
        return FocusedValidationReport(
            origin=tuple(r for r in results if r.kind == ValidationKind.ORIGIN),
            worked=tuple(r for r in results if r.kind == ValidationKind.WORKED),
            regression=tuple(r for r in results if r.kind == ValidationKind.REGRESSION),
            generalization=tuple(r for r in results if r.kind == ValidationKind.GENERALIZATION),
        )

    # ------------------------------------------------------------------ #
    # One outer iteration
    # ------------------------------------------------------------------ #
    def run_iteration(
        self,
        tasks: Sequence[EvolutionTask],
    ) -> IterationResult:
        """Run one outer iteration over the given task coreset.

        For the minimal profile:
        * Roll out the base candidate on each task (G rollouts each).
        * For each task where the base fails, run one editor attempt.
        * Record accepted attempts in the pool.

        For research_sequential:
        * Same, but use causal blame to choose the artifact to edit, and
          use focused validation + retry budget.

        For research_parallel:
        * Same as research_sequential, but batch attempts via the snapshot/
          lease manager + batch coordinator.
        """
        self._iteration += 1
        self.cluster_registry.begin_iteration(self._iteration)
        self.entropy.refresh_at_barrier(self._iteration)
        # Retained analyses are per-iteration evidence. A prior iteration's
        # diagnosis describes a prior version of the artifacts, so carrying it
        # forward would attribute a stale mechanism to a fresh failure.
        self._base_analyses.clear()
        attempts: list[EditAttempt] = []
        accepted: list[str] = []
        rejected: list[str] = []
        regressions: list[str] = []

        # 1. Roll out the base on each task to gather evidence.
        base_entry = self.pool.base
        for task in tasks:
            clusterer = self.cluster_registry.clusterer_for(task.task_id)
            for r in range(self.profile.base_rollout_group_size):
                # Materialize a no-op workspace from the base so the adapter
                # has something to roll out. The base version itself stays
                # read-only; we never apply edits to this workspace.
                base_attempt_id = f"i{self._iteration}-base-{task.task_id}-{r}"
                base_rollout_ws = self.adapter.materialize_candidate(
                    base_entry.version, base_attempt_id
                )
                trace, analysis = self._rollout(
                    base_rollout_ws,
                    task,
                    base_attempt_id,
                )
                if self.profile.use_causal_blame:
                    assignment = clusterer.assign(analysis)
                    cluster_id = assignment.cluster_id
                else:
                    cluster_id = "c0"
                # Retain the analysis for issue synthesis below. The cell is
                # keyed exactly as the score tensor is, and the lowest-scoring
                # rollout is kept: an issue describes a failure, so the rollout
                # that failed hardest is the one worth diagnosing.
                cell_key = (task.task_id, cluster_id)
                previous = self._base_analyses.get(cell_key)
                if previous is None or analysis.score < previous.score:
                    self._base_analyses[cell_key] = analysis
                self._record_score(
                    base_entry, task, analysis, cluster_id, trace.trace_id
                )

        # 2. For each (task, mechanism) where the base scored below 1.0,
        #    run an editor attempt.
        issues: list[tuple[EvolutionTask, str, CausalAnalysis]] = []
        for task in tasks:
            for (t_id, m_id), cell in base_entry.score_tensor.items():
                if t_id != task.task_id:
                    continue
                if cell.max >= 1.0:
                    continue  # No issue; base already succeeds.
                # Reuse the analysis this iteration's base rollouts actually
                # produced for this cell. The old code synthesized
                # f"base-failed-{t_id}-{m_id}" with a hardcoded
                # BlameNode(actor_id="agent") on the premise that "we don't have
                # the trace anymore" -- but the analysis was computed a few lines
                # above and simply discarded. Fabricating a mechanism that no
                # analyzer produced made every base failure on a task collapse to
                # one identical string, and the invented blame node was what the
                # editor selected its target from.
                retained = self._base_analyses.get((t_id, m_id))
                if retained is None:
                    # No analysis was retained for this cell (e.g. it was scored
                    # in an earlier iteration). There is genuinely nothing to
                    # report, so abstain explicitly rather than invent a verdict.
                    issue_analysis = abstained_analysis(
                        "insufficient_evidence",
                        score=cell.max,
                        evidence=(
                            "no analysis was retained for this cell in the "
                            "current iteration; no mechanism was diagnosed",
                        ),
                    )
                else:
                    # The retained analysis is the real diagnosis; only the score
                    # is replaced with the cell aggregate, which is the measured
                    # evidence the acceptance decision compares against.
                    issue_analysis = CausalAnalysis(
                        mechanism=retained.mechanism,
                        severity=retained.severity,
                        score=cell.max,
                        blame_graph=retained.blame_graph,
                        counterfactual_evidence=retained.counterfactual_evidence,
                        analyzer_model_id=retained.analyzer_model_id,
                        judge_model_id=retained.judge_model_id,
                    )
                self._issue_mechanisms.append(issue_analysis.mechanism)
                self._issue_blame_actors.append(issue_analysis.actor_ids)
                issues.append((task, f"{t_id}:{m_id}", issue_analysis))

        # 3. Build the write set from the base inventory.
        write_set = tuple(
            d.artifact_id
            for d in self.adapter.artifact_inventory(base_entry.version)
            if d.writable
        )

        # 4. Run attempts (sequential or parallel).
        # Trim to what the run budget can still afford, before either branch, so
        # --max-attempts / --max-accepted-edits bound the whole run rather than
        # each iteration. Trimming (not raising) keeps the iteration's already
        # -collected rollout evidence: hitting a cap is a planned stop, not a
        # failure, and the surviving issues are still reported.
        if self.budgets is not None:
            issues = self._affordable_issues(issues)
        if self.profile.use_parallel_batch:
            batch_results = self._run_parallel_attempts(
                base_entry, issues, write_set
            )
            for attempt, decision in batch_results:
                attempts.append(attempt)
                if decision.accepted:
                    accepted.append(attempt.attempt_id)
                    self._budget.accepted_edits += 1
                elif decision.status == AttemptStatus.REGRESSION:
                    regressions.append(attempt.attempt_id)
                else:
                    rejected.append(attempt.attempt_id)
        else:
            for task, issue_id, analysis in issues:
                # Retry budget check.
                if self.profile.use_edit_memory:
                    from agent_evolve.core.memory import artifact_group_of
                    group = artifact_group_of(write_set)
                    lineage = base_entry.version
                    if self.edit_memory.retry_budget.is_exhausted(issue_id, group, lineage):
                        # Mark as exhausted; skip.
                        attempts.append(EditAttempt(
                            attempt_id=self._next_attempt_id(),
                            candidate_id=base_entry.version,
                            issue_id=issue_id,
                            artifact_ids=write_set,
                            operation="skipped",
                            sanitized_reasoning="retry budget exhausted",
                            sanitized_diff={},
                            status=AttemptStatus.EXHAUSTED,
                        ))
                        continue
                attempt, decision = self._run_attempt(
                    base_entry, task, analysis, issue_id, write_set
                )
                attempts.append(attempt)
                if decision.accepted:
                    accepted.append(attempt.attempt_id)
                    self._budget.accepted_edits += 1
                elif decision.status == AttemptStatus.REGRESSION:
                    regressions.append(attempt.attempt_id)
                else:
                    rejected.append(attempt.attempt_id)

        return IterationResult(
            iteration=self._iteration,
            attempts=tuple(attempts),
            accepted=tuple(accepted),
            rejected=tuple(rejected),
            regressions=tuple(regressions),
            pool_size=len(self.pool),
            pareto_frontier=self.pool.pareto_frontier(),
        )

    def _affordable_issues(self, issues: list) -> list:
        """Trim ``issues`` to what the run's attempt budget can still fund.

        Reserves one attempt per surviving issue up front, so the cap is checked
        before any editor call rather than discovered mid-attempt. Trimming
        rather than raising is deliberate: reaching a cap is a planned stop, and
        the rollout evidence already gathered this iteration stays reportable.

        ``max_accepted_edits`` also trims here, because an attempt that cannot
        possibly be accepted is pure spend.
        """
        limits = self.budgets
        if limits is None:
            return issues
        room = len(issues)
        if limits.max_attempts is not None:
            room = min(room, max(0, limits.max_attempts - self._budget.attempts))
        if limits.max_accepted_edits is not None:
            room = min(
                room,
                max(0, limits.max_accepted_edits - self._budget.accepted_edits),
            )
        allowed = issues[:room]
        if allowed:
            self._budget.reserve(limits, attempts=len(allowed))
        return allowed

    def _run_parallel_attempts(
        self,
        base_entry: PoolEntry,
        issues: list[tuple[EvolutionTask, str, CausalAnalysis]],
        write_set: tuple[str, ...],
    ) -> list[tuple[EditAttempt, AcceptanceDecision]]:
        """Run attempts in parallel via snapshot/lease + batch coordinator.

        The architecture requires the orchestrator to "select compatible
        issues" — i.e., no two parallel attempts should target the same
        artifact. We enforce this by tracking which artifacts have been
        claimed by a submitted attempt and skipping later issues whose
        editor-proposed edit would clash.

        Each issue gets its own worker (sequentially in this implementation;
        the snapshot/lease + batch coordinator infrastructure is in place but
        the actual thread pool is the orchestrator caller's responsibility).
        """
        snapshot = snapshot_pool(self.pool, self._iteration)
        lease_mgr = SnapshotLeaseManager(snapshot=snapshot, adapter=self.adapter)
        bc = BatchCoordinator(snapshot=snapshot)

        # Pre-select compatible issues: track claimed artifacts and skip
        # any issue whose editor response would touch a claimed artifact.
        claimed_artifacts: set[str] = set()
        results: list[tuple[EditAttempt, AcceptanceDecision]] = []
        skipped: list[EditAttempt] = []
        for task, issue_id, analysis in issues:
            attempt_id = self._next_attempt_id()
            workspace = lease_mgr.materialize_workspace(base_entry.version, attempt_id)
            current = self.adapter.read_artifacts(base_entry.version, write_set)
            request = EditorRequest(
                base_workspace=workspace,
                task=task,
                analysis=analysis,
                issue_id=issue_id,
                write_set=write_set,
                current_artifacts=dict(current),
                history_refs=tuple(
                    a.attempt_id for a in self.edit_memory.for_issue(issue_id)
                ),
            )
            response = self.editor.propose_edit(request)

            # Check compatibility: skip if any touched artifact is already claimed.
            touched = {e.artifact_id for e in response.edits}
            if touched & claimed_artifacts:
                from agent_evolve.core.memory import AttemptStatus
                skipped.append(EditAttempt(
                    attempt_id=attempt_id,
                    candidate_id=workspace.version,
                    issue_id=issue_id,
                    artifact_ids=tuple(touched),
                    operation="skipped",
                    sanitized_reasoning="parallel clash; deferred to next iteration",
                    sanitized_diff={},
                    status=AttemptStatus.EXHAUSTED,
                ))
                continue
            claimed_artifacts |= touched

            # Acquire leases for all touched artifacts.
            for e in response.edits:
                lease_mgr.acquire_lease(e.artifact_id, attempt_id)

            self.adapter.apply_structured_edits(workspace, response.edits)
            report = self._validate(workspace, task, response)
            decision = decide_acceptance(
                report,
                protected_floors=self.protected_floors,
                net_gain_threshold=self.profile.net_gain_threshold,
            )
            trace = self.adapter.capture_trace(
                self.adapter.run_full_rollout(workspace, task, f"{attempt_id}-origin")
            )
            attempt = build_attempt(
                attempt_id=attempt_id,
                candidate_id=workspace.version,
                issue_id=issue_id,
                response=response,
                evidence_refs=(),
                history_refs=request.history_refs,
                report=report,
                decision=decision,
            )
            # Build a WorkerResult for the coordinator.
            worker_result = WorkerResult(
                attempt_id=attempt_id,
                workspace=workspace,
                edits=response.edits,
                trace=trace,
                attempt=attempt,
            )
            bc.submit(worker_result)
            results.append((attempt, decision))
            # Release leases.
            for e in response.edits:
                lease_mgr.release_lease(e.artifact_id, attempt_id)

        # Commit barrier: add accepted candidates to the pool.
        def on_committed(r: WorkerResult) -> None:
            for attempt, decision in results:
                if attempt.attempt_id == r.attempt_id and decision.accepted:
                    new_candidate = EvolutionCandidate(
                        candidate_id=r.workspace.version,
                        version=r.workspace.version,
                        artifact_hashes={
                            d.artifact_id: d.version_hash
                            for d in self.adapter.artifact_inventory(r.workspace.version)
                        },
                        parent_ids=(base_entry.candidate_id,),
                        ancestor_ids=base_entry.candidate.ancestor_ids + (base_entry.candidate_id,),
                        attempt_ids=(r.attempt_id,),
                    )
                    self.pool.add_candidate(
                        new_candidate, origin_attempt_ids=(r.attempt_id,)
                    )
                    break

        bc.commit_barrier(on_attempt_committed=on_committed)

        # Append skipped attempts to results so the caller sees them.
        for att in skipped:
            from agent_evolve.core.editor import AcceptanceDecision as _AD
            from agent_evolve.core.memory import AttemptStatus as _AS
            results.append((att, _AD(
                accepted=False,
                status=_AS.EXHAUSTED,
                reason="parallel clash; deferred",
                weighted_net_gain=0.0,
            )))
        return results


# ====================================================================== #
# Phase 6: sequential GEPA runner
#
# This is the target-correct sequential loop. It deliberately does NOT reuse
# :class:`Orchestrator` above, which is bound to the legacy ``entropy.Issue``
# and legacy selector. The runner uses the target modules:
#
#   * :mod:`agent_evolve.core.issues`  -- trace-backed Issue + DPP selector
#   * :mod:`agent_evolve.core.blame`   -- CausalFinding (validated contract)
#   * :mod:`agent_evolve.core.pool`    -- ChampionReport + parent_frequencies
#
# Lifecycle per docs/architecture/orchestration-lifecycle.md:40-66:
#   observe -> build_issues -> select_issues -> select_parent -> propose_edits
#   -> validate -> commit_to_pool
#
# Merge, parallel batching, RHO proposal generation, tracing, checkpoints, and
# replay are explicitly out of scope for Phase 6.
# ====================================================================== #
_DEFAULT_ISSUE_CONFIDENCE = 1.0


@dataclass(frozen=True, slots=True)
class GepaAttemptOutcome:
    """Terminal result of one sequential GEPA attempt.

    ``status is AttemptStatus.PENDING`` with an empty ``issue_id`` means no
    evidence-backed work item existed, so no attempt was executed. That is a
    legitimate terminal state, not a failure.
    """

    attempt_id: str
    issue_id: str
    parent_candidate_id: str
    result_candidate_id: str | None
    status: AttemptStatus
    accepted: bool
    weighted_net_gain: float
    reason: str
    artifact_ids: tuple[str, ...] = ()
    fallback_reason: str | None = None
    #: SV-13: the parent retired because this attempt's offspring superseded it,
    #: or ``None``. A pool that shrank must say so in its own attempt record;
    #: inferring it later from two pool snapshots is not auditable.
    retired_parent_id: str | None = None
    #: Why retirement did or did not happen, verbatim from the decision. Retained
    #: even when nothing was retired, because "the judge was unavailable" and "the
    #: child was not preferred" are different facts about the run.
    retirement_reason: str = ""


@dataclass(frozen=True, slots=True)
class _EntropyUnavailableCategories:
    """Stable keys for *why* entropy was unavailable.

    Recorded at the point of failure rather than parsed back out of the prose
    reason string. A tally keyed by free text fragments the moment a message is
    reworded, turning one recognisable cause into several unrecognisable ones --
    the same fragmentation that mechanism clustering exists to prevent.

    Each category implies a different operator action, which is why they are not
    collapsed into a single rate.
    """

    #: The rollout was scored but never diagnosed, so no mechanism exists.
    NO_ANALYSIS: str = "no_analysis"
    #: The runner has no cluster registry, so mechanisms cannot be identified.
    NO_REGISTRY: str = "no_registry"
    #: The clusterer refused to assign, e.g. the cluster cap with no near match.
    UNASSIGNED: str = "unassigned"
    #: Every positivity strength on a passing rollout was refused by the
    #: clusterer, so the candidate's success cannot be keyed to a mechanism.
    STRENGTHS_REFUSED: str = "strengths_refused"
    #: A cell exists but has too few comparable candidates or rollouts. Needs
    #: more evidence, not a code fix.
    FLOOR_UNMET: str = "floor_unmet"


ENTROPY_UNAVAILABLE_CATEGORIES = _EntropyUnavailableCategories()


@dataclass(frozen=True, slots=True)
class EntropyAvailabilityReport:
    """How often cross-candidate entropy could actually be measured.

    Without this, a run in which **no** mechanism cell ever cleared the evidence
    floors is indistinguishable in the summary from one where entropy genuinely
    drove diversity: both report ``H = 0.0`` per task, because a measured zero
    variance and an unmeasurable cell both damp to zero. The difference matters
    because the first ran on issue *quality* alone, so any conclusion about
    entropy-guided selection drawn from it would be unsupported.

    ``reasons`` tallies *why* cells were unavailable. "floors unmet" calls for
    more rollouts per candidate; "adjudication unavailable" calls for fixing the
    dedup endpoint. Collapsing them into a single rate would hide the actionable
    part.
    """

    cells_available: int = 0
    cells_unavailable: int = 0
    reasons: Mapping[str, int] = field(default_factory=dict)

    @property
    def cells_total(self) -> int:
        return self.cells_available + self.cells_unavailable

    @property
    def fallback_rate(self) -> float | None:
        """Share of cells with no usable entropy, or ``None`` when none existed.

        ``0/0`` is undefined and must not render as ``0.0``: that would claim
        perfect availability for a run that measured nothing at all.
        """
        total = self.cells_total
        if total == 0:
            return None
        return self.cells_unavailable / total

    @property
    def entropy_never_available(self) -> bool:
        """True only when cells existed and every one of them was unavailable."""
        return self.cells_total > 0 and self.cells_available == 0

    def line(self) -> str:
        rate = self.fallback_rate
        if rate is None:
            return "entropy: no cells observed"
        detail = ", ".join(
            f"{reason}={count}" for reason, count in sorted(self.reasons.items())
        )
        suffix = f" ({detail})" if detail else ""
        return (
            f"entropy: {self.cells_unavailable}/{self.cells_total} cells "
            f"unavailable = {rate * 100:.0f}% fallback{suffix}"
        )


@dataclass(frozen=True, slots=True)
class GepaRunResult:
    """Summary of ``n_attempts`` sequential GEPA attempts."""

    attempts: tuple[GepaAttemptOutcome, ...]
    champion: ChampionReport | None
    pool_size: int
    pareto_frontier: tuple[str, ...]
    #: SV-12: how often entropy was measurable. ``None`` means the run did not
    #: aggregate it -- deliberately not a zeroed report, which would read as
    #: "entropy was fully available".
    entropy_availability: EntropyAvailabilityReport | None = None

    @property
    def attempts_run(self) -> int:
        return len(self.attempts)

    @property
    def accepted_count(self) -> int:
        return sum(1 for a in self.attempts if a.accepted)

    @property
    def rejected_count(self) -> int:
        return sum(
            1
            for a in self.attempts
            if not a.accepted and a.status is not AttemptStatus.PENDING
        )

    @property
    def no_issue_count(self) -> int:
        return sum(1 for a in self.attempts if a.status is AttemptStatus.PENDING)


@dataclass(slots=True)
class SequentialGepaRunner:
    """Deterministic single-threaded GEPA loop over an adapter-backed pool.

    The runner never mutates a parent version: every attempt is applied to an
    adapter-materialized workspace, and only an accepted attempt is published to
    the pool. Parent sampling is proportional to
    :meth:`PersistentPool.parent_frequencies` under a seeded RNG, per
    ``docs/architecture/selection-algorithms.md:306-315``.
    """

    adapter: EvolutionAdapter
    pool: PersistentPool
    analyzer_judge: AnalyzerJudge
    editor: Editor
    embedder: MechanismEmbedder | None = None
    storage: StorageBackend | None = None
    config: ResolvedConfig | None = None
    mechanism_cluster_id: str = "c0"
    #: SV-12 step 3. Per-task mechanism clustering for the **entropy tracker's**
    #: cell keys. Deliberately separate from ``mechanism_cluster_id`` above,
    #: which stays constant because the two structures answer different
    #: questions and need opposite key policies:
    #:
    #: * the pool's score tensor asks *"is c1 better than base?"* and needs
    #:   **shared** keys -- champion selection intersects on the exact full key,
    #:   so mechanism-keyed pool cells yield an empty intersection and SV-2
    #:   regresses **silently** (``dominates()`` correctly returns ``False`` on
    #:   no overlap, and a frontier containing everything looks like healthy
    #:   diversity while meaning nothing could be compared to anything);
    #: * the entropy tracker asks *"how much do candidates disagree on this
    #:   mechanism?"* and needs **separated** keys, or unrelated faults pool
    #:   into one cell and their score spread reads as within-mechanism
    #:   variance -- "a fix is reachable here" for a mechanism that does not
    #:   exist.
    cluster_registry: ClusterRegistry | None = None
    #: SV-12 step 3. Mechanism-keyed cross-candidate entropy with the spec's
    #: evidence floors (>=3 comparable candidates, >=2 rollouts each). This is
    #: the single implementation of ``H(t, m) = Var * max(max_score,
    #: score_floor)``; the genetic path previously recomputed variance inline
    #: over the constant ``mechanism_cluster_id`` bucket, which measured
    #: cross-candidate spread inside one synthetic cell rather than
    #: per-mechanism variance, and applied no floors at all.
    entropy: EntropyTracker = field(default_factory=EntropyTracker)
    #: Rollouts recorded per ``(task, mechanism, candidate)``, used only to decide
    #: when a candidate clears the tracker's per-candidate rollout floor and may
    #: be promoted to comparable.
    _entropy_rollout_counts: dict[tuple[str, str, str], int] = field(
        default_factory=dict, repr=False, compare=False
    )
    #: Why a task's entropy evidence could not be filed, keyed by task. This is
    #: what makes the coarse fallback *reportable*: an unavailable entropy term
    #: with no reason is indistinguishable from a measured zero.
    _last_entropy_unavailable_reasons: dict[str, str] = field(
        default_factory=dict, repr=False, compare=False
    )
    #: SV-12: the same facts keyed by a stable category for run-level
    #: aggregation. Kept alongside the prose reasons rather than replacing them:
    #: a category cannot explain one task to a reader, and prose cannot be
    #: tallied across a run without fragmenting on wording.
    _entropy_unavailable_categories: dict[str, str] = field(
        default_factory=dict, repr=False, compare=False
    )
    seed: int = 0
    protected_floors: tuple[ProtectedFloor, ...] = ()
    net_gain_threshold: float = 0.0
    #: Attempt history + retry budget. Every attempt this runner completes is
    #: recorded here, which is what makes the editor's history tools return
    #: anything: ``search_edit_history`` reads :meth:`EditMemory.retrieve` and
    #: ``get_attempt_outcome`` reads :meth:`EditMemory.get`. Recording is also
    #: the only path that charges :class:`RetryBudget`, so retry exhaustion
    #: depends on it. This runner previously had no memory at all, so both tools
    #: reported "no prior attempts" on every call and the retry budget never
    #: counted an attempt (SV-6).
    edit_memory: EditMemory = field(default_factory=EditMemory)
    # Donor parents offered to the editor alongside the primary (spec §7).
    donor_count: int = 2
    #: SV-13 generational retirement. Injected symmetric preference judge:
    #: ``compare(task, baseline_trace, candidate_trace) -> verdict`` with
    #: ``.score`` and ``.available``. Same seam ``core/rho/rounds.py`` uses, so
    #: ``core/`` never imports an adapter.
    #:
    #: ``None`` disables retirement entirely, which is the default: every offline
    #: path and every existing caller keeps its current behaviour, and a run only
    #: opts into pool shaping by supplying a judge.
    compare_preference: Callable[..., object] | None = None
    #: Per-task traces captured during the last ``validate`` (the child) and the
    #: last ``build_issues`` (the parent). Both rollout sets already happen; these
    #: exist so retirement reuses them instead of paying for them twice.
    _last_validation_traces: dict[str, ExecutionTrace] = field(
        default_factory=dict, repr=False
    )
    _last_observation_traces: dict[str, ExecutionTrace] = field(
        default_factory=dict, repr=False
    )
    #: SV-14: the child's own diagnoses from the last ``validate``, keyed by
    #: task id (present only where the diagnose gate produced one). These
    #: analyses were already paid for by ``rollout_group``'s phase 3; without
    #: this map they were discarded, and ``commit_to_pool`` stamped the
    #: *parent's* analysis onto every offspring cell instead.
    _last_validation_analyses: dict[str, CausalAnalysis] = field(
        default_factory=dict, repr=False
    )
    #: SV-14: every scorable child rollout from the last ``validate``, retained
    #: so an accepted commit can file the offspring's mechanism evidence via
    #: ``_record_entropy_evidence`` -- the only route that function had ran
    #: through ``build_issues``, i.e. through whoever was observed as parent.
    _last_validation_rollouts: tuple[ObservedRollout, ...] = field(
        default=(), repr=False
    )
    #: TS2 (D5 prerequisite): the cross-attempt trace store. Every *scorable*
    #: rollout of any score is appended under ``(candidate_id, task_id)`` --
    #: no quality gate at capture, because complementarity is relative per
    #: mechanism: a 0.4 may be the best any candidate has done on a task, and
    #: the editor tool degrades to least-bad failures, which requires failure
    #: traces here. Unscorable rollouts stay excluded (SV-9). This survives
    #: the per-attempt resets of ``_last_validation_traces`` and
    #: ``_last_observation_traces``; it dies with the runner, deliberately:
    #: raw traces never reach storage (``_persist_attempt`` invariant), and
    #: the storage sanitizer's 2000-char truncation would return them
    #: silently amputated.
    _trace_store: dict[tuple[str, str], list[ObservedRollout]] = field(
        default_factory=dict, repr=False
    )
    #: D5/J2B: the positivity judge (Judge 2). ``None`` -- the default --
    #: keeps the runner byte-identical to the pre-D5 behaviour and costs
    #: nothing; when configured, the phase-3 gate ALSO analyzes *passing
    #: scorable* rollouts (+1 model call each, opt-in spend) and their
    #: strength findings ride ``ObservedRollout.strengths`` into the TS2
    #: store for the future signed index.
    positivity_judge: PositivityJudge | None = None
    #: How many success analyses the configured judge has been asked for.
    _positivity_calls: int = field(default=0, repr=False)
    #: Refused batches: (trace_id, reason). A positivity judge returning any
    #: non-(-1) finding has its WHOLE batch refused here -- refuse, never
    #: flip, so a mis-wired adapter surfaces instead of poisoning the index.
    _positivity_failures: list[tuple[str, str]] = field(
        default_factory=list, repr=False
    )
    #: SV-10: the parent ``build_issues`` observed, and every fault it diagnosed
    #: for that parent. ``run_attempt`` reuses both instead of drawing a second,
    #: independent parent and discarding all but the one worked issue.
    #:
    #: ``select_parent`` consumes ``rng.random()``, so two draws in one attempt
    #: are *independent*: measured on a four-candidate pool they disagreed on the
    #: first run (``['cand-2', 'cand-2', 'cand-1']``), which diagnoses one
    #: parent's faults and then materializes a different parent's workspace.
    _last_observed_parent_id: str = field(default="", repr=False)
    _last_parent_issues: tuple[TargetIssue, ...] = field(
        default=(), repr=False
    )
    #: Measures every rollout and names the grader that did it. Defaults to the
    #: task-contract scorer, which is what the offline suite and the fake stack
    #: use. A real run supplies a benchmark-driven scorer so the headline number
    #: comes from the benchmark's own grader.
    scorer: Scorer | None = None
    #: Executes a whole task batch for one version. ``None`` means "roll tasks
    #: out one at a time through the adapter", which is the existing behaviour
    #: and the only safe mode for a workspace-bound real harness on one thread.
    rollout_batch: RolloutBatch | None = None
    #: Zero-arg builder for a report-based analyzer, used only when
    #: ``analyzer_workers > 1``. Required for concurrency because the analyzer is
    #: expected to become a stateful agent: sharing one across threads would
    #: interleave two trajectories into one conversation.
    analyzer_factory: Callable[[], ReportAnalyzerJudge] | None = None
    _selector: TargetIssueSelector | None = field(default=None, init=False, repr=False)
    _rng: random.Random | None = field(default=None, init=False, repr=False)
    _iteration: int = field(default=0, init=False, repr=False)
    _attempt_seq: int = field(default=0, init=False, repr=False)
    _probe_seq: int = field(default=0, init=False, repr=False)
    _resolved_analyzer: AnalyzerJudge | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _resolved_scorer: Scorer | None = field(
        default=None, init=False, repr=False, compare=False
    )
    #: Probes whose rollout produced no measurement. Counted, never scored: a
    #: probe with no evidence must not become a passing validation result.
    _unscorable_probes: int = field(default=0, init=False, repr=False)
    #: Run-scoped spend ledger. Accumulates across outer iterations on purpose:
    #: ``--max-rollouts`` means "for this run", not "per iteration", so a
    #: per-iteration counter would let N iterations spend N times the cap.
    _budget: BudgetUsage = field(
        default_factory=BudgetUsage, init=False, repr=False, compare=False
    )
    #: Analyzer failures observed this run, as data. A model outage is recorded,
    #: not converted into a diagnosis and not allowed to abort the batch.
    _analysis_failures: list[AnalysisOutcome] = field(
        default_factory=list, init=False, repr=False
    )
    #: Every mechanism that reached issue synthesis, in order, for auditing.
    _observed_mechanisms: list[str] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.mechanism_cluster_id:
            raise ValueError("mechanism_cluster_id is required")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.embedder is None:
            self.embedder = LexicalEmbedder()
        if self.cluster_registry is None:
            # Reuse the runner's embedder so mechanism identity is decided by the
            # embedder the run was configured with, not a second silent default.
            # Bound to a local so the factory does not re-read a later mutation.
            configured = self.embedder
            self.cluster_registry = ClusterRegistry(
                embedder_factory=lambda: configured,
            )
        self._rng = random.Random(self.seed)
        config = self.config
        # Honour the configured entropy floors and thresholds on the tracker, so
        # the one instrument computing H(t, m) is governed by the run's config
        # rather than its own dataclass defaults.
        if config is not None:
            for attr, cfg_attr in (
                ("score_floor", "entropy_score_floor"),
                ("recombination_score_threshold", "entropy_recombination_score_threshold"),
                ("frontier_weight", "entropy_frontier_weight"),
                ("min_comparable_candidates", "entropy_min_comparable_candidates"),
                ("min_rollouts_per_candidate", "entropy_min_rollouts_per_candidate"),
            ):
                value = getattr(config, cfg_attr, None)
                if value is not None:
                    setattr(self.entropy, attr, value)
        if self.analyzer_workers > 1 and self.analyzer_factory is None:
            raise ValueError(
                f"max_analyzer_workers={self.analyzer_workers} requires an "
                "analyzer_factory: a stateful analyzer shared across threads "
                "would interleave two trajectories into one conversation, so "
                "each worker must build its own instance"
            )
        self._selector = TargetIssueSelector(
            theta=config.dpp_theta if config is not None else TARGET_THETA,
            score_floor=(
                config.dpp_score_floor if config is not None else TARGET_SCORE_FLOOR
            ),
            max_items=config.dpp_max_items if config is not None else 100,
            min_gain=config.dpp_min_gain if config is not None else 1e-12,
            seed=self.seed,
            frontier_weight=(
                config.entropy_frontier_weight if config is not None else 0.30
            ),
        )

    # ------------------------------------------------------------------ #
    # Scoring, concurrency and failure surfaces
    # ------------------------------------------------------------------ #
    @property
    def resolved_scorer(self) -> Scorer:
        """``scorer``, defaulting to the task-contract scorer.

        Cached so a stateful benchmark is built once, and so ``grader_name`` is
        stable across an entire run: a run whose grader could change mid-flight
        would report a pass rate with no single denominator definition.
        """
        resolved = self._resolved_scorer
        if resolved is None:
            resolved = self.scorer if self.scorer is not None else ContractScorer()
            self._resolved_scorer = resolved
        return resolved

    @property
    def grader_name(self) -> str:
        return self.resolved_scorer.grader_name

    @property
    def analyzer_workers(self) -> int:
        """Bounded analyzer concurrency, from config. Defaults to 1.

        Analyzer fan-out is pure LLM calls with no CUGA process involved, so
        threads are safe here -- unlike rollouts, where ``CUGA_FOLDER`` is
        process-global.
        """
        if self.config is None:
            return 1
        return max(1, int(getattr(self.config, "max_analyzer_workers", 1)))

    @property
    def unscorable_probe_count(self) -> int:
        return self._unscorable_probes

    @property
    def analysis_failures(self) -> tuple[AnalysisOutcome, ...]:
        return tuple(self._analysis_failures)

    @property
    def observed_mechanisms(self) -> tuple[str, ...]:
        return tuple(self._observed_mechanisms)


    # ------------------------------------------------------------------ #
    # Analyzer protocol resolution
    # ------------------------------------------------------------------ #
    @property
    def resolved_analyzer(self) -> AnalyzerJudge:
        """``analyzer_judge`` adapted to this runner's ``(task, trace)`` call sites.

        Identical semantics to :attr:`Orchestrator.resolved_analyzer`: a legacy
        analyzer passes through by identity, a report-based analyzer is wrapped
        once and cached.
        """
        resolved = self._resolved_analyzer
        if resolved is None:
            resolved = as_legacy_analyzer(self.analyzer_judge)
            self._resolved_analyzer = resolved
        return resolved

    # ------------------------------------------------------------------ #
    # Identifiers
    # ------------------------------------------------------------------ #
    @property
    def selector(self) -> TargetIssueSelector:
        assert self._selector is not None
        return self._selector

    def _next_attempt_id(self) -> str:
        attempt_id = make_attempt_id(self._iteration, self._attempt_seq)
        self._attempt_seq += 1
        return attempt_id

    def _next_probe_id(self, prefix: str) -> str:
        self._probe_seq += 1
        return f"{prefix}-p{self._probe_seq:05d}"

    def _writable_artifact_ids(self, version: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                d.artifact_id
                for d in self.adapter.artifact_inventory(version)
                if d.writable
            )
        )

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _execute_rollouts(
        self, version: str, tasks: Sequence[EvolutionTask], *, prefix: str
    ) -> tuple[RolloutOutcome, ...]:
        """Run ``tasks`` against ``version``, returning failures as data.

        With a ``rollout_batch`` the whole batch is delegated -- that is how a
        real run gets process-isolated parallel CUGA execution. Without one,
        tasks run through the adapter one at a time on this thread, which is the
        pre-existing behaviour and the only safe in-process mode for a
        workspace-bound harness (``CUGA_FOLDER`` is process-global).

        An adapter that raises is recorded as one failed rollout, not propagated:
        one broken task must not discard the evidence from its siblings.

        The whole batch is reserved against the run budget BEFORE any rollout is
        issued, so ``--max-rollouts`` refuses a batch it cannot afford rather
        than stopping halfway through one and leaving a partially-measured
        candidate that looks like a real result.
        """
        if tasks and self.config is not None:
            self._budget.reserve(self.config.budgets, rollouts=len(tasks))
        if self.rollout_batch is not None:
            return tuple(self.rollout_batch.run_rollouts(version, tasks, prefix=prefix))

        outcomes: list[RolloutOutcome] = []
        for task in tasks:
            probe_id = self._next_probe_id(f"{prefix}-{task.task_id}")
            try:
                workspace = self.adapter.materialize_candidate(version, probe_id)
                result = self.adapter.run_full_rollout(workspace, task, probe_id)
                trace = self.adapter.capture_trace(result)
            except Exception as exc:  # noqa: BLE001 - a failure is data
                outcomes.append(
                    RolloutOutcome(
                        task=task,
                        trace=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            outcomes.append(RolloutOutcome(task=task, trace=trace))
        return tuple(outcomes)

    def rollout_group(
        self, version: str, tasks: Sequence[EvolutionTask], *, prefix: str
    ) -> tuple[ObservedRollout, ...]:
        """Roll out, score, and diagnose one candidate version over ``tasks``.

        Three phases, in this order and for these reasons:

        1. **Execute.** Rollouts happen first so the analyzer never blocks on
           agent execution, and so a batch executor can parallelise them under
           process isolation.
        2. **Score.** The score comes from ``resolved_scorer`` -- a measurement.
           A rollout that produced no answer is marked unscorable here and is
           excluded from every denominator from this point on.
        3. **Diagnose.** Only *answered, failing* rollouts are analyzed, with
           bounded concurrency. Analyzing an unanswered rollout would spend a
           model call to diagnose an outage, and analyzing a passing one would
           ask for a mechanism where there is no failure.

        Diagnosis never supplies the score: ``analysis_from_finding`` requires
        the caller's own measurement, so a diagnosis can never be mistaken for
        a verdict on correctness.
        """
        executed = self._execute_rollouts(version, tasks, prefix=prefix)
        scorer = self.resolved_scorer

        scored: list[tuple[RolloutOutcome, RolloutScore]] = []
        for outcome in executed:
            if outcome.trace is None:
                scored.append(
                    (
                        outcome,
                        RolloutScore(
                            task_id=outcome.task.task_id,
                            grader_name=scorer.grader_name,
                            score=0.0,
                            scorable=False,
                            reason=outcome.error or "no rollout was produced",
                        ),
                    )
                )
                continue
            scored.append((outcome, scorer.score_rollout(outcome.task, outcome.trace)))

        # Only answered failures are worth a model call.
        to_analyze = [
            (outcome, score)
            for outcome, score in scored
            if outcome.trace is not None and score.scorable and not score.passed
        ]
        analyses = self._analyze(to_analyze)

        # D5/J2B: with a positivity judge configured, the gate ALSO opens for
        # answered SUCCESSES -- strengths are evidence for the editor index.
        # Off by default (positivity_judge=None): this loop never runs.
        strengths_by_rollout: dict[tuple[str, int], tuple[CausalFinding, ...]] = {}
        if self.positivity_judge is not None:
            to_posit = [
                (outcome, score)
                for outcome, score in scored
                if outcome.trace is not None and score.scorable and score.passed
            ]
            for pos_index, (outcome, _score) in enumerate(to_posit):
                assert outcome.trace is not None
                self._positivity_calls += 1
                # ?03: label these calls as Judge-2 spend on this rollout.
                with correlation_scope(
                    candidate=outcome.trace.candidate_id,
                    task=outcome.task.task_id,
                    rollout=pos_index,
                    phase="positivity",
                ):
                    findings = self.positivity_judge.analyze_success(
                        outcome.task, outcome.trace
                    )
                bad = [
                    f
                    for f in findings
                    if getattr(f, "valence", -1) != -1
                ]
                if bad:
                    reason = (
                        "polarity violation: the positivity judge returned "
                        f"{len(bad)} non-strength finding(s) for trace "
                        f"{outcome.trace.trace_id!r}; Judge 2 may only emit "
                        "strengths -- batch refused, not flipped"
                    )
                    self._positivity_failures.append(
                        (outcome.trace.trace_id, reason)
                    )
                    continue
                strengths_by_rollout[
                    (outcome.task.task_id, id(outcome))
                ] = tuple(findings)

        observed: list[ObservedRollout] = []
        for outcome, score in scored:
            analysis, error = analyses.get(
                (outcome.task.task_id, id(outcome)), (None, "")
            )
            strengths = strengths_by_rollout.get(
                (outcome.task.task_id, id(outcome)), ()
            )
            observed.append(
                ObservedRollout(
                    task=outcome.task,
                    trace=outcome.trace,
                    score=score,
                    analysis=analysis,
                    error=outcome.error or error,
                    strengths=strengths,
                )
            )
        return tuple(observed)

    def _analyze(
        self, items: Sequence[tuple[RolloutOutcome, RolloutScore]]
    ) -> dict[tuple[str, int], tuple[CausalAnalysis | None, str]]:
        """Diagnose the given rollouts, with concurrency when configured.

        At ``analyzer_workers == 1`` the legacy per-rollout path is used
        verbatim, so existing behaviour (including error propagation from a
        legacy analyzer) is unchanged. Above 1,
        :class:`ParallelAnalysisRunner` fans the sanitized reports out over
        threads and returns per-item failures as data; the score is still the
        caller's, supplied to ``analysis_from_finding``.
        """
        if not items:
            return {}

        out: dict[tuple[str, int], tuple[CausalAnalysis | None, str]] = {}
        if self.analyzer_workers == 1:
            analyzer = self.resolved_analyzer
            for analyze_index, (outcome, score) in enumerate(items):
                assert outcome.trace is not None  # filtered by the caller
                # ?03: label this diagnosis with its rollout identity so the
                # proxy can tie the model call to (candidate, task, rollout).
                with correlation_scope(
                    candidate=outcome.trace.candidate_id,
                    task=outcome.task.task_id,
                    rollout=analyze_index,
                    phase="diagnose",
                ):
                    analysis = analyzer.analyze(outcome.task, outcome.trace)
                self._observed_mechanisms.append(analysis.mechanism)
                out[(outcome.task.task_id, id(outcome))] = (analysis, "")
            return out

        factory = self.analyzer_factory
        assert factory is not None  # validated in __post_init__
        reports = tuple(
            rollout_group_report(outcome.task, outcome.trace)  # type: ignore[arg-type]
            for outcome, _ in items
        )
        # ?03: labels travel INTO the workers -- pool threads do not inherit
        # the submitting thread's context, so each report carries its own.
        labels = tuple(
            CorrelationContext(
                candidate=outcome.trace.candidate_id if outcome.trace else None,
                task=outcome.task.task_id,
                rollout=analyze_index,
                phase="diagnose",
            )
            for analyze_index, (outcome, _) in enumerate(items)
        )
        runner = ParallelAnalysisRunner(
            analyzer_factory=factory, max_workers=self.analyzer_workers
        )
        outcomes = runner.run(reports, labels=labels)
        for (rollout, score), analysis_outcome in zip(items, outcomes, strict=True):
            key = (rollout.task.task_id, id(rollout))
            if not analysis_outcome.ok:
                self._analysis_failures.append(analysis_outcome)
                out[key] = (None, analysis_outcome.error)
                continue
            findings = analysis_outcome.findings
            if len(findings) != 1:
                # A single-rollout report yields exactly one finding. Anything
                # else cannot be projected onto one verdict without discarding
                # evidence, so it is recorded as a gap instead of guessed at.
                error = (
                    f"a single-rollout report yielded {len(findings)} findings; "
                    "exactly one verdict is representable"
                )
                self._analysis_failures.append(
                    AnalysisOutcome(
                        report=analysis_outcome.report,
                        findings=(),
                        error=error,
                        ok=False,
                    )
                )
                out[key] = (None, error)
                continue
            finding = findings[0]
            # D5.3 wall: this receive site belongs to Judge 1, whose only
            # legal polarity is the fault (+1). A strength here means a
            # mis-wired adapter; it is refused and recorded, never flipped
            # -- a silent flip would hide the wiring defect.
            if getattr(finding, "valence", 1) != 1:
                error = (
                    "polarity violation: the failure analyzer produced a "
                    f"valence={getattr(finding, 'valence', 1)} finding "
                    f"({finding.verdict_id!r}); Judge 1 may only emit faults"
                )
                self._analysis_failures.append(
                    AnalysisOutcome(
                        report=analysis_outcome.report,
                        findings=(),
                        error=error,
                        ok=False,
                    )
                )
                out[key] = (None, error)
                continue
            analysis = analysis_from_finding(
                finding,
                score=score.score,
                analyzer_model_id=str(
                    getattr(factory, "analyzer_model_id", "") or ""
                ),
            )
            self._observed_mechanisms.append(analysis.mechanism)
            out[key] = (analysis, "")
        return out

    def observe(
        self, entry: PoolEntry, task: EvolutionTask
    ) -> tuple[ExecutionTrace, CausalAnalysis]:
        """Roll a candidate out on one task and analyze the resulting trace.

        The rollout uses a throwaway workspace materialized from the candidate's
        version so the candidate's own artifacts are never written.

        Raises ``ValueError`` when the rollout produced no measurement. There is
        no analysis of a rollout that did not happen, and returning a zero-scored
        analysis would put a fabricated failure into the caller's hands; callers
        that must tolerate a failed rollout use :meth:`rollout_group`.
        """
        observed = self.rollout_group(
            entry.version, (task,), prefix=f"obs-{entry.candidate_id}"
        )[0]
        if observed.trace is None or observed.score is None or not observed.scorable:
            reason = (
                observed.error
                or (observed.score.reason if observed.score is not None else "")
                or "unknown"
            )
            raise ValueError(
                f"rollout of {entry.candidate_id!r} on task {task.task_id!r} "
                f"produced no measurement ({reason}); a failed rollout is not a "
                f"wrong answer and must not be scored"
            )
        if observed.analysis is not None:
            return observed.trace, observed.analysis
        if observed.score.passed:
            return observed.trace, empty_analysis()
        return observed.trace, abstained_analysis(
            "insufficient_evidence",
            score=observed.score.score,
            evidence=(observed.error or "no analysis was produced for this rollout",),
        )

    # ------------------------------------------------------------------ #
    # Finding synthesis
    # ------------------------------------------------------------------ #
    def finding_from_analysis(
        self,
        analysis: CausalAnalysis,
        *,
        task: EvolutionTask,
        candidate_id: str,
        trace_id: str,
        verdict_id: str,
        writable_artifact_ids: Sequence[str],
    ) -> CausalFinding:
        """Attribute an analysis to adapter-declared writable artifacts.

        A generic analyzer+judge reports *actors*, not artifacts: the fake
        analyzer's blame nodes carry ``artifacts=()``. An issue with no writable
        attribution is rejected before ranking
        (``selection-algorithms.md:136``), so the runner attributes the
        adapter's declared writable set for the blamed actors and records each
        attributed artifact as a trace-backed evidence reference.

        Synthetic placeholder nodes are forbidden. When the analysis names no
        blamed actor, absence of evidence is expressed as
        ``status="insufficient_evidence"`` with an empty blame graph; the caller
        must not build an issue from it.

        Only sanitized material crosses this boundary: the rationale names the
        mechanism and the artifacts, never the task's expected contract.
        """
        write_set = tuple(sorted(set(writable_artifact_ids)))
        if not write_set:
            raise ValueError("writable_artifact_ids must not be empty")

        blamed = tuple(
            sorted(
                (n for n in analysis.blame_graph.nodes),
                key=lambda n: (-n.blame, n.actor_id),
            )
        )
        if not blamed:
            return CausalFinding(
                verdict_id=verdict_id,
                candidate_id=candidate_id,
                task_id=task.task_id,
                trace_id=trace_id,
                # D5.3: this seam converts Judge 1 analyses, so the polarity
                # is stamped in code -- faults only.
                valence=1,
                status="insufficient_evidence",
                mechanism_description=analysis.mechanism,
                mechanism_cluster_id=self.mechanism_cluster_id,
                blame_graph=BlameGraph(nodes=()),
                evidence_refs=(),
                rationale=(
                    "no blamed actor in the analysis; artifact attribution is "
                    f"un-evidenced for task {task.task_id}"
                ),
            )

        top = blamed[0]
        nodes = (
            BlameNode(actor_id=top.actor_id, blame=top.blame, artifacts=write_set),
        ) + tuple(
            BlameNode(actor_id=n.actor_id, blame=n.blame, artifacts=())
            for n in blamed[1:]
        )

        return CausalFinding(
            verdict_id=verdict_id,
            candidate_id=candidate_id,
            task_id=task.task_id,
            trace_id=trace_id,
            # D5.3: this seam converts Judge 1 analyses, so the polarity is
            # stamped in code -- faults only.
            valence=1,
            status="observed",
            mechanism_description=analysis.mechanism,
            mechanism_cluster_id=self.mechanism_cluster_id,
            severity=analysis.severity,
            confidence=_DEFAULT_ISSUE_CONFIDENCE,
            blame_graph=BlameGraph(nodes=nodes),
            evidence_refs=write_set,
            rationale=(
                "attributed the adapter-declared writable set for the "
                f"highest-blame actor on task {task.task_id}"
            ),
        )

    # ------------------------------------------------------------------ #
    # build_issues
    # ------------------------------------------------------------------ #
    def build_issues(
        self, tasks: Sequence[EvolutionTask]
    ) -> tuple[TargetIssue, ...]:
        """Build trace-backed issues for every task the selected parent fails.

        Returns target :class:`agent_evolve.core.issues.Issue` values. A task the
        parent already satisfies produces no issue, and a finding with no writable
        attribution is dropped by :func:`build_issue` rather than ranked.

        A task whose rollout produced no measurement yields no issue **and no
        score**. It is neither an observed failure to diagnose nor a data point
        to record: a broken harness must not look like a candidate that answered
        wrongly.

        **SV-11: the subject is the selected parent, not always the base.** This
        method used to hardcode ``self.pool.base`` for the rollout, the write
        set, the inventory *and* the score attribution. Because it runs once per
        attempt, base absorbed every re-observation while each candidate kept only
        the rollouts its own attempt produced -- measured over six attempts on a
        two-task pool: base 12 rollouts, every candidate 2, and **every cell stuck
        at one comparable candidate** against an entropy floor of 3
        (``core/entropy.py:110``). That is SV-12, and it was a direct consequence:
        cross-candidate diversity could never acquire the evidence it requires.

        Observing the parent instead is **cost-neutral** -- one rollout per task
        either way, the subject moves and the count does not. Two properties this
        must preserve, because breaking either would be worse than the defect:

        * **The base remains reachable.** ``select_parent`` returns base whenever
          no candidate holds winning-cell evidence, so a fresh pool behaves
          exactly as before. Base loses only its *guaranteed* per-iteration
          refresh, which is the accepted trade of the cost-neutral option.
        * **The write set follows the subject.** Diagnosing candidate X while
          offering base's artifacts would attribute X's mechanism to surfaces it
          does not own.
        """
        parent = self.select_parent()
        inventory = self.adapter.artifact_inventory(parent.version)
        write_set = self._writable_artifact_ids(parent.version)
        observed = self.rollout_group(
            parent.version, tasks, prefix=f"obs-{parent.candidate_id}"
        )
        out: list[TargetIssue] = []
        # SV-13: retain the parent's traces by task for the retirement judge.
        # These rollouts are the observation this method already performs.
        self._last_observation_traces = {}
        # SV-10: name the parent this observation belongs to, so run_attempt
        # edits the entry whose faults were actually diagnosed.
        self._last_observed_parent_id = parent.candidate_id
        for rollout in observed:
            if rollout.trace is None or rollout.score is None or not rollout.scorable:
                continue
            self._last_observation_traces[rollout.task.task_id] = rollout.trace
            self._record_rollout_score(parent.candidate_id, rollout)
            # TS2: the parent's observations are evidence too, pass or fail.
            self._record_stored_trace(parent.candidate_id, rollout)
            if rollout.score.passed:
                continue
            analysis = rollout.analysis
            if analysis is None:
                # The rollout answered and failed, but no diagnosis exists (an
                # analyzer outage). The score is already recorded; there is
                # nothing to attribute an artifact from, so no issue is built.
                continue
            finding = self.finding_from_analysis(
                analysis,
                task=rollout.task,
                candidate_id=parent.candidate_id,
                trace_id=rollout.trace.trace_id,
                verdict_id=f"{rollout.task.task_id}:{self.mechanism_cluster_id}",
                writable_artifact_ids=write_set,
            )
            if finding.status == "insufficient_evidence":
                continue
            issue = build_target_issue(
                finding,
                inventory,
                entropy=self._cell_entropy(rollout.task.task_id),
                coverage_need=self._coverage_need(rollout.task.task_id),
                pareto_relevance=self._pareto_relevance(parent.candidate_id),
                embedding=self._embed_finding(finding, rollout.task),
                lineage=parent.version,
                entropy_tier=self._entropy_tier(rollout.task.task_id),
            )
            if issue is not None:
                out.append(issue)
        # SV-10: retain the parent's FULL diagnosed fault set. run_attempt works
        # one of these, and the editor is shown all of them: every one is already
        # paid for with the rollouts and analyzer calls above, so discarding them
        # spends evidence rather than saving cost.
        self._last_parent_issues = tuple(out)
        return tuple(out)

    def _record_rollout_score(
        self, candidate_id: str, rollout: ObservedRollout
    ) -> None:
        """Record one *scorable* rollout's measurement in the pool.

        Refuses an unscorable rollout outright rather than skipping it quietly:
        this is the last line of defence for the property that a failed rollout
        can never reach a score denominator, and a silent no-op here would make
        a future miswiring invisible.
        """
        score = rollout.score
        if score is None or not score.scorable:
            raise ValueError(
                "refusing to record an unscorable rollout: a rollout that "
                "produced no measurement is not a wrong answer"
            )
        assert rollout.trace is not None  # scorable implies a trace
        entry = self.pool.get(candidate_id)
        cell = entry.cell(rollout.task.task_id, self.mechanism_cluster_id)
        analysis = rollout.analysis
        self.pool.record_score(
            candidate_id,
            score.score,
            ScoreProvenance(
                task_id=rollout.task.task_id,
                mechanism_cluster_id=self.mechanism_cluster_id,
                trace_id=rollout.trace.trace_id,
                rollout_seq=cell.rollout_count,
                analyzer_model_id=(
                    analysis.analyzer_model_id if analysis is not None else None
                ),
                # The grader that produced this number, recorded as the judge:
                # a score whose grader is unnamed cannot be compared later.
                judge_model_id=score.grader_name,
                blame_confidence=(
                    min(1.0, analysis.blame_graph.total_blame())
                    if analysis is not None
                    else None
                ),
                blame_stability=1.0,
                artifact_versions=dict(entry.candidate.artifact_hashes),
            ),
        )
        # SV-12 step 3: file the same measurement into the mechanism-keyed
        # entropy tracker. Keyed at write time because the genetic path already
        # holds the diagnosis when it records a score (unlike RHO, whose phase-5
        # base rollouts precede phase-6 diagnosis), so no retroactive re-filing
        # is needed. The pool write above deliberately keeps the constant key.
        self._record_entropy_evidence(candidate_id, rollout)

    # ------------------------------------------------------------------ #
    # TS2: cross-attempt trace store
    # ------------------------------------------------------------------ #
    def _record_stored_trace(
        self, candidate_id: str, rollout: ObservedRollout
    ) -> None:
        """Append one scorable rollout to the cross-attempt store.

        Callers have already filtered unscorable rollouts; this re-checks
        rather than trusting that, because a crashed rollout in the store
        would hand Judge 2 an outage to diagnose as if it were behaviour.
        """
        if rollout.trace is None or rollout.score is None or not rollout.scorable:
            return
        key = (candidate_id, rollout.task.task_id)
        self._trace_store.setdefault(key, []).append(rollout)

    def traces_for(
        self, candidate_id: str, task_id: str
    ) -> tuple[ObservedRollout, ...]:
        """Every stored scorable rollout for one candidate on one task.

        Read API for the future positivity judge and signed index (D5.6).
        Returns them in observation order; empty when nothing was stored.
        """
        return tuple(self._trace_store.get((candidate_id, task_id), ()))

    def signed_mechanism_index(self) -> SignedMechanismIndex:
        """IDX2: build the ranked complementary-parenthood index.

        Walks the TS2 cross-attempt store and files every diagnosed member:

        * strengths (``valence=-1``) from passing rollouts, clustered via
          ``assign_finding`` -- the SAME namespace faults use (D5.1);
        * faults (``valence=+1``) from failing rollouts' analyses, clustered
          via ``assign``.

        Honesty: unscorable rollouts, undiagnosed failures, and clusterer
        refusals contribute no entry -- an empty cluster is never padded.
        Raises when no cluster registry is configured rather than returning
        a silently empty index.
        """
        registry = self.cluster_registry
        if registry is None:
            raise ValueError(
                "no cluster registry configured: mechanism indexing requires "
                "an embedder-backed clusterer (pass embedder=/configure one)"
            )
        index = SignedMechanismIndex()
        for (candidate_id, task_id), rollouts in self._trace_store.items():
            clusterer = registry.clusterer_for(task_id)
            for rollout in rollouts:
                if (
                    rollout.trace is None
                    or rollout.score is None
                    or not rollout.scorable
                ):
                    continue
                # Strengths first: solvers are the reason this index exists.
                for finding in rollout.strengths:
                    assignment = clusterer.assign(finding)
                    if not assignment.cluster_id:
                        continue
                    index.add(
                        IndexEntry(
                            valence=-1,
                            severity=float(finding.severity or 0.0),
                            candidate_id=candidate_id,
                            task_id=task_id,
                            cluster_id=f"{task_id}:{assignment.cluster_id}",
                            artifact_ids=tuple(
                                sorted(
                                    {
                                        a
                                        for n in finding.blame_graph.nodes
                                        for a in n.artifacts
                                    }
                                )
                            ),
                            trace_id=rollout.trace.trace_id,
                        )
                    )
                analysis = rollout.analysis
                if analysis is None:
                    continue
                assignment = clusterer.assign(analysis)
                if not assignment.cluster_id:
                    continue
                index.add(
                    IndexEntry(
                        valence=1,
                        severity=float(analysis.severity),
                        candidate_id=candidate_id,
                        task_id=task_id,
                        cluster_id=f"{task_id}:{assignment.cluster_id}",
                        artifact_ids=tuple(
                            sorted(
                                {
                                    a
                                    for n in analysis.blame_graph.nodes
                                    for a in n.artifacts
                                }
                            )
                        ),
                        trace_id=rollout.trace.trace_id,
                    )
                )
        return index

    def _record_entropy_evidence(
        self, candidate_id: str, rollout: ObservedRollout
    ) -> None:
        """Record one rollout into the mechanism-keyed entropy tracker.

        Returns without recording when no mechanism can be established, rather
        than substituting a placeholder. Three distinct cases, all of which mean
        *this observation cannot support a per-mechanism variance claim*:

        * no analysis (an analyzer outage): there is no mechanism to key by;
        * an unassigned assignment (the cluster cap is full and the nearest
          cluster is below the join threshold): the clusterer explicitly refused,
          and inventing a cell here would file two unrelated faults together --
          the exact defect the refusal exists to prevent;
        * a blank mechanism id from any other source.

        Recording under a stand-in key would be worse than not recording: a cell
        that exists but means nothing is indistinguishable from a real one, and
        the floors would eventually declare it comparable.
        """
        score = rollout.score
        if score is None or not score.scorable:
            return
        cluster_ids = self._entropy_cluster_ids(rollout)
        if not cluster_ids:
            return
        for cluster_id in cluster_ids:
            self.entropy.record_score(
                task_id=rollout.task.task_id,
                mechanism_cluster_id=cluster_id,
                candidate_id=candidate_id,
                score=score.score,
            )
            # Promote to comparable only once this candidate clears the
            # per-candidate rollout floor in this cell. ``entropy()`` ignores
            # non-comparable candidates entirely, so promoting eagerly would let
            # a single rollout contribute to a variance the spec says it cannot
            # support.
            #
            # The count is kept here rather than read back from the tracker
            # because ``EntropyTracker`` exposes no per-candidate rollout count
            # publicly and is under a no-change constraint; reaching into
            # ``_cells`` would couple this to its private layout.
            seen_key = (rollout.task.task_id, cluster_id, candidate_id)
            self._entropy_rollout_counts[seen_key] = (
                self._entropy_rollout_counts.get(seen_key, 0) + 1
            )
            if (
                self._entropy_rollout_counts[seen_key]
                >= self.entropy.min_rollouts_per_candidate
            ):
                self.entropy.mark_comparable(
                    task_id=rollout.task.task_id,
                    mechanism_cluster_id=cluster_id,
                    candidate_id=candidate_id,
                )

    def _entropy_cluster_ids(self, rollout: ObservedRollout) -> tuple[str, ...]:
        """Mechanism keys for this rollout -- one per cluster actually assigned.

        A diagnosed failure keys one cell through ``assign(analysis)``, exactly
        as before. A passing rollout carries no analysis; since ?13 its
        positivity strengths key the cells they were actually assigned to, so
        an accepted child's measured quality is visible to cross-candidate
        variance on every mechanism it demonstrably solved. Refused assignments
        contribute nothing -- filing them would put unrelated faults in one
        cell -- and are reported rather than silently dropped.
        """
        task_id = rollout.task.task_id
        analysis = rollout.analysis
        strengths = rollout.strengths
        if analysis is None and not strengths:
            self._note_entropy_unavailable(
                task_id,
                "no analysis: the rollout was scored but not diagnosed",
                ENTROPY_UNAVAILABLE_CATEGORIES.NO_ANALYSIS,
            )
            return ()
        registry = self.cluster_registry
        if registry is None:
            self._note_entropy_unavailable(
                task_id,
                "no cluster registry configured on this runner",
                ENTROPY_UNAVAILABLE_CATEGORIES.NO_REGISTRY,
            )
            return ()
        clusterer = registry.clusterer_for(task_id)
        if analysis is not None:
            assignment = clusterer.assign(analysis)
            if not assignment.cluster_id:
                self._note_entropy_unavailable(
                    task_id,
                    assignment.unassigned_reason
                    or "the clusterer did not assign a mechanism",
                    ENTROPY_UNAVAILABLE_CATEGORIES.UNASSIGNED,
                )
                return ()
            # Namespaced by task so two tasks' ``c0`` are never confused.
            return (f"{task_id}:{assignment.cluster_id}",)
        ids: list[str] = []
        for strength in strengths:
            assignment = clusterer.assign_finding(strength)
            if assignment.cluster_id and assignment.cluster_id not in ids:
                ids.append(assignment.cluster_id)
        if not ids:
            self._note_entropy_unavailable(
                task_id,
                "every strength assignment was refused by the clusterer",
                ENTROPY_UNAVAILABLE_CATEGORIES.STRENGTHS_REFUSED,
            )
            return ()
        # Namespaced by task so two tasks' ``c0`` are never confused. Cells stay
        # indexed per task, which is what the (task, mechanism) key means.
        return tuple(f"{task_id}:{cid}" for cid in ids)

    def _entropy_cluster_id(self, rollout: ObservedRollout) -> str:
        """Single-key form of :meth:`_entropy_cluster_ids` (``""`` when none)."""
        ids = self._entropy_cluster_ids(rollout)
        return ids[0] if ids else ""

    def _note_entropy_unavailable(
        self, task_id: str, reason: str, category: str | None = None
    ) -> None:
        """Retain why a task's entropy evidence could not be filed.

        ``reason`` is prose for a human reading one task; ``category`` is a
        stable key for aggregation across a run. Both are kept because a rate
        without a cause is not actionable and a cause without a count does not
        show how widespread it is.
        """
        self._last_entropy_unavailable_reasons[task_id] = reason
        if category is not None:
            self._entropy_unavailable_categories[task_id] = category

    def entropy_availability(self) -> "EntropyAvailabilityReport":
        """Aggregate how often cross-candidate entropy was measurable.

        SV-12's last named remainder. Per-task reasons already existed; without
        this aggregate, a run in which **no** cell ever cleared the floors reads
        in the summary exactly like one where entropy drove diversity, because a
        measured zero and an unmeasurable cell both surface as ``H = 0.0``.

        Counted per **cell**, ``(task, mechanism)``, because that is the unit the
        floors apply to. A task whose mechanism could never be established has no
        cell at all, so it cannot be counted as one; it still contributes to
        ``reasons`` so a clustering outage is visible rather than appearing as
        "nothing to report".

        An **existing** cell that returns ``None`` is always ``FLOOR_UNMET``:
        ``EntropyTracker.entropy`` returns ``None`` only when the cell is absent
        or fails ``_meets_evidence_floor``, so for a cell that is present there
        is exactly one cause. The per-task category dict must *not* be consulted
        here -- it is keyed by task and last-write-wins, so a later undiagnosed
        rollout on a task whose cell was already filed would relabel that cell.
        Measured: a 4-attempt offline run reported ``no_analysis=3`` for three
        cells that existed, which is self-contradictory, because a cell only
        exists once a mechanism *was* assigned.
        """
        available = 0
        unavailable = 0
        reasons: dict[str, int] = {}
        cell_tasks: set[str] = set()

        for key in self.entropy.all_cells():
            cell_tasks.add(key.task_id)
            if self.entropy.entropy(key.task_id, key.mechanism_cluster_id) is None:
                unavailable += 1
                category = ENTROPY_UNAVAILABLE_CATEGORIES.FLOOR_UNMET
                reasons[category] = reasons.get(category, 0) + 1
            else:
                available += 1

        # Tasks that never produced a cell at all: no mechanism could be
        # established, so there is nothing to count as a cell, but the cause must
        # still be reported or a clustering outage would be invisible.
        for task_id, category in self._entropy_unavailable_categories.items():
            if task_id in cell_tasks:
                continue
            reasons[category] = reasons.get(category, 0) + 1

        return EntropyAvailabilityReport(
            cells_available=available,
            cells_unavailable=unavailable,
            reasons=dict(reasons),
        )

    def _embed_finding(
        self, finding: CausalFinding, task: EvolutionTask
    ) -> tuple[float, ...]:
        """Embed the mechanism and its attributed artifacts.

        The embedded text carries no expected contract: only the mechanism
        description and the attributed artifact IDs.

        ``task_id`` is deliberately **excluded**. A task name is not evidence
        about a failure *mechanism*, and including it makes two findings that
        describe the same fault on different tasks embed differently -- biasing
        the clusterer toward same-task grouping, which is the opposite of the
        cross-task evidence pooling the per-mechanism entropy floors need. The
        ``task`` argument is retained because cells remain indexed per task; only
        the embedded *text* drops it.
        """
        parts = [finding.mechanism_description or ""]
        parts.extend(finding.evidence_refs)
        text = " ".join(part for part in parts if part)
        embedder = self.embedder
        assert embedder is not None
        return tuple(embedder.embed(text))

    def _entropy_cell_for(self, task_id: str) -> tuple[str, float] | None:
        """The single mechanism cell that represents this task, or ``None``.

        One resolution point for both the entropy value and its tier. They must
        describe the *same* cell: ``raw_issue_quality`` treats the tier as an
        instruction about that specific number (``frontier_exploration`` damps it
        to ``frontier_weight``, ``skip`` zeroes it, otherwise full weight), so
        sourcing the value from the strongest cell while sourcing the tier from
        a different one applies the wrong weight -- measured promoting a
        ``frontier_exploration`` value to ``recombination_target``, from 30% to
        100%.

        The strongest cell that clears the floors wins, because the question the
        DPP asks is "how much disagreement is reachable on this task?". Cells
        below the floors return ``None`` from the tracker and are skipped rather
        than counted as zero: an unmeasured cell must not look like a
        measured-and-uniform one.
        """
        best: tuple[str, float] | None = None
        for key in self.entropy.all_cells():
            if key.task_id != task_id:
                continue
            value = self.entropy.entropy(task_id, key.mechanism_cluster_id)
            if value is None:
                continue
            if best is None or value > best[1]:
                best = (key.mechanism_cluster_id, value)
        return best

    def _cell_entropy(self, task_id: str) -> float:
        """Cross-candidate entropy for this task, from the entropy tracker.

        Reads :class:`EntropyTracker`, which is the single implementation of the
        spec's ``H(t, m) = Var * max(max_score, score_floor)`` *with* the
        evidence floors. Previously this recomputed variance inline over the pool
        score tensor filtered on the constant ``mechanism_cluster_id``, which
        measured the spread of one score per *candidate* inside a single
        synthetic bucket -- pooling candidates that failed for unrelated reasons
        -- and enforced no floors at all.
        """
        cell = self._entropy_cell_for(task_id)
        if cell is None:
            self._note_entropy_unavailable(
                task_id,
                self._last_entropy_unavailable_reasons.get(
                    task_id,
                    "no mechanism cell for this task meets the evidence floors "
                    f"(>={self.entropy.min_comparable_candidates} comparable "
                    f"candidates, >={self.entropy.min_rollouts_per_candidate} "
                    "rollouts each)",
                ),
                ENTROPY_UNAVAILABLE_CATEGORIES.FLOOR_UNMET,
            )
            return 0.0
        return cell[1]

    def entropy_unavailable_reason(self, task_id: str) -> str | None:
        """Why this task's entropy term is unavailable, or ``None`` if it is not.

        Exposed so a caller can distinguish "entropy measured zero" from
        "entropy could not be measured" -- the distinction the spec requires and
        the reason ``EntropyTracker.entropy`` returns ``None`` rather than 0.0.
        """
        return self._last_entropy_unavailable_reasons.get(task_id)

    def _entropy_tier(self, task_id: str) -> str:
        """The tier of the same cell :meth:`_cell_entropy` reports.

        ``skip`` when no mechanism cell for this task clears the floors -- a
        single sample must never contribute a high-variance signal, and
        ``raw_issue_quality`` zeroes the entropy component for this tier.
        Otherwise the tracker's own classification **for the cell that supplied
        the number**, so one instrument decides both the value and how it is
        weighted.

        With mechanism-keyed cells this returns ``skip`` **more often** than the
        previous constant-bucket version did, because the >=3 comparable
        candidates floor is genuinely harder to clear per mechanism than it was
        across one pooled cell. That is the correct direction: a
        correct-but-unavailable entropy term beats a confidently wrong one.
        """
        cell = self._entropy_cell_for(task_id)
        if cell is None:
            return "skip"
        return self.entropy.classify(task_id, cell[0])

    def _coverage_need(self, task_id: str) -> float:
        """Fraction of pool candidates lacking evidence for this task's cell."""
        entries = self.pool.all_entries()
        if not entries:
            return 0.0
        evaluated = sum(
            1
            for entry in entries
            if any(
                t_id == task_id
                and m_id == self.mechanism_cluster_id
                and cell.rollout_count >= 1
                for (t_id, m_id), cell in entry.score_tensor.items()
            )
        )
        return max(0.0, min(1.0, 1.0 - evaluated / len(entries)))

    def _pareto_relevance(self, candidate_id: str) -> float:
        return 1.0 if candidate_id in self.pool.pareto_frontier() else 0.0

    # ------------------------------------------------------------------ #
    # select_issues
    # ------------------------------------------------------------------ #
    def select_issues(
        self, issues: Sequence[TargetIssue], k: int = 1
    ) -> TargetIssueSelectionReport:
        """Select up to ``k`` issues via the target hierarchical DPP selector."""
        return self.selector.select(tuple(issues), k=k)

    # ------------------------------------------------------------------ #
    # select_parent
    # ------------------------------------------------------------------ #
    def select_parent(self) -> PoolEntry:
        """Sample a parent proportional to frequency, else fall back to base.

        ``frequency(c)`` sums ``severity * confidence`` over the cells ``c``
        wins. A pool with no winning cell (no comparable evidence yet) yields no
        mass, and the base harness is the only defensible parent.
        """
        frequencies = self.pool.parent_frequencies()
        weighted = [
            (candidate_id, weight)
            for candidate_id, weight in sorted(frequencies.items())
            if weight > 0.0
        ]
        if not weighted:
            return self.pool.base
        rng = self._rng
        assert rng is not None
        total = sum(weight for _, weight in weighted)
        draw = rng.random() * total  # type: ignore[attr-defined]
        cumulative = 0.0
        for candidate_id, weight in weighted:
            cumulative += weight
            if draw < cumulative:
                return self.pool.get(candidate_id)
        return self.pool.get(weighted[-1][0])

    # ------------------------------------------------------------------ #
    # propose_edits
    # ------------------------------------------------------------------ #
    def select_parents(self, k: int = 3) -> tuple[PoolEntry, ...]:
        """Select the primary parent plus up to ``k - 1`` donor parents.

        The primary keeps the architecture's frequency-proportional sampling and
        owns the workspace being written. Donors come from the Pareto frontier
        and are exposed read-only, so an editor can transplant a capability
        without the prompt growing with the pool.

        **SV-10: the primary is the already-observed parent when there is one.**
        ``select_parent`` consumes ``rng.random()``, so drawing here as well would
        make this a third independent draw within one attempt and could offer a
        primary that is neither the diagnosed nor the edited candidate.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be >= 1")
        primary = (
            self.pool.get(self._last_observed_parent_id)
            if self._last_observed_parent_id
            else self.select_parent()
        )
        if k == 1:
            return (primary,)
        donors: list[PoolEntry] = []
        for candidate_id in self.pool.pareto_frontier():
            if candidate_id == primary.candidate_id:
                continue
            donors.append(self.pool.get(candidate_id))
            if len(donors) >= k - 1:
                break
        return (primary, *donors)

    def propose_edits(
        self,
        parent_entry: PoolEntry,
        issue: TargetIssue,
        task: EvolutionTask,
        analysis: CausalAnalysis,
        attempt_id: str,
    ) -> tuple[CandidateWorkspace, EditorResponse | None, int, tuple[str, ...]]:
        """Materialize a workspace and obtain a validated editor response.

        The editor receives the primary parent plus up to ``donor_count`` donor
        parents, so one call can refine the primary or transplant a capability
        from a donor. Donors are read-only: writes always land in the primary's
        workspace.

        The fourth return element is the donor parents the editor actually read.
        It comes from the editor's own tool-execution ledger, never from its
        prose, so lineage cannot claim a donor that was merely offered.
        """
        workspace = self.adapter.materialize_candidate(
            parent_entry.version, attempt_id
        )
        write_set = tuple(issue.writable_artifact_ids)
        current = self.adapter.read_artifacts(parent_entry.version, write_set)

        entries = self.select_parents(k=self.donor_count + 1)
        parents = tuple(
            ParentContext(
                candidate_id=entry.candidate_id,
                version=entry.version,
                is_primary=entry.candidate_id == parent_entry.candidate_id,
                score_summary={
                    t_id: cell.mean
                    for (t_id, _m), cell in entry.score_tensor.items()
                },
                issues=self._issues_for_parent(entry.candidate_id),
            )
            for entry in entries
        )
        # select_parents samples independently, so the chosen parent may not be
        # in the returned set. The workspace owner must always be the primary.
        if not any(p.is_primary for p in parents):
            parents = (
                ParentContext(
                    candidate_id=parent_entry.candidate_id,
                    version=parent_entry.version,
                    is_primary=True,
                    score_summary={
                        t_id: cell.mean
                        for (t_id, _m), cell in parent_entry.score_tensor.items()
                    },
                    issues=self._issues_for_parent(parent_entry.candidate_id),
                ),
                *(p for p in parents if not p.is_primary),
            )

        request = EditorRequest(
            base_workspace=workspace,
            task=task,
            analysis=analysis,
            issue_id=issue.issue_id,
            write_set=write_set,
            current_artifacts=dict(current),
            parents=parents,
            creatable_prefixes=getattr(self.adapter, "creatable_prefixes", ()),
            pool_created_count=self._pool_created_count(),
            # Prior attempts on this issue, so the editor is told what has
            # already been tried instead of rediscovering it.
            history_refs=tuple(
                a.attempt_id for a in self.edit_memory.for_issue(issue.issue_id)
            ),
        )
        repair = repair_once_then_classify(self.editor, request)
        observed = tuple(getattr(self.editor, "last_parents_read", ()))
        return workspace, repair.response, repair.correction_requests, observed

    def _issues_for_parent(self, candidate_id: str) -> tuple[TargetIssue, ...]:
        """This candidate's diagnosed faults, or ``()`` when it has none.

        Only the observed parent has a diagnosis: ``build_issues`` runs the
        analyzer on one candidate per attempt, so a donor legitimately returns
        empty. Empty means *no diagnosis yet* -- it is not an error, and it must
        not be filled with the observed parent's faults, which would attribute
        one candidate's weaknesses to another.
        """
        if not candidate_id or candidate_id != self._last_observed_parent_id:
            return ()
        return self._last_parent_issues

    def _pool_created_count(self) -> int:
        """Generated artifacts already present, for the creation cap."""
        counter = getattr(self.adapter, "created_artifact_count", None)
        if counter is None:
            return 0
        return max(
            (counter(entry.version) for entry in self.pool.all_entries()),
            default=0,
        )

    # ------------------------------------------------------------------ #
    # validate
    # ------------------------------------------------------------------ #
    def validate(
        self,
        workspace: CandidateWorkspace,
        origin_task: EvolutionTask,
        regression_tasks: Sequence[EvolutionTask] = (),
    ) -> FocusedValidationReport:
        """Run origin and regression probes against the edited workspace.

        A probe whose rollout produced no measurement is **dropped**, not
        recorded. Recording it as a 0.0 would invent a regression; recording it
        as a pass would accept an edit on evidence that does not exist. Both are
        worse than an absent probe, which merely leaves the edit unsupported --
        and an unsupported edit is rejected by :func:`decide_acceptance`, because
        an empty origin set produces zero weighted net gain. The count is
        exposed via :attr:`unscorable_probe_count` so a run can report how much
        of its validation evidence went missing.
        """
        planner = ValidationPlanner(
            origin_task=origin_task,
            regression_tasks=tuple(regression_tasks),
        )
        probes = planner.build_probes()
        observed = self.rollout_group(
            workspace.version,
            tuple(probe.task for probe in probes),
            prefix=f"{workspace.attempt_id}-probe",
        )
        origin: list[ValidationResult] = []
        regression: list[ValidationResult] = []
        # SV-13: retain the child's traces by task. These rollouts already
        # happened -- the probe set is origin + every regression task, i.e. the
        # whole coreset -- and ``ValidationResult`` keeps only ``trace_id``, so
        # without this the retirement judge would have to re-roll the child on
        # every task it was just measured on. Retaining them makes generational
        # retirement cost judge calls only, and a judge call is far cheaper than
        # a rollout.
        self._last_validation_traces = {}
        # SV-14: same retention pattern as the traces above, for the child's
        # own analyses and its scorable rollouts. Reset per call so a later
        # commit can never see a previous attempt's diagnoses.
        self._last_validation_analyses = {}
        validation_rollouts: list[ObservedRollout] = []
        for probe, rollout in zip(probes, observed, strict=True):
            if rollout.trace is None or rollout.score is None or not rollout.scorable:
                self._unscorable_probes += 1
                continue
            self._last_validation_traces[probe.task.task_id] = rollout.trace
            if rollout.analysis is not None:
                self._last_validation_analyses[probe.task.task_id] = rollout.analysis
            validation_rollouts.append(rollout)
            # TS2: cross-attempt retention, no quality gate (see field docs).
            self._record_stored_trace(workspace.version, rollout)
            outcome = ValidationResult(
                kind=probe.kind,
                task_id=probe.task.task_id,
                score=rollout.score.score,
                trace_id=rollout.trace.trace_id,
                passed=rollout.score.score >= 0.5,
                mechanism_cluster_id=self.mechanism_cluster_id,
            )
            if probe.kind is ValidationKind.REGRESSION:
                regression.append(outcome)
            else:
                origin.append(outcome)
        self._last_validation_rollouts = tuple(validation_rollouts)
        return FocusedValidationReport(
            origin=tuple(origin), worked=(), regression=tuple(regression)
        )

    # ------------------------------------------------------------------ #
    # commit_to_pool
    # ------------------------------------------------------------------ #
    def _commit_single_parent_for_test(self) -> PoolEntry:
        """Test seam: commit the base's workspace with no extra parents."""
        return self._commit_for_test(())

    def _commit_with_extra_parents_for_test(
        self, extra_parent_ids: Sequence[str]
    ) -> PoolEntry:
        """Test seam: commit with observed donor parents."""
        return self._commit_for_test(extra_parent_ids)

    def _commit_for_test(self, extra_parent_ids: Sequence[str]) -> PoolEntry:
        from agent_evolve.core.editor import (
            FocusedValidationReport as _Report,
        )

        parent = self.pool.base
        attempt_id = self._next_attempt_id()
        workspace = self.adapter.materialize_candidate(parent.version, attempt_id)
        return self.commit_to_pool(
            parent,
            workspace,
            attempt_id,
            _Report(origin=(), worked=(), regression=()),
            extra_parent_ids=extra_parent_ids,
        )

    def commit_to_pool(
        self,
        parent_entry: PoolEntry,
        workspace: CandidateWorkspace,
        attempt_id: str,
        report: FocusedValidationReport,
        validation_rollouts: Sequence[ObservedRollout] = (),
        extra_parent_ids: Sequence[str] = (),
    ) -> PoolEntry:
        """Publish an accepted candidate with its post-edit score evidence.

        ``extra_parent_ids`` carries donor parents the editor actually read.
        They come from tool-execution evidence, never from editor narration, so
        lineage cannot claim a donor the editor merely had access to.

        SV-14: provenance is per task and describes **the child**. Where
        ``validation_rollouts`` carries a diagnosis for a result's task, that
        diagnosis supplies ``analyzer_model_id`` and ``blame_confidence``;
        where it does not -- a passing probe legitimately has none -- absence
        is recorded explicitly (empty analyzer id, ``blame_confidence=None``).
        The pre-SV-14 behaviour copied one parent analysis across every cell of
        the new candidate, attributing the parent's diagnosis to the child,
        regression tasks included. The diagnosed rollouts are also filed into
        the mechanism-keyed entropy tracker under the child, so an accepted
        offspring no longer stays invisible to cross-candidate variance until
        some later attempt happens to observe it. The pool's score-tensor key
        stays constant; only the tracker keys by mechanism.
        """
        parent_ids = tuple(
            sorted({parent_entry.candidate_id, *extra_parent_ids})
        )
        candidate = EvolutionCandidate(
            candidate_id=workspace.version,
            version=workspace.version,
            artifact_hashes={
                d.artifact_id: d.version_hash
                for d in self.adapter.artifact_inventory(workspace.version)
            },
            parent_ids=parent_ids,
            ancestor_ids=tuple(
                sorted(set(parent_entry.candidate.ancestor_ids) | set(parent_ids))
            ),
            attempt_ids=(attempt_id,),
        )
        entry = self.pool.add_candidate(candidate, origin_attempt_ids=(attempt_id,))
        analyses_by_task = {
            ro.task.task_id: ro.analysis
            for ro in validation_rollouts
            if ro.scorable and ro.analysis is not None
        }
        for result in report.all_results:
            cell = entry.cell(result.task_id, self.mechanism_cluster_id)
            child_analysis = analyses_by_task.get(result.task_id)
            self.pool.record_score(
                entry.candidate_id,
                result.score,
                ScoreProvenance(
                    task_id=result.task_id,
                    mechanism_cluster_id=self.mechanism_cluster_id,
                    trace_id=result.trace_id,
                    rollout_seq=cell.rollout_count,
                    # SV-14: the child's own diagnosis on this task, or
                    # explicit absence. Never another candidate's analysis.
                    analyzer_model_id=(
                        child_analysis.analyzer_model_id
                        if child_analysis is not None
                        else ""
                    ),
                    # The grader that produced this score, not the analyzer's
                    # judge: the validation results came from ``validate``, which
                    # measures with ``resolved_scorer``.
                    judge_model_id=self.grader_name,
                    blame_confidence=(
                        min(1.0, child_analysis.blame_graph.total_blame())
                        if child_analysis is not None
                        else None
                    ),
                    blame_stability=1.0,
                    artifact_versions=dict(candidate.artifact_hashes),
                ),
            )
        # SV-14 step 3: file the offspring's mechanism evidence at commit.
        # ``_record_entropy_evidence`` skips honestly when a rollout carries no
        # usable mechanism (no diagnosis, unassigned cluster), so passing every
        # scorable rollout records exactly what exists -- nothing invented.
        for rollout in validation_rollouts:
            self._record_entropy_evidence(entry.candidate_id, rollout)
        return entry

    def measure(
        self, version: str, tasks: Sequence[EvolutionTask], *, prefix: str = "measure"
    ) -> ScoreTally:
        """Score one version over ``tasks`` and report the tally with its denominator.

        Used for the before/after numbers a run reports. It records nothing in
        the pool: a measurement pass is an observation of a version, and folding
        it into the score tensor would double-count the evidence that selection
        reads.
        """
        observed = self.rollout_group(version, tasks, prefix=prefix)
        return tally_scores(
            tuple(r.score for r in observed if r.score is not None),
            grader_name=self.grader_name,
        )

    # ------------------------------------------------------------------ #
    # One attempt
    # ------------------------------------------------------------------ #
    def run_attempt(self, tasks: Sequence[EvolutionTask]) -> GepaAttemptOutcome:
        """Execute one full GEPA attempt over the given task coreset.

        The attempt is reserved against the run budget first, so ``--max-attempts``
        and ``--max-accepted-edits`` refuse before any rollout or editor call is
        issued. Reaching a cap is reported as a normal non-accepted outcome, not
        raised: a planned stop must not look like a crash, and the caller's loop
        keeps its already-collected evidence.
        """
        limits = self.config.budgets if self.config is not None else None
        if limits is not None:
            exhausted = (
                limits.max_attempts is not None
                and self._budget.attempts >= limits.max_attempts
            ) or (
                limits.max_accepted_edits is not None
                and self._budget.accepted_edits >= limits.max_accepted_edits
            )
            if exhausted:
                return GepaAttemptOutcome(
                    attempt_id=self._next_attempt_id(),
                    issue_id="",
                    parent_candidate_id=self.pool.base.candidate_id,
                    result_candidate_id=None,
                    status=AttemptStatus.PENDING,
                    accepted=False,
                    weighted_net_gain=0.0,
                    reason="budget exhausted: attempt cap reached",
                )
            self._budget.attempts += 1
        self._iteration += 1
        issues = self.build_issues(tasks)
        report = self.select_issues(issues, k=1)
        selected = report.items  # type: ignore[attr-defined]
        if not selected:
            return GepaAttemptOutcome(
                attempt_id=self._next_attempt_id(),
                issue_id="",
                parent_candidate_id=self.pool.base.candidate_id,
                result_candidate_id=None,
                status=AttemptStatus.PENDING,
                accepted=False,
                weighted_net_gain=0.0,
                reason="no evidence-backed work item available",
                fallback_reason=report.fallback_reason,  # type: ignore[attr-defined]
            )

        issue = selected[0]
        task = self._task_for(issue, tasks)
        # SV-10: reuse the parent build_issues already observed rather than
        # drawing again. select_parent consumes rng.random(), so a second draw is
        # independent and can name a different candidate -- which would diagnose
        # one parent's faults and then materialize another parent's workspace.
        # Falls back to a draw only if no observation was recorded.
        parent = (
            self.pool.get(self._last_observed_parent_id)
            if self._last_observed_parent_id
            else self.select_parent()
        )
        attempt_id = self._next_attempt_id()
        parent_rollout = self.rollout_group(
            parent.version, (task,), prefix=f"obs-{parent.candidate_id}"
        )[0]
        if parent_rollout.score is None or not parent_rollout.scorable:
            # The parent's own rollout produced no measurement, so there is no
            # evidence to hand the editor and no baseline to compare an edit
            # against. Reported as "no work item" rather than attempted on a
            # fabricated zero.
            outcome = GepaAttemptOutcome(
                attempt_id=attempt_id,
                issue_id=issue.issue_id,
                parent_candidate_id=parent.candidate_id,
                result_candidate_id=None,
                status=AttemptStatus.PENDING,
                accepted=False,
                weighted_net_gain=0.0,
                reason=(
                    "the parent rollout produced no measurement: "
                    + (
                        parent_rollout.error
                        or (
                            parent_rollout.score.reason
                            if parent_rollout.score is not None
                            else "unknown"
                        )
                    )
                ),
                fallback_reason=report.fallback_reason,  # type: ignore[attr-defined]
            )
            self._persist_attempt(outcome, 0)
            return outcome
        analysis = parent_rollout.analysis
        if analysis is None:
            analysis = (
                empty_analysis()
                if parent_rollout.score.passed
                else abstained_analysis(
                    "insufficient_evidence",
                    score=parent_rollout.score.score,
                    evidence=(
                        parent_rollout.error
                        or "no analysis was produced for the parent rollout",
                    ),
                )
            )

        workspace, response, corrections, observed_parents = self.propose_edits(
            parent, issue, task, analysis, attempt_id
        )
        if response is None:
            outcome = GepaAttemptOutcome(
                attempt_id=attempt_id,
                issue_id=issue.issue_id,
                parent_candidate_id=parent.candidate_id,
                result_candidate_id=None,
                status=AttemptStatus.REJECTED,
                accepted=False,
                weighted_net_gain=0.0,
                reason="editor response malformed after one correction request",
                fallback_reason=report.fallback_reason,  # type: ignore[attr-defined]
            )
            self._persist_attempt(outcome, corrections)
            return outcome

        self.adapter.apply_structured_edits(workspace, response.edits)
        regression_tasks = tuple(t for t in tasks if t.task_id != task.task_id)
        validation = self.validate(workspace, task, regression_tasks)
        decision = decide_acceptance(
            validation,
            protected_floors=self.protected_floors,
            net_gain_threshold=self.net_gain_threshold,
        )

        result_candidate_id: str | None = None
        retired_parent_id: str | None = None
        retirement_reason = ""
        if decision.accepted:
            self._budget.accepted_edits += 1
            # SV-14: the child commits with its own validation-time diagnoses
            # (and files its own entropy evidence); the parent's ``analysis``
            # stays out of the offspring's provenance.
            committed = self.commit_to_pool(
                parent,
                workspace,
                attempt_id,
                validation,
                validation_rollouts=self._last_validation_rollouts,
                extra_parent_ids=observed_parents,
            )
            result_candidate_id = committed.candidate_id
            retired_parent_id, retirement_reason = self._maybe_retire_parent(
                parent, committed, tasks
            )

        outcome = GepaAttemptOutcome(
            attempt_id=attempt_id,
            issue_id=issue.issue_id,
            parent_candidate_id=parent.candidate_id,
            result_candidate_id=result_candidate_id,
            status=decision.status,
            accepted=decision.accepted,
            weighted_net_gain=decision.weighted_net_gain,
            reason=decision.reason,
            artifact_ids=tuple(e.artifact_id for e in response.edits),
            fallback_reason=report.fallback_reason,  # type: ignore[attr-defined]
            retired_parent_id=retired_parent_id,
            retirement_reason=retirement_reason,
        )
        self._record_in_edit_memory(
            outcome, workspace, response, validation, decision
        )
        self._persist_attempt(outcome, corrections)
        return outcome

    def _maybe_retire_parent(
        self,
        parent: PoolEntry,
        child: PoolEntry,
        tasks: Sequence[EvolutionTask],
    ) -> tuple[str | None, str]:
        """Retire ``parent`` when the judge prefers ``child`` on the coreset.

        SV-13. An offspring is generated to fix its parent's diagnosed faults, so
        a child the judge prefers has made the parent redundant *as a parent* --
        continuing to breed from a version its own descendant improved on spends
        rollouts to re-derive a fix that already exists.

        **Costs judge calls only.** Both trace sets are reused: the parent's from
        ``build_issues`` (which after SV-11 observes the selected parent) and the
        child's from ``validate`` (whose probe set is origin + every regression
        task, i.e. the coreset). ``2k`` model calls, zero extra rollouts.

        **Never blocks the attempt.** No judge, an unavailable verdict, a missing
        trace, or a raising judge all leave the parent alive and the committed
        candidate intact. The edit is the expensive artifact; an optional
        pool-shaping step must not be able to discard it. Structural refusals
        (retiring the last live entry) are enforced by :meth:`PersistentPool.retire`
        and reported here rather than raised.
        """
        if self.compare_preference is None:
            return None, "no preference judge configured; retirement disabled"

        judged_tasks = tuple(
            t
            for t in tasks
            if t.task_id in self._last_observation_traces
            and t.task_id in self._last_validation_traces
        )
        decision = decide_retirement(
            parent_id=parent.candidate_id,
            child_id=child.candidate_id,
            tasks=judged_tasks if judged_tasks else tasks,
            parent_traces=self._last_observation_traces,
            child_traces=self._last_validation_traces,
            compare=self.compare_preference,
        )
        if not decision.should_retire:
            return None, decision.reason
        # The verdict is a measurement of the child against the version it was
        # derived from, which is exactly the evidence the SV-4 promotion gate asks
        # for. Recording it here is what lets a *genetic* offspring be promotable
        # at all: only the RHO path calls ``record_preference``, so without this a
        # judged genetic candidate would be retired-parent-superseding and still
        # ineligible for export -- an incoherent pair.
        #
        # The baseline differs by path and that is deliberate: RHO measures against
        # the incumbent base, retirement against the immediate parent. Both express
        # "preferred over the version it came from", which is the property the gate
        # is testing.
        if decision.mean_preference is not None:
            self.pool.record_preference(
                child.candidate_id,
                decision.mean_preference,
                available=decision.judged,
                unavailable=decision.unavailable,
            )
        try:
            self.pool.retire(
                parent.candidate_id, superseded_by=child.candidate_id
            )
        except (ValueError, KeyError) as exc:
            # Structural refusal (e.g. the parent is the only live entry). The
            # committed candidate stands; only the pool shaping was declined.
            return None, f"retirement declined: {exc}"
        return parent.candidate_id, decision.reason

    def _record_in_edit_memory(
        self,
        outcome: GepaAttemptOutcome,
        workspace: CandidateWorkspace,
        response: EditorResponse,
        report: FocusedValidationReport,
        decision: AcceptanceDecision,
    ) -> None:
        """Record a completed attempt so the next editor call can see it.

        Recorded for **rejected and regressed attempts too, not only accepted
        ones**: the whole point of history is "do not repeat a strategy that
        already failed", so the failures are the load-bearing entries.

        Also the only path that charges :class:`RetryBudget`, whose counter lives
        inside :meth:`EditMemory.record`.

        A duplicate ``attempt_id`` would raise, which must not turn a completed
        attempt into a crash; ids come from :meth:`_next_attempt_id` and are
        unique per run, so a collision means two runners share one memory. That
        is a wiring defect worth surfacing, but not at the cost of the attempt's
        own result, so it is swallowed here and left to tests to catch.
        """
        attempt = build_attempt(
            attempt_id=outcome.attempt_id,
            candidate_id=workspace.version,
            issue_id=outcome.issue_id,
            response=response,
            evidence_refs=(),
            history_refs=(),
            report=report,
            decision=decision,
        )
        try:
            record_attempt(self.edit_memory, attempt, workspace)
        except ValueError:
            # Duplicate attempt_id: already recorded. Nothing to add.
            pass

    def _task_for(
        self, issue: TargetIssue, tasks: Sequence[EvolutionTask]
    ) -> EvolutionTask:
        task_id = issue.task_id
        for task in tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f"selected issue references unknown task: {task_id!r}")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _persist_attempt(
        self, outcome: GepaAttemptOutcome, correction_requests: int
    ) -> None:
        """Persist a sanitized attempt record when storage is configured.

        Only references and decisions are written. Task inputs, expected
        contracts, editor payloads, and raw traces never reach storage.
        """
        if self.storage is None:
            return
        self.storage.write_record(  # type: ignore[attr-defined]
            "attempts",
            outcome.attempt_id,
            {
                "attempt_id": outcome.attempt_id,
                "issue_id": outcome.issue_id,
                "parent_candidate_id": outcome.parent_candidate_id,
                "result_candidate_id": outcome.result_candidate_id,
                "status": outcome.status.value,
                "accepted": outcome.accepted,
                "weighted_net_gain": outcome.weighted_net_gain,
                "reason": outcome.reason,
                "artifact_ids": list(outcome.artifact_ids),
                "correction_requests": correction_requests,
                "mechanism_cluster_id": self.mechanism_cluster_id,
                "selection_fallback_reason": outcome.fallback_reason,
            },
        )

    # ------------------------------------------------------------------ #
    # N attempts
    # ------------------------------------------------------------------ #
    def run(
        self, tasks: Sequence[EvolutionTask], n_attempts: int
    ) -> GepaRunResult:
        """Run ``n_attempts`` sequential GEPA attempts and select a champion."""
        if isinstance(n_attempts, bool) or not isinstance(n_attempts, int):
            raise ValueError("n_attempts must be a positive integer")
        if n_attempts < 1:
            raise ValueError("n_attempts must be a positive integer")
        if not tasks:
            raise ValueError("tasks must not be empty")

        outcomes = [self.run_attempt(tasks) for _ in range(n_attempts)]
        try:
            champion = self.pool.select_champion(config=self.config)
        except ValueError:
            champion = None
        return GepaRunResult(
            attempts=tuple(outcomes),
            champion=champion,
            pool_size=len(self.pool),
            pareto_frontier=self.pool.pareto_frontier(),
            entropy_availability=self.entropy_availability(),
        )
