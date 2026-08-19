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
    MechanismEmbedder,
)
from agent_evolve.core.config import BudgetLimits, BudgetUsage, ResolvedConfig
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
from agent_evolve.core.fake_editor import FakeEditor
from agent_evolve.core.issues import (
    DEFAULT_SCORE_FLOOR as TARGET_SCORE_FLOOR,
    DEFAULT_THETA as TARGET_THETA,
    HierarchicalDPPSelector as TargetIssueSelector,
    Issue as TargetIssue,
    IssueSelectionReport as TargetIssueSelectionReport,
    build_issue as build_target_issue,
)
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


@dataclass(frozen=True, slots=True)
class GepaRunResult:
    """Summary of ``n_attempts`` sequential GEPA attempts."""

    attempts: tuple[GepaAttemptOutcome, ...]
    champion: ChampionReport | None
    pool_size: int
    pareto_frontier: tuple[str, ...]

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
        self._rng = random.Random(self.seed)
        config = self.config
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

        observed: list[ObservedRollout] = []
        for outcome, score in scored:
            analysis, error = analyses.get(
                (outcome.task.task_id, id(outcome)), (None, "")
            )
            observed.append(
                ObservedRollout(
                    task=outcome.task,
                    trace=outcome.trace,
                    score=score,
                    analysis=analysis,
                    error=outcome.error or error,
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
            for outcome, score in items:
                assert outcome.trace is not None  # filtered by the caller
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
        runner = ParallelAnalysisRunner(
            analyzer_factory=factory, max_workers=self.analyzer_workers
        )
        outcomes = runner.run(reports)
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
            analysis = analysis_from_finding(
                findings[0],
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
        """Build trace-backed issues for every task the base currently fails.

        Returns target :class:`agent_evolve.core.issues.Issue` values. A task the
        base already satisfies produces no issue, and a finding with no writable
        attribution is dropped by :func:`build_issue` rather than ranked.

        A task whose rollout produced no measurement yields no issue **and no
        score**. It is neither an observed failure to diagnose nor a data point
        to record: a broken harness must not look like a candidate that answered
        wrongly.
        """
        base = self.pool.base
        inventory = self.adapter.artifact_inventory(base.version)
        write_set = self._writable_artifact_ids(base.version)
        observed = self.rollout_group(
            base.version, tasks, prefix=f"obs-{base.candidate_id}"
        )
        out: list[TargetIssue] = []
        for rollout in observed:
            if rollout.trace is None or rollout.score is None or not rollout.scorable:
                continue
            self._record_rollout_score(base.candidate_id, rollout)
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
                candidate_id=base.candidate_id,
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
                pareto_relevance=self._pareto_relevance(base.candidate_id),
                embedding=self._embed_finding(finding, rollout.task),
                lineage=base.version,
                entropy_tier=self._entropy_tier(rollout.task.task_id),
            )
            if issue is not None:
                out.append(issue)
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
                    analysis.analyzer_model_id if analysis is not None else ""
                ),
                # The grader that produced this number, recorded as the judge:
                # a score whose grader is unnamed cannot be compared later.
                judge_model_id=score.grader_name,
                blame_confidence=(
                    min(1.0, analysis.blame_graph.total_blame())
                    if analysis is not None
                    else 0.0
                ),
                blame_stability=1.0,
                artifact_versions=dict(entry.candidate.artifact_hashes),
            ),
        )

    def _embed_finding(
        self, finding: CausalFinding, task: EvolutionTask
    ) -> tuple[float, ...]:
        """Embed mechanism + task + artifact context.

        The embedded text carries no expected contract: only the mechanism
        description, the task ID, and the attributed artifact IDs.
        """
        parts = [finding.mechanism_description or "", task.task_id]
        parts.extend(finding.evidence_refs)
        text = " ".join(part for part in parts if part)
        embedder = self.embedder
        assert embedder is not None
        return tuple(embedder.embed(text))

    def _cell_entropy(self, task_id: str) -> float:
        """Population variance of comparable scores for this task's cells.

        Evidence floors are enforced by :meth:`_entropy_tier`; this is the raw
        statistic only.
        """
        scores = [
            cell.mean
            for entry in self.pool.all_entries()
            for (t_id, m_id), cell in entry.score_tensor.items()
            if t_id == task_id
            and m_id == self.mechanism_cluster_id
            and cell.rollout_count >= 1
        ]
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        variance = max(0.0, sum(s * s for s in scores) / len(scores) - mean * mean)
        floor = (
            getattr(self.config, "entropy_score_floor", 0.15)
            if self.config is not None
            else 0.15
        )
        return variance * max(max(scores), floor)

    def _entropy_tier(self, task_id: str) -> str:
        """Resolve the entropy tier from the evidence floors.

        ``skip`` when the comparable-candidate floor is unmet (a single sample
        must never contribute a high-variance signal), ``frontier_exploration``
        when variance is meaningful but the best score is below the
        recombination threshold, otherwise ``recombination_target``.
        """
        min_candidates = (
            getattr(self.config, "entropy_min_comparable_candidates", 3)
            if self.config is not None
            else 3
        )
        threshold = (
            getattr(self.config, "entropy_recombination_score_threshold", 0.30)
            if self.config is not None
            else 0.30
        )
        scores = [
            cell.mean
            for entry in self.pool.all_entries()
            for (t_id, m_id), cell in entry.score_tensor.items()
            if t_id == task_id
            and m_id == self.mechanism_cluster_id
            and cell.rollout_count >= 1
        ]
        if len(scores) < min_candidates:
            return "skip"
        if max(scores) < threshold:
            return "frontier_exploration"
        return "recombination_target"

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
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be >= 1")
        primary = self.select_parent()
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
            creatable_prefix=getattr(self.adapter, "creatable_prefix", ""),
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
        for probe, rollout in zip(probes, observed, strict=True):
            if rollout.trace is None or rollout.score is None or not rollout.scorable:
                self._unscorable_probes += 1
                continue
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
            empty_analysis(),
            extra_parent_ids=extra_parent_ids,
        )

    def commit_to_pool(
        self,
        parent_entry: PoolEntry,
        workspace: CandidateWorkspace,
        attempt_id: str,
        report: FocusedValidationReport,
        analysis: CausalAnalysis,
        extra_parent_ids: Sequence[str] = (),
    ) -> PoolEntry:
        """Publish an accepted candidate with its post-edit score evidence.

        ``extra_parent_ids`` carries donor parents the editor actually read.
        They come from tool-execution evidence, never from editor narration, so
        lineage cannot claim a donor the editor merely had access to.
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
        for result in report.all_results:
            cell = entry.cell(result.task_id, self.mechanism_cluster_id)
            self.pool.record_score(
                entry.candidate_id,
                result.score,
                ScoreProvenance(
                    task_id=result.task_id,
                    mechanism_cluster_id=self.mechanism_cluster_id,
                    trace_id=result.trace_id,
                    rollout_seq=cell.rollout_count,
                    analyzer_model_id=analysis.analyzer_model_id,
                    # The grader that produced this score, not the analyzer's
                    # judge: the validation results came from ``validate``, which
                    # measures with ``resolved_scorer``.
                    judge_model_id=self.grader_name,
                    blame_confidence=min(1.0, analysis.blame_graph.total_blame()),
                    blame_stability=1.0,
                    artifact_versions=dict(candidate.artifact_hashes),
                ),
            )
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
        parent = self.select_parent()
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
        if decision.accepted:
            self._budget.accepted_edits += 1
            committed = self.commit_to_pool(
                parent,
                workspace,
                attempt_id,
                validation,
                analysis,
                extra_parent_ids=observed_parents,
            )
            result_candidate_id = committed.candidate_id

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
        )
        self._record_in_edit_memory(
            outcome, workspace, response, validation, decision
        )
        self._persist_attempt(outcome, corrections)
        return outcome

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
        )
