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
from typing import Iterable, Mapping, Sequence

from agent_evolve.core.analyzer import AnalyzerJudge, FakeAnalyzerJudge
from agent_evolve.core.blame import (
    BlameGraph,
    BlameNode,
    CausalAnalysis,
    CausalFinding,
    empty_analysis,
)
from agent_evolve.core.clustering import (
    ClusterRegistry,
    LexicalEmbedder,
    MechanismClusterer,
    MechanismEmbedder,
)
from agent_evolve.core.config import ResolvedConfig
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
    _iteration: int = 0
    _attempt_seq: int = 0

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
            analysis = self.analyzer_judge.analyze(task, trace)
        else:
            # Minimal profile: just use the substring check to score.
            score = 1.0 if task.expected_contract.get("expected_substring", "") in trace.final_output else 0.0
            if score == 1.0:
                analysis = empty_analysis()
            else:
                # Minimal: blame the first actor in the trace, if any.
                actor = next(
                    (e.actor_id for e in trace.events if e.actor_id),
                    "unknown",
                )
                from agent_evolve.core.blame import BlameGraph, BlameNode
                analysis = CausalAnalysis(
                    mechanism=f"failed-to-match-{task.task_id}",
                    severity=1.0,
                    score=0.0,
                    blame_graph=BlameGraph(
                        nodes=(BlameNode(actor_id=actor, blame=1.0, artifacts=()),
                    )),
                )
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
                # Re-analyze the worst rollout for the issue.
                worst_prov = min(cell.provenance, key=lambda p: 0.0)
                # We don't have the trace anymore; synthesize an analysis
                # from the score.
                from agent_evolve.core.blame import BlameGraph, BlameNode
                fake_analysis = CausalAnalysis(
                    mechanism=f"base-failed-{t_id}-{m_id}",
                    severity=1.0 - cell.max,
                    score=cell.max,
                    blame_graph=BlameGraph(
                        nodes=(BlameNode(actor_id="agent", blame=1.0, artifacts=()),)
                    ),
                )
                issues.append((task, f"{t_id}:{m_id}", fake_analysis))

        # 3. Build the write set from the base inventory.
        write_set = tuple(
            d.artifact_id
            for d in self.adapter.artifact_inventory(base_entry.version)
            if d.writable
        )

        # 4. Run attempts (sequential or parallel).
        if self.profile.use_parallel_batch:
            batch_results = self._run_parallel_attempts(
                base_entry, issues, write_set
            )
            for attempt, decision in batch_results:
                attempts.append(attempt)
                if decision.accepted:
                    accepted.append(attempt.attempt_id)
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
    # Donor parents offered to the editor alongside the primary (spec §7).
    donor_count: int = 2
    _selector: TargetIssueSelector | None = field(default=None, init=False, repr=False)
    _rng: random.Random | None = field(default=None, init=False, repr=False)
    _iteration: int = field(default=0, init=False, repr=False)
    _attempt_seq: int = field(default=0, init=False, repr=False)
    _probe_seq: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.mechanism_cluster_id:
            raise ValueError("mechanism_cluster_id is required")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.embedder is None:
            self.embedder = LexicalEmbedder()
        self._rng = random.Random(self.seed)
        config = self.config
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
    def observe(
        self, entry: PoolEntry, task: EvolutionTask
    ) -> tuple[ExecutionTrace, CausalAnalysis]:
        """Roll a candidate out on one task and analyze the resulting trace.

        The rollout uses a throwaway workspace materialized from the candidate's
        version so the candidate's own artifacts are never written.
        """
        probe_id = self._next_probe_id(f"obs-{entry.candidate_id}")
        workspace = self.adapter.materialize_candidate(entry.version, probe_id)
        result = self.adapter.run_full_rollout(workspace, task, probe_id)
        trace = self.adapter.capture_trace(result)
        analysis = self.analyzer_judge.analyze(task, trace)
        return trace, analysis

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
        """
        base = self.pool.base
        inventory = self.adapter.artifact_inventory(base.version)
        write_set = self._writable_artifact_ids(base.version)
        out: list[TargetIssue] = []
        for task in tasks:
            trace, analysis = self.observe(base, task)
            if analysis.score >= 1.0:
                continue
            finding = self.finding_from_analysis(
                analysis,
                task=task,
                candidate_id=base.candidate_id,
                trace_id=trace.trace_id,
                verdict_id=f"{task.task_id}:{self.mechanism_cluster_id}",
                writable_artifact_ids=write_set,
            )
            if finding.status == "insufficient_evidence":
                continue
            issue = build_target_issue(
                finding,
                inventory,
                entropy=self._cell_entropy(task.task_id),
                coverage_need=self._coverage_need(task.task_id),
                pareto_relevance=self._pareto_relevance(base.candidate_id),
                embedding=self._embed_finding(finding, task),
                lineage=base.version,
                entropy_tier=self._entropy_tier(task.task_id),
            )
            if issue is not None:
                out.append(issue)
        return tuple(out)

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
        """Run origin and regression probes against the edited workspace."""
        planner = ValidationPlanner(
            origin_task=origin_task,
            regression_tasks=tuple(regression_tasks),
        )
        origin: list[ValidationResult] = []
        regression: list[ValidationResult] = []
        for probe in planner.build_probes():
            probe_id = self._next_probe_id(f"{workspace.attempt_id}-{probe.kind.value}")
            result = self.adapter.run_full_rollout(workspace, probe.task, probe_id)
            trace = self.adapter.capture_trace(result)
            analysis = self.analyzer_judge.analyze(probe.task, trace)
            outcome = ValidationResult(
                kind=probe.kind,
                task_id=probe.task.task_id,
                score=analysis.score,
                trace_id=trace.trace_id,
                passed=analysis.score >= 0.5,
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
                    judge_model_id=analysis.judge_model_id,
                    blame_confidence=min(1.0, analysis.blame_graph.total_blame()),
                    blame_stability=1.0,
                    artifact_versions=dict(candidate.artifact_hashes),
                ),
            )
        return entry

    # ------------------------------------------------------------------ #
    # One attempt
    # ------------------------------------------------------------------ #
    def run_attempt(self, tasks: Sequence[EvolutionTask]) -> GepaAttemptOutcome:
        """Execute one full GEPA attempt over the given task coreset."""
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
        _, analysis = self.observe(parent, task)

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
        self._persist_attempt(outcome, corrections)
        return outcome

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
