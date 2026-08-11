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
from typing import Iterable, Mapping, Sequence

from agent_evolve.core.analyzer import AnalyzerJudge, FakeAnalyzerJudge
from agent_evolve.core.blame import CausalAnalysis, empty_analysis
from agent_evolve.core.clustering import (
    ClusterRegistry,
    LexicalEmbedder,
    MechanismClusterer,
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
)
from agent_evolve.core.entropy import (
    EntropyTracker,
    HierarchicalDPPSelector,
    Issue,
)
from agent_evolve.core.fake_editor import FakeEditor
from agent_evolve.core.memory import (
    AttemptStatus,
    EditAttempt,
    EditMemory,
    make_attempt_id,
)
from agent_evolve.core.pool import (
    PersistentPool,
    PoolEntry,
    ScoreProvenance,
)
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
