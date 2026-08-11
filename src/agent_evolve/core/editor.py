"""Editor protocol, structured edit attempts, and focused validation.

Per docs/architecture/target-rho-parallel-gepa.md:

    An editor may modify any adapter-declared artifact in its approved write
    set. It must request/read current content before editing and returns
    rationale, reads, writes, edits, risks, and expected effects. Every
    attempt records sanitized reasoning, diff, evidence references, history
    IDs, validation results, and status.

    Focused validation covers:
        origin mechanism cases
        worked-set cases for written artifacts
        regression probes for written artifacts

    Small regressions are allowed only when weighted net gain is positive and
    no protected critical floor is violated. Retry state is scoped to issue,
    artifact group, and lineage, with a default maximum of three attempts.

    Generalization probes are deferred until a mechanism edit cluster
    completes. They are budgeted, may replay only when the adapter supports
    it, and otherwise perform full rollouts. Probe failures become future
    regression evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, Sequence

from agent_evolve.core.blame import CausalAnalysis
from agent_evolve.core.contracts import (
    ArtifactEdit,
    ArtifactDescriptor,
    CandidateWorkspace,
    EvolutionAdapter,
    EvolutionTask,
    ExecutionTrace,
)
from agent_evolve.core.memory import (
    AttemptStatus,
    EditAttempt,
    EditMemory,
    artifact_group_of,
    sanitize_payload,
)


# ---------------------------------------------------------------------- #
# Editor request / response
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class EditorRequest:
    """Inputs handed to the editor for one attempt."""

    base_workspace: CandidateWorkspace
    task: EvolutionTask
    analysis: CausalAnalysis
    issue_id: str
    # Artifacts the editor is allowed to modify (adapter-declared write set).
    write_set: tuple[str, ...]
    # Prior attempts the editor should consult (from edit-memory RAG).
    history_refs: tuple[str, ...] = ()
    # Current artifact contents, read by the adapter before the editor runs.
    current_artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.write_set:
            raise ValueError("write_set is required (cannot be empty)")
        if not self.issue_id:
            raise ValueError("issue_id is required")
        # current_artifacts must be a subset of write_set.
        extra = set(self.current_artifacts.keys()) - set(self.write_set)
        if extra:
            raise ValueError(
                f"current_artifacts references ids outside write_set: {sorted(extra)}"
            )


@dataclass(frozen=True, slots=True)
class EditorResponse:
    """What the editor returned, before validation.

    The editor must NOT directly mutate the workspace; it returns a structured
    set of edits and the orchestrator applies them via the adapter. This keeps
    the editor's reasoning auditable and replay-safe.
    """

    rationale: str
    edits: tuple[ArtifactEdit, ...]
    reads: Mapping[str, str]
    writes: Mapping[str, str]
    risks: Mapping[str, str]
    expected_effects: Mapping[str, str]
    editor_model_id: str = ""

    def __post_init__(self) -> None:
        if not self.edits:
            raise ValueError("edits is required (cannot be empty)")
        # edits must target artifacts within the request's write_set. We can't
        # see the request here, so the caller validates that; but we can ensure
        # every edit has a non-empty artifact_id and operation.
        for e in self.edits:
            if not e.artifact_id:
                raise ValueError("each edit must have an artifact_id")
            if not e.operation:
                raise ValueError("each edit must have an operation")
        # Sanitize writes/risks/expected_effects: refuse denied keys.
        sanitize_payload(self.writes)
        sanitize_payload(self.risks)
        sanitize_payload(self.expected_effects)


# ---------------------------------------------------------------------- #
# Validation
# ---------------------------------------------------------------------- #
class ValidationKind(str, Enum):
    ORIGIN = "origin"
    WORKED = "worked"
    REGRESSION = "regression"
    GENERALIZATION = "generalization"


@dataclass(frozen=True, slots=True)
class ValidationProbe:
    """One validation rollout to run."""

    kind: ValidationKind
    task: EvolutionTask
    # Why this probe was selected; recorded in the attempt.
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of one validation probe."""

    kind: ValidationKind
    task_id: str
    score: float
    trace_id: str
    passed: bool
    mechanism_cluster_id: str = "*"

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError("score must be in [0, 1]")
        if not self.mechanism_cluster_id:
            raise ValueError("mechanism_cluster_id is required")


@dataclass(frozen=True, slots=True)
class FocusedValidationReport:
    """Aggregated result of all probes for one attempt."""

    origin: tuple[ValidationResult, ...]
    worked: tuple[ValidationResult, ...]
    regression: tuple[ValidationResult, ...]
    generalization: tuple[ValidationResult, ...] = ()

    @property
    def all_results(self) -> tuple[ValidationResult, ...]:
        return self.origin + self.worked + self.regression + self.generalization

    @property
    def origin_passed(self) -> bool:
        return all(r.passed for r in self.origin) if self.origin else True

    @property
    def worked_passed(self) -> bool:
        return all(r.passed for r in self.worked) if self.worked else True

    @property
    def regression_violated(self) -> bool:
        """True if any regression probe regressed below its protected floor."""
        return any(not r.passed for r in self.regression)

    def weighted_net_gain(self, weights: Mapping[ValidationKind, float] | None = None) -> float:
        """Weighted sum of probe scores. Default weights favor origin + worked."""
        w = weights or {
            ValidationKind.ORIGIN: 1.0,
            ValidationKind.WORKED: 0.5,
            ValidationKind.REGRESSION: -1.0,
            ValidationKind.GENERALIZATION: 0.25,
        }
        total = 0.0
        for r in self.all_results:
            total += w.get(r.kind, 0.0) * r.score
        return total


# ---------------------------------------------------------------------- #
# Protected floors
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ProtectedFloor:
    """A minimum score floor for a specific (task, mechanism) cell.

    Used to forbid accepting edits that regress critical floors even when
    weighted net gain is positive.
    """

    task_id: str
    mechanism_cluster_id: str
    min_score: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_score <= 1.0):
            raise ValueError("min_score must be in [0, 1]")


def floors_violated(
    results: Sequence[ValidationResult],
    floors: Sequence[ProtectedFloor],
) -> tuple[ProtectedFloor, ...]:
    """Return the floors violated by these results."""
    out: list[ProtectedFloor] = []
    for f in floors:
        relevant = [
            result
            for result in results
            if result.task_id == f.task_id
            and (
                result.mechanism_cluster_id == f.mechanism_cluster_id
                # Existing validation callers did not carry a cluster. Their
                # wildcard remains conservative rather than bypassing floors.
                or result.mechanism_cluster_id == "*"
            )
        ]
        if relevant and max(r.score for r in relevant) < f.min_score:
            out.append(f)
    return tuple(out)


# ---------------------------------------------------------------------- #
# Acceptance
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class AcceptanceDecision:
    """Final accept/reject decision for one attempt."""

    accepted: bool
    status: AttemptStatus
    reason: str
    weighted_net_gain: float
    protected_floors_violated: tuple[ProtectedFloor, ...] = ()


def decide_acceptance(
    report: FocusedValidationReport,
    protected_floors: Sequence[ProtectedFloor] = (),
    net_gain_threshold: float = 0.0,
    weights: Mapping[ValidationKind, float] | None = None,
) -> AcceptanceDecision:
    """Apply the architecture's acceptance rule.

    Accept iff:
    * weighted net gain > threshold, AND
    * no protected floor is violated, AND
    * origin and worked probes all passed (no critical regression).

    Otherwise reject. If the rejection is due to a regression, the status is
    REGRESSION; if due to net gain, REJECTED; if due to a protected floor,
    also REGRESSION (the architecture treats protected-floor violations as
    regressions for retry-budget accounting).
    """
    gain = report.weighted_net_gain(weights)
    floors = floors_violated(report.all_results, protected_floors)

    if report.origin_passed and report.worked_passed and not floors and gain > net_gain_threshold:
        return AcceptanceDecision(
            accepted=True,
            status=AttemptStatus.ACCEPTED,
            reason="passed origin + worked, no floor violated, positive net gain",
            weighted_net_gain=gain,
        )

    if floors or report.regression_violated:
        return AcceptanceDecision(
            accepted=False,
            status=AttemptStatus.REGRESSION,
            reason="regression detected or protected floor violated",
            weighted_net_gain=gain,
            protected_floors_violated=floors,
        )

    return AcceptanceDecision(
        accepted=False,
        status=AttemptStatus.REJECTED,
        reason="insufficient net gain or origin/worked failure",
        weighted_net_gain=gain,
        protected_floors_violated=floors,
    )


# ---------------------------------------------------------------------- #
# Editor protocol
# ---------------------------------------------------------------------- #
class Editor(Protocol):
    """Adapter-agnostic editor: maps a request to a structured response."""

    editor_model_id: str

    def propose_edit(self, request: EditorRequest) -> EditorResponse: ...


# ---------------------------------------------------------------------- #
# Validation planner
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class ValidationPlanner:
    """Builds the probe list for one attempt.

    Generalization probes are deferred until the orchestrator signals that a
    mechanism edit cluster has completed (``emit_generalization_probes=True``).
    """

    origin_task: EvolutionTask
    worked_tasks: tuple[EvolutionTask, ...] = ()
    regression_tasks: tuple[EvolutionTask, ...] = ()
    generalization_tasks: tuple[EvolutionTask, ...] = ()
    emit_generalization_probes: bool = False

    def build_probes(self) -> tuple[ValidationProbe, ...]:
        out: list[ValidationProbe] = [
            ValidationProbe(
                kind=ValidationKind.ORIGIN,
                task=self.origin_task,
                reason="origin mechanism case",
            )
        ]
        for t in self.worked_tasks:
            out.append(
                ValidationProbe(kind=ValidationKind.WORKED, task=t, reason="worked-set case")
            )
        for t in self.regression_tasks:
            out.append(
                ValidationProbe(
                    kind=ValidationKind.REGRESSION, task=t, reason="regression probe"
                )
            )
        if self.emit_generalization_probes:
            for t in self.generalization_tasks:
                out.append(
                    ValidationProbe(
                        kind=ValidationKind.GENERALIZATION,
                        task=t,
                        reason="deferred generalization probe",
                    )
                )
        return tuple(out)


# ---------------------------------------------------------------------- #
# Attempt builder
# ---------------------------------------------------------------------- #
def build_attempt(
    attempt_id: str,
    candidate_id: str,
    issue_id: str,
    response: EditorResponse,
    evidence_refs: Sequence[str],
    history_refs: Sequence[str],
    report: FocusedValidationReport,
    decision: AcceptanceDecision,
) -> EditAttempt:
    """Assemble an :class:`EditAttempt` from the editor's response + validation."""
    summary: dict[str, str] = {}
    for r in report.all_results:
        summary[f"{r.kind.value}:{r.task_id}"] = "pass" if r.passed else "fail"
    summary["decision"] = decision.status.value
    return EditAttempt(
        attempt_id=attempt_id,
        candidate_id=candidate_id,
        issue_id=issue_id,
        artifact_ids=tuple(e.artifact_id for e in response.edits),
        operation=response.edits[0].operation,
        sanitized_reasoning=response.rationale,
        sanitized_diff={"writes": dict(response.writes), "risks": dict(response.risks)},
        evidence_refs=tuple(evidence_refs),
        history_refs=tuple(history_refs),
        validation_summary=summary,
        status=decision.status,
    )


def lineage_of(workspace: CandidateWorkspace, parent_versions: Sequence[str] = ()) -> str:
    """Stable lineage key for retry-budget scoping.

    Defaults to the workspace's parent_version; if multiple parents (e.g.,
    merge), use their sorted concatenation.
    """
    if parent_versions:
        return "|".join(sorted(parent_versions))
    return workspace.parent_version or "root"


def record_attempt(
    memory: EditMemory,
    attempt: EditAttempt,
    workspace: CandidateWorkspace,
    parent_versions: Sequence[str] = (),
) -> None:
    """Record an attempt in edit memory with proper retry-budget scoping."""
    group = artifact_group_of(attempt.artifact_ids)
    lineage = lineage_of(workspace, parent_versions)
    memory.record(attempt, artifact_group=group, lineage=lineage)
