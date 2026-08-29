"""Agent-neutral contracts for RHO-Parallel-GEPA adapters.

This module intentionally contains no CUGA, Gaia, filesystem-layout, or model
provider imports. Adapters map their own execution model to these contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Literal, Mapping, Protocol, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    """An adapter-declared editable unit or atomic editable group."""

    artifact_id: str
    kind: str
    format: str
    version_hash: str
    readable: bool
    writable: bool
    merge_strategy: str
    bindings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvolutionCandidate:
    """A materialized harness version tracked by the persistent pool."""

    candidate_id: str
    version: str
    artifact_hashes: Mapping[str, str]
    parent_ids: tuple[str, ...] = ()
    ancestor_ids: tuple[str, ...] = ()
    attempt_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvolutionTask:
    """Agent-neutral evaluation input with redacted historical context."""

    task_id: str
    input_text: str
    expected_contract: Mapping[str, object] = field(default_factory=dict)
    source_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """A sanitized state, tool, subagent, or model event from an adapter."""

    event_id: str
    kind: str
    actor_id: str | None
    parent_event_id: str | None
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    """Exact adapter-provided trace evidence for one candidate/task rollout."""

    trace_id: str
    candidate_id: str
    task_id: str
    events: tuple[TraceEvent, ...]
    final_output: str
    status: str
    checkpoint_ids: tuple[str, ...] = ()
    #: On-disk location of the persisted trace (the tape), e.g. the directory
    #: holding ``causal-trace.json`` + ``payloads/``. ``""`` means *no location
    #: is known* -- explicit absence, never a fabricated path (W1).
    trace_dir: str = ""


class ArtifactEdit(BaseModel):
    """One structured edit proposed by an editor model."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    payload: Mapping[str, object]


# ---------------------------------------------------------------------- #
# Persisted boundary contracts
#
# These models coexist with the prototype runtime dataclasses above while
# callers migrate. They reject invalid evidence before it can enter selection
# or persistence; the core remains adapter-neutral.
# ---------------------------------------------------------------------- #
_CONTENT_HASH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*:[0-9A-Fa-f]+$")


def _validate_id_values(field_name: str, values: tuple[str, ...]) -> None:
    if any(not value for value in values):
        raise ValueError(f"{field_name} must not contain blank IDs")


def _validate_mapping_keys(field_name: str, values: Mapping[str, object]) -> None:
    if any(not key for key in values):
        raise ValueError(f"{field_name} must not contain blank keys")


def _validate_content_hash(field_name: str, value: str) -> None:
    if not _CONTENT_HASH.fullmatch(value):
        raise ValueError(f"{field_name} must be an algorithm:hexdigest content hash")


def _validate_hash_mapping(field_name: str, values: Mapping[str, str]) -> None:
    if any(not key for key in values) or any(
        not _CONTENT_HASH.fullmatch(value) for value in values.values()
    ):
        raise ValueError(
            f"{field_name} hash mappings must contain non-blank keys and "
            "algorithm:hexdigest content hashes"
        )


class ScoreCell(BaseModel):
    """Validated atomic evidence cell for persisted/comparable score data."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    mechanism_cluster_id: str = Field(min_length=1)
    mechanism_ids: tuple[str, ...] = ()
    score: float = Field(ge=0.0, le=1.0)
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    stability: float | None = Field(default=None, ge=0.0, le=1.0)
    rollout_count: int = Field(ge=1)
    rollout_ids: tuple[str, ...]
    verdict_refs: tuple[str, ...] = Field(min_length=1)
    artifact_versions: Mapping[str, str] = Field(min_length=1)
    evaluator_id: str = Field(min_length=1)
    coverage: Literal["evaluated", "unavailable", "excluded"]
    coverage_reason: str | None = None

    @model_validator(mode="after")
    def validate_provenance(self) -> "ScoreCell":
        _validate_id_values("rollout_ids", self.rollout_ids)
        _validate_id_values("verdict_refs", self.verdict_refs)
        _validate_hash_mapping("artifact_versions", self.artifact_versions)
        if len(self.rollout_ids) != self.rollout_count:
            raise ValueError("rollout_ids length must equal rollout_count")
        if len(set(self.rollout_ids)) != len(self.rollout_ids):
            raise ValueError("rollout_ids must be unique")
        if self.rollout_count == 1 and self.stability is not None:
            raise ValueError("stability must be unknown for rollout_count == 1")
        if self.rollout_count != 1 and self.stability is None:
            raise ValueError("stability is required unless rollout_count == 1")
        if self.coverage == "evaluated" and self.coverage_reason is not None:
            raise ValueError("coverage_reason is only permitted for unavailable or excluded coverage")
        if self.coverage != "evaluated" and (
            not self.coverage_reason or not self.coverage_reason.strip()
        ):
            raise ValueError("coverage_reason is required for unavailable or excluded coverage")
        return self


TerminalAttemptStatus = Literal[
    "accepted", "rejected", "no_op", "malformed", "exhausted", "unavailable"
]


class AttemptRecord(BaseModel):
    """Validated terminal result of one immutable edit attempt."""

    model_config = ConfigDict(frozen=True)

    attempt_id: str = Field(min_length=1)
    snapshot_version: str = Field(min_length=1)
    parent_candidate_id: str = Field(min_length=1)
    result_candidate_id: str | None = None
    status: TerminalAttemptStatus
    issue_fingerprint: str = Field(min_length=1)
    task_refs: tuple[str, ...]
    mechanism_cluster_refs: tuple[str, ...]
    read_set: tuple[str, ...]
    write_set: tuple[str, ...]
    workspace_sealed: bool
    hashes_before: Mapping[str, str]
    hashes_after: Mapping[str, str]
    analysis_refs: tuple[str, ...]
    verdict_refs: tuple[str, ...]
    memory_refs: tuple[str, ...]
    validation_result_ref: str | None = None
    rationale_summary: str
    risk_summary: str
    budget_usage: Mapping[str, object]
    retry_state: Mapping[str, object]
    timestamps: Mapping[str, object]

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "AttemptRecord":
        for field_name, values in (
            ("task_refs", self.task_refs),
            ("mechanism_cluster_refs", self.mechanism_cluster_refs),
            ("read_set", self.read_set),
            ("write_set", self.write_set),
            ("analysis_refs", self.analysis_refs),
            ("verdict_refs", self.verdict_refs),
            ("memory_refs", self.memory_refs),
        ):
            _validate_id_values(field_name, values)
        _validate_hash_mapping("hashes_before", self.hashes_before)
        _validate_hash_mapping("hashes_after", self.hashes_after)
        if self.workspace_sealed and not self.hashes_after:
            raise ValueError("hashes_after is required when workspace_sealed")
        if not self.workspace_sealed and self.hashes_after:
            raise ValueError("hashes_after is only permitted when workspace_sealed")
        if self.status != "unavailable":
            for field_name, references in (
                ("analysis_refs", self.analysis_refs),
                ("verdict_refs", self.verdict_refs),
                ("memory_refs", self.memory_refs),
            ):
                if not references:
                    raise ValueError(f"{field_name} is required unless status is unavailable")
        if self.status == "accepted":
            if not self.result_candidate_id:
                raise ValueError("result_candidate_id is required for accepted attempts")
            if not self.validation_result_ref:
                raise ValueError("validation_result_ref is required for accepted attempts")
        elif self.result_candidate_id is not None:
            raise ValueError("result_candidate_id is only permitted for accepted attempts")
        if self.status == "rejected" and not self.validation_result_ref:
            raise ValueError("validation_result_ref is required for rejected attempts")
        return self


class EditPlan(BaseModel):
    """Editor boundary record separating requested reads from write authority."""

    model_config = ConfigDict(frozen=True)

    attempt_id: str = Field(min_length=1)
    issue_fingerprint: str = Field(min_length=1)
    read_requests: tuple[str, ...]
    authorized_writes: tuple[str, ...] = Field(min_length=1)
    edits: tuple[ArtifactEdit, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    risks: tuple[str, ...] = ()
    expected_effect: "ExpectedEffect"

    @model_validator(mode="after")
    def validate_edits(self) -> "EditPlan":
        _validate_id_values("read_requests", self.read_requests)
        _validate_id_values("authorized_writes", self.authorized_writes)
        unauthorized = {
            edit.artifact_id for edit in self.edits
        } - set(self.authorized_writes)
        if unauthorized:
            raise ValueError(
                "edits must target artifacts in authorized_writes; "
                f"unauthorized: {sorted(unauthorized)}"
            )
        return self


class ExpectedEffect(BaseModel):
    """Expected mechanism-level outcome of a proposed edit plan."""

    model_config = ConfigDict(frozen=True)

    mechanism_cluster_refs: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_mechanism_cluster_refs(self) -> "ExpectedEffect":
        _validate_id_values("mechanism_cluster_refs", self.mechanism_cluster_refs)
        return self


ValidationCaseOutcome = Literal["passed", "failed", "unavailable"]


class ValidationCase(BaseModel):
    """Validation evidence for one case used in an edit decision."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    outcome: ValidationCaseOutcome


ProtectedFloorOutcome = Literal["satisfied", "violated", "unavailable"]
ValidationDecision = Literal["accept", "reject"]


class ValidationResult(BaseModel):
    """Validated aggregate evidence for accepting or rejecting an edit."""

    model_config = ConfigDict(frozen=True)

    origin_cases: tuple[ValidationCase, ...] = Field(min_length=1)
    worked_cases: tuple[ValidationCase, ...]
    regression_cases: tuple[ValidationCase, ...]
    generalization_cases: tuple[ValidationCase, ...] = ()
    primary_gain: float
    weighted_net_gain: float
    protected_floor_outcome: ProtectedFloorOutcome
    decision: ValidationDecision
    decision_reason: str = Field(min_length=1)
    unavailable_cases: tuple[ValidationCase, ...]

    @model_validator(mode="after")
    def validate_protected_floor(self) -> "ValidationResult":
        if any(case.outcome != "unavailable" for case in self.unavailable_cases):
            raise ValueError("unavailable_cases must contain only unavailable cases")
        if self.protected_floor_outcome == "violated" and self.decision == "accept":
            raise ValueError("protected_floor_outcome violated requires decision == reject")
        return self


ArtifactInheritance = Literal["ancestor", "left", "right", "shared", "refined"]


class ArtifactMergeDecision(BaseModel):
    """Per-artifact immutable three-way merge provenance."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(min_length=1)
    ancestor_hash: str = Field(min_length=1)
    left_hash: str = Field(min_length=1)
    right_hash: str = Field(min_length=1)
    resulting_hash: str = Field(min_length=1)
    inheritance: ArtifactInheritance
    evidence_score_left: float = Field(ge=0.0)
    evidence_score_right: float = Field(ge=0.0)
    decision_reason: str = Field(min_length=1)
    refinement_request_ref: str | None = None
    operation_emitted: bool

    @model_validator(mode="after")
    def validate_inheritance(self) -> "ArtifactMergeDecision":
        for field_name, value in (
            ("ancestor_hash", self.ancestor_hash),
            ("left_hash", self.left_hash),
            ("right_hash", self.right_hash),
            ("resulting_hash", self.resulting_hash),
        ):
            _validate_content_hash(field_name, value)
        if self.inheritance == "shared" and self.left_hash != self.right_hash:
            raise ValueError("shared inheritance requires equal left_hash and right_hash")
        if self.inheritance == "ancestor" and self.resulting_hash != self.ancestor_hash:
            raise ValueError("ancestor inheritance requires resulting_hash == ancestor_hash")
        if self.inheritance == "refined" and not self.refinement_request_ref:
            raise ValueError("refinement_request_ref is required for refined inheritance")
        if self.inheritance != "refined" and self.refinement_request_ref is not None:
            raise ValueError("refinement_request_ref is only permitted for refined inheritance")
        if self.resulting_hash == self.ancestor_hash and self.operation_emitted:
            raise ValueError("operation_emitted must be false when resulting_hash equals ancestor_hash")
        return self


class MergeProvenance(BaseModel):
    """Immutable provenance for a candidate-level three-way merge."""

    model_config = ConfigDict(frozen=True)

    merge_id: str = Field(min_length=1)
    ancestor_candidate_id: str = Field(min_length=1)
    left_candidate_id: str = Field(min_length=1)
    right_candidate_id: str = Field(min_length=1)
    child_admitted: bool
    child_candidate_id: str | None = None
    artifact_decisions: tuple[ArtifactMergeDecision, ...] = Field(min_length=1)
    complementarity: float = Field(ge=0.0)
    eligibility_checks: Mapping[str, bool]

    @model_validator(mode="after")
    def validate_distinct_candidates(self) -> "MergeProvenance":
        _validate_id_values(
            "candidate_ids",
            tuple(
                candidate_id
                for candidate_id in (
                    self.ancestor_candidate_id,
                    self.left_candidate_id,
                    self.right_candidate_id,
                    self.child_candidate_id,
                )
                if candidate_id is not None
            ),
        )
        parents = {
            self.ancestor_candidate_id,
            self.left_candidate_id,
            self.right_candidate_id,
        }
        if len(parents) != 3:
            raise ValueError("ancestor, left, and right candidate IDs must be distinct")
        if self.child_admitted and not self.child_candidate_id:
            raise ValueError("child_candidate_id is required when child_admitted")
        if not self.child_admitted and self.child_candidate_id is not None:
            raise ValueError("child_candidate_id is only permitted when child_admitted")
        return self


class RedactionReport(BaseModel):
    """Summary of sanitization applied before persisting a memory record."""

    model_config = ConfigDict(frozen=True)

    rule_hits: tuple[str, ...]
    truncations: int = Field(ge=0)


class MemoryRecord(BaseModel):
    """Append-only sanitized, reference-based memory from an attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_record_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    artifact_ids: tuple[str, ...]
    issue_fingerprint: str = Field(min_length=1)
    outcome: TerminalAttemptStatus
    summary: str = Field(min_length=1)
    evidence_refs: tuple[str, ...]
    redaction_report: RedactionReport

    @model_validator(mode="after")
    def validate_references(self) -> "MemoryRecord":
        _validate_id_values("artifact_ids", self.artifact_ids)
        _validate_id_values("evidence_refs", self.evidence_refs)
        return self


@dataclass(frozen=True, slots=True)
class CandidateWorkspace:
    """Adapter-owned isolated workspace for one attempted candidate change."""

    attempt_id: str
    version: str
    path: Path
    parent_version: str


@dataclass(frozen=True, slots=True)
class CheckpointDescriptor:
    """An adapter-declared replay-safe execution checkpoint."""

    checkpoint_id: str
    trace_id: str
    event_id: str
    state_hash: str
    replay_scope: tuple[str, ...]


class EvolutionAdapter(Protocol):
    """Capability boundary between generic evolution and a concrete agent."""

    adapter_name: str

    def artifact_inventory(self, version: str) -> Sequence[ArtifactDescriptor]: ...

    def read_artifacts(
        self, version: str, artifact_ids: Sequence[str]
    ) -> Mapping[str, str]: ...

    def materialize_candidate(
        self, parent_version: str, attempt_id: str
    ) -> CandidateWorkspace: ...

    def apply_structured_edits(
        self, workspace: CandidateWorkspace, edits: Sequence[ArtifactEdit]
    ) -> Mapping[str, str]: ...

    def run_full_rollout(
        self, workspace: CandidateWorkspace, task: EvolutionTask, rollout_id: str
    ) -> object: ...

    def capture_trace(self, rollout_result: object) -> ExecutionTrace: ...

    def supports_counterfactual_replay(self) -> bool: ...

    def discover_checkpoints(
        self, trace: ExecutionTrace
    ) -> Sequence[CheckpointDescriptor]: ...

    def replay_from_checkpoint(
        self,
        checkpoint: CheckpointDescriptor,
        workspace: CandidateWorkspace,
        task: EvolutionTask,
        rollout_id: str,
    ) -> object: ...
