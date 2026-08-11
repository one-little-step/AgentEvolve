"""Agent-neutral contracts for RHO-Parallel-GEPA adapters.

This module intentionally contains no CUGA, Gaia, filesystem-layout, or model
provider imports. Adapters map their own execution model to these contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


@dataclass(frozen=True, slots=True)
class ArtifactEdit:
    """One structured edit proposed by an editor model."""

    artifact_id: str
    operation: str
    payload: Mapping[str, object]


# ---------------------------------------------------------------------- #
# Persisted boundary contracts
#
# These models coexist with the prototype runtime dataclasses above while
# callers migrate. They reject invalid evidence before it can enter selection
# or persistence; the core remains adapter-neutral.
# ---------------------------------------------------------------------- #
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

    @model_validator(mode="after")
    def validate_provenance(self) -> "ScoreCell":
        if len(self.rollout_ids) != self.rollout_count:
            raise ValueError("rollout_ids length must equal rollout_count")
        if len(set(self.rollout_ids)) != len(self.rollout_ids):
            raise ValueError("rollout_ids must be unique")
        if self.rollout_count == 1 and self.stability is not None:
            raise ValueError("stability must be unknown for rollout_count == 1")
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
    edit_targets: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    risks: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_edit_targets(self) -> "EditPlan":
        unauthorized = set(self.edit_targets) - set(self.authorized_writes)
        if unauthorized:
            raise ValueError(
                "edit_targets must be contained in authorized_writes; "
                f"unauthorized: {sorted(unauthorized)}"
            )
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

    @model_validator(mode="after")
    def validate_inheritance(self) -> "ArtifactMergeDecision":
        if self.inheritance == "shared" and self.left_hash != self.right_hash:
            raise ValueError("shared inheritance requires equal left_hash and right_hash")
        if self.inheritance == "ancestor" and self.resulting_hash != self.ancestor_hash:
            raise ValueError("ancestor inheritance requires resulting_hash == ancestor_hash")
        return self


class MergeProvenance(BaseModel):
    """Immutable provenance for a candidate-level three-way merge."""

    model_config = ConfigDict(frozen=True)

    merge_id: str = Field(min_length=1)
    ancestor_candidate_id: str = Field(min_length=1)
    left_candidate_id: str = Field(min_length=1)
    right_candidate_id: str = Field(min_length=1)
    child_candidate_id: str | None = None
    artifact_decisions: tuple[ArtifactMergeDecision, ...] = Field(min_length=1)
    complementarity: float = Field(ge=0.0)
    eligibility_checks: Mapping[str, bool]

    @model_validator(mode="after")
    def validate_distinct_candidates(self) -> "MergeProvenance":
        parents = {
            self.ancestor_candidate_id,
            self.left_candidate_id,
            self.right_candidate_id,
        }
        if len(parents) != 3:
            raise ValueError("ancestor, left, and right candidate IDs must be distinct")
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
