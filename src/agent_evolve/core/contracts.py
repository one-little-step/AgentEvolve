"""Agent-neutral contracts for RHO-Parallel-GEPA adapters.

This module intentionally contains no CUGA, Gaia, filesystem-layout, or model
provider imports. Adapters map their own execution model to these contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence


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
