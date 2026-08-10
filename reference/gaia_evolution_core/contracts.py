"""Agent-neutral contracts consumed by the RHO-GEPA population engine."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class EvolutionBundle:
    version: str
    modules: dict[str, str]


@dataclass(frozen=True, slots=True)
class NormalizedTrajectory:
    task_id: str
    input_text: str
    output_text: str
    status: str
    events: tuple[dict[str, object], ...] = ()
    source_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiagnosisRecord:
    failure_mode: str
    root_cause: str
    fix: str
    severity: str = "medium"
    phase: str = "cross_phase"
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class RolloutLimits:
    rerun_workers: int = 1
    rollout_workers: int = 1
    global_workers: int = 1


class CandidateEditor(Protocol):
    def append_section(self, filename: str, heading: str, content: str) -> None: ...
    def replace_section(self, filename: str, heading: str, content: str) -> None: ...
    def delete_section(self, filename: str, heading: str) -> None: ...
    def close(self) -> None: ...


class EvolutionLLM(Protocol):
    """Agent-neutral two-prompt LLM boundary for evolution operators."""

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class AgentEvolutionAdapter(Protocol):
    agent_name: str
    module_names: tuple[str, ...]

    def load_bundle(self, version: str) -> EvolutionBundle: ...
    def materialize_bundle(self, bundle: EvolutionBundle, target: Path) -> None: ...
    def run_rollouts(
        self,
        bundle: EvolutionBundle,
        tasks: Sequence[NormalizedTrajectory],
        *,
        rollout_count: int,
        limits: RolloutLimits,
        artifact_dir: Path,
    ) -> Mapping[str, Sequence[NormalizedTrajectory]]: ...
    def score_rollouts(
        self,
        bundle: EvolutionBundle,
        tasks: Sequence[NormalizedTrajectory],
        rollouts: Mapping[str, Sequence[NormalizedTrajectory]],
        *,
        reference_bundle: EvolutionBundle | None,
        reference_rollouts: Mapping[str, Sequence[NormalizedTrajectory]] | None,
        artifact_dir: Path,
    ) -> Mapping[str, float | None]: ...
    def diagnose(self, tasks: Sequence[NormalizedTrajectory], parent: EvolutionBundle) -> Sequence[DiagnosisRecord]: ...
    def phase_evidence(self, trajectory: NormalizedTrajectory, module: str) -> Sequence[Mapping[str, object]]: ...
    def open_editor(self, candidate_dir: Path, candidate_id: str) -> CandidateEditor: ...
