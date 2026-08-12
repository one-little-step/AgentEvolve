"""EvolutionAdapter bridge over the CUGA wrapper observation boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from agent_evolve.core.contracts import (
    ArtifactDescriptor,
    ArtifactEdit,
    CandidateWorkspace,
    CheckpointDescriptor,
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)
from agent_evolve.cuga_wrapper import CugaWrapper


@dataclass
class CugaAdapter:
    """Minimal adapter that delegates rollout collection to ``CugaWrapper``.

    It provides an in-process artifact snapshot only for wrapper-exposed text
    artifacts. CUGA checkpoint/replay remains explicitly unsupported.
    """

    wrapper: CugaWrapper
    adapter_name: str = "cuga"
    _workspaces: dict[str, dict[str, str]] = field(default_factory=dict)

    def artifact_inventory(self, version: str) -> Sequence[ArtifactDescriptor]:
        return tuple(
            ArtifactDescriptor(
                artifact_id=artifact_id,
                kind="cuga-wrapper-artifact",
                format="text/plain",
                version_hash=f"sha256:{sha256(content.encode()).hexdigest()}",
                readable=True,
                writable=True,
                merge_strategy="adapter-defined",
            )
            for artifact_id, content in self.read_artifacts(
                version, tuple(self.wrapper.get_artifacts())
            ).items()
        )

    def read_artifacts(self, version: str, artifact_ids: Sequence[str]) -> Mapping[str, str]:
        artifacts = self._workspaces.get(version, self.wrapper.get_artifacts())
        return {artifact_id: artifacts[artifact_id] for artifact_id in artifact_ids}

    def materialize_candidate(self, parent_version: str, attempt_id: str) -> CandidateWorkspace:
        version = f"{parent_version}:{attempt_id}"
        self._workspaces[version] = dict(self._workspaces.get(parent_version, self.wrapper.get_artifacts()))
        return CandidateWorkspace(attempt_id, version, Path("."), parent_version)

    def apply_structured_edits(
        self, workspace: CandidateWorkspace, edits: Sequence[ArtifactEdit]
    ) -> Mapping[str, str]:
        artifacts = self._workspaces[workspace.version]
        for edit in edits:
            if edit.operation != "replace":
                raise ValueError(f"unsupported CUGA wrapper edit operation: {edit.operation}")
            if edit.artifact_id not in artifacts:
                raise KeyError(edit.artifact_id)
            content = edit.payload.get("content")
            if not isinstance(content, str):
                raise ValueError("replace edits require a string payload.content")
            artifacts[edit.artifact_id] = content
        return dict(artifacts)

    def run_full_rollout(
        self, workspace: CandidateWorkspace, task: EvolutionTask, rollout_id: str
    ) -> object:
        return {
            "rollout_id": rollout_id,
            "candidate_id": workspace.version,
            "trace": self.wrapper.run_task(task.task_id, {"input": task.input_text}),
        }

    def capture_trace(self, rollout_result: object) -> ExecutionTrace:
        result = cast(Mapping[str, Any], rollout_result)
        raw = cast(Mapping[str, Any], result["trace"])
        events = tuple(
            TraceEvent(
                event_id=str(event["event_id"]),
                kind=str(event["kind"]),
                actor_id=None,
                parent_event_id=None,
                payload={key: value for key, value in event.items() if key not in {"event_id", "kind"}},
            )
            for event in cast(Sequence[Mapping[str, object]], raw["events"])
        )
        return ExecutionTrace(
            trace_id=str(result["rollout_id"]),
            candidate_id=str(result["candidate_id"]),
            task_id=str(raw["task_id"]),
            events=events,
            final_output=str(raw["final_output"]),
            status=str(raw["status"]),
        )

    def supports_counterfactual_replay(self) -> bool:
        return False

    def discover_checkpoints(self, trace: ExecutionTrace) -> Sequence[CheckpointDescriptor]:
        return ()

    def replay_from_checkpoint(
        self,
        checkpoint: CheckpointDescriptor,
        workspace: CandidateWorkspace,
        task: EvolutionTask,
        rollout_id: str,
    ) -> object:
        raise RuntimeError("CUGA checkpoint replay is not verified")
