"""EvolutionAdapter bridge over the CUGA wrapper observation boundary."""
from __future__ import annotations

import json
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

# Artifact-id prefixes that map onto CUGA harness groups understood by
# ``materialize_harness``. ``instructions`` is a single scalar artifact.
_GROUP_PREFIXES = ("skills", "policies", "memory")
_SCALAR_ARTIFACTS = ("instructions",)

# Rich-trace event payload keys that hold content-addressed blob references.
# They are forwarded verbatim; the adapter never dereferences them, because
# blob contents may carry raw prompts, agent state, or expected answers.
_PAYLOAD_REF_KEYS = frozenset(
    {"state_before_ref", "state_after_ref", "messages_ref", "response_ref"}
)

_RESERVED_EVENT_KEYS = frozenset(
    {"event_id", "kind", "actor_id", "parent_event_id", "timestamp", "sequence", "payload"}
)


@dataclass
class CugaAdapter:
    """Adapter that maps CUGA harness artifacts to the evolution contracts.

    Candidate artifacts are held per version so that N sibling candidates
    (from a future seed generator or RHO proposal stage) can be materialized
    and run independently. CUGA checkpoint/replay remains unsupported.
    """

    wrapper: CugaWrapper
    adapter_name: str = "cuga"
    _workspaces: dict[str, dict[str, str]] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Seeding seam
    # ------------------------------------------------------------------ #

    def register_candidate(self, version: str, artifacts: Mapping[str, str]) -> None:
        """Register a candidate's artifact set under ``version``.

        This is the entry point for base-harness seeding and for any future
        seed generator or RHO proposal stage. Artifact ids must be mappable
        onto CUGA harness groups; unmappable ids are rejected here so that a
        bad seed fails at registration rather than silently degrading into a
        no-op rollout.
        """
        for artifact_id in artifacts:
            self._harness_slot(artifact_id)
        self._workspaces[version] = dict(artifacts)

    # ------------------------------------------------------------------ #
    # Artifact surface
    # ------------------------------------------------------------------ #

    def _artifacts_for(self, version: str) -> dict[str, str]:
        if version in self._workspaces:
            return self._workspaces[version]
        return dict(self.wrapper.get_artifacts())

    def artifact_inventory(self, version: str) -> Sequence[ArtifactDescriptor]:
        return tuple(
            ArtifactDescriptor(
                artifact_id=artifact_id,
                kind=self._harness_slot(artifact_id)[0],
                format="text/markdown",
                version_hash=f"sha256:{sha256(content.encode()).hexdigest()}",
                readable=True,
                writable=True,
                merge_strategy="adapter-defined",
            )
            for artifact_id, content in sorted(self._artifacts_for(version).items())
        )

    def read_artifacts(self, version: str, artifact_ids: Sequence[str]) -> Mapping[str, str]:
        artifacts = self._artifacts_for(version)
        return {artifact_id: artifacts[artifact_id] for artifact_id in artifact_ids}

    def materialize_candidate(self, parent_version: str, attempt_id: str) -> CandidateWorkspace:
        version = f"{parent_version}:{attempt_id}"
        # Deep-copy the parent's artifacts so sibling candidates never alias
        # the same mutable mapping.
        self._workspaces[version] = dict(self._artifacts_for(parent_version))
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

    # ------------------------------------------------------------------ #
    # Harness mapping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _harness_slot(artifact_id: str) -> tuple[str, str | None]:
        """Resolve an artifact id to a ``(harness_key, member_name)`` slot.

        Raises ``ValueError`` for ids CUGA cannot receive. Failing loudly is
        deliberate: silently dropping an artifact would let the loop report a
        successful edit that never reached the agent.
        """
        if artifact_id in _SCALAR_ARTIFACTS:
            return artifact_id, None
        prefix, _, remainder = artifact_id.partition("/")
        if prefix in _GROUP_PREFIXES and remainder:
            return prefix, remainder
        raise ValueError(
            f"artifact_id {artifact_id!r} does not map to a CUGA harness slot; "
            f"expected one of {_SCALAR_ARTIFACTS} or a "
            f"{'/'.join(_GROUP_PREFIXES)}/<name> prefix"
        )

    def _harness_config(self, version: str, task: EvolutionTask) -> dict[str, object]:
        """Build the harness config CUGA needs to load a candidate's artifacts."""
        harness: dict[str, object] = {"input": task.input_text}
        groups: dict[str, dict[str, str]] = {}
        for artifact_id, content in sorted(self._artifacts_for(version).items()):
            key, member = self._harness_slot(artifact_id)
            if member is None:
                harness[key] = content
            else:
                groups.setdefault(key, {})[member] = content
        harness.update(groups)
        return harness

    def run_full_rollout(
        self, workspace: CandidateWorkspace, task: EvolutionTask, rollout_id: str
    ) -> object:
        return {
            "rollout_id": rollout_id,
            "candidate_id": workspace.version,
            "trace": self.wrapper.run_task(
                task.task_id, self._harness_config(workspace.version, task)
            ),
        }

    # ------------------------------------------------------------------ #
    # Trace mapping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rich_events(trace_dir: Path) -> tuple[TraceEvent, ...] | None:
        """Load the persisted causal trace's events, preserving the DAG.

        Returns ``None`` when no rich trace is available so the caller can
        fall back to the thin runtime event list.
        """
        causal_file = trace_dir / "causal-trace.json"
        if not causal_file.is_file():
            return None
        raw = json.loads(causal_file.read_text(encoding="utf-8"))
        events = raw.get("events")
        if not isinstance(events, list):
            return None
        return tuple(
            TraceEvent(
                event_id=str(event["event_id"]),
                kind=str(event["kind"]),
                actor_id=(
                    str(event["actor_id"]) if event.get("actor_id") is not None else None
                ),
                parent_event_id=(
                    str(event["parent_event_id"])
                    if event.get("parent_event_id") is not None
                    else None
                ),
                payload=dict(event.get("payload") or {}),
            )
            for event in events
            if isinstance(event, Mapping)
        )

    def capture_trace(self, rollout_result: object) -> ExecutionTrace:
        result = cast(Mapping[str, Any], rollout_result)
        raw = cast(Mapping[str, Any], result["trace"])

        events: tuple[TraceEvent, ...] | None = None
        trace_path = raw.get("causal_trace_path")
        if trace_path:
            events = self._rich_events(Path(str(trace_path)))

        if events is None:
            events = tuple(
                TraceEvent(
                    event_id=str(event["event_id"]),
                    kind=str(event["kind"]),
                    actor_id=(
                        str(event["actor_id"]) if event.get("actor_id") is not None else None
                    ),
                    parent_event_id=(
                        str(event["parent_event_id"])
                        if event.get("parent_event_id") is not None
                        else None
                    ),
                    payload={
                        key: value
                        for key, value in event.items()
                        if key not in _RESERVED_EVENT_KEYS
                    },
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
