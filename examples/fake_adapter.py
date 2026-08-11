"""Self-contained in-memory FakeAdapter demonstrating the EvolutionAdapter contract.

This adapter is **agent-neutral** and lives outside ``src/agent_evolve/core`` to
keep the core free of any concrete runtime. It is intended for:

* Smoke-testing the contract surface without a real agent SDK.
* Showing how an adapter maps its own execution model to
  :class:`agent_evolve.core.contracts.EvolutionAdapter`.
* Providing a fixture other tests can import.

It does NOT implement counterfactual replay (``supports_counterfactual_replay``
returns ``False``); the core must therefore perform a full rollout when it needs
fresh evidence. This mirrors the explicit "replay is optional" contract.

Design notes
------------
* Artifacts are simple ``text/plain`` strings stored in a version-keyed dict.
* A "rollout" is a deterministic function of (artifact contents, task input).
  The fake never calls an LLM — it scores output by substring match against
  ``task.expected_contract["expected_substring"]`` if present, else by length
  heuristic. This keeps tests deterministic and offline.
* Materialized candidates are isolated copies of the parent's artifact dict so
  that edits in one workspace never leak into siblings — matching the
  snapshot/lease semantics the target architecture requires of real adapters.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from agent_evolve.core.contracts import (
    ArtifactDescriptor,
    ArtifactEdit,
    CandidateWorkspace,
    CheckpointDescriptor,
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)

# Adapter-declared merge strategies are free-form strings; the core must not
# hardcode any particular value. We pick a single one here for clarity.
_MERGE_STRATEGY = "replace-overwrites"

# Default base-harness artifacts. An artifact_id is opaque to the core; the
# adapter decides what each one means.
_BASE_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    # (artifact_id, kind, initial content)
    ("skills/retrieval", "skill", "retrieve(query): return top_k docs by bm25"),
    ("policies/execution", "policy", "execute(tool, args): call tool, return output"),
    ("prompts/system", "prompt", "You are a helpful assistant."),
)


def _hash_content(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class _RolloutResult:
    """Internal adapter object returned by run_full_rollout.

    The contract types this as ``object`` precisely because each adapter owns
    its own rollout-result shape; only capture_trace knows how to read it.
    """

    workspace: CandidateWorkspace
    task: EvolutionTask
    events: tuple[TraceEvent, ...]
    final_output: str
    status: str


class FakeAdapter:
    """In-memory, offline, no-replay adapter for the AgentEvolve contract."""

    adapter_name: str = "fake"

    def __init__(self, base_artifacts: Sequence[tuple[str, str, str]] = _BASE_ARTIFACTS) -> None:
        # version -> {artifact_id: content}
        self._versions: dict[str, dict[str, str]] = {}
        # version -> tuple[ArtifactDescriptor, ...]
        self._inventory: dict[str, tuple[ArtifactDescriptor, ...]] = {}
        # attempt_id -> workspace artifact dict (mutable staging area)
        self._workspaces: dict[str, dict[str, str]] = {}
        # version -> (parent_version, attempt_id)
        self._lineage: dict[str, tuple[str, str]] = {}

        # Seed the base version.
        base_version = "base-v0"
        base_dict: dict[str, str] = {}
        descriptors: list[ArtifactDescriptor] = []
        for artifact_id, kind, content in base_artifacts:
            base_dict[artifact_id] = content
            descriptors.append(
                ArtifactDescriptor(
                    artifact_id=artifact_id,
                    kind=kind,
                    format="text/plain",
                    version_hash=_hash_content(content),
                    readable=True,
                    writable=True,
                    merge_strategy=_MERGE_STRATEGY,
                    bindings=(),
                )
            )
        self._versions[base_version] = base_dict
        self._inventory[base_version] = tuple(descriptors)
        self._lineage[base_version] = ("", "")

    # ------------------------------------------------------------------ #
    # Read APIs
    # ------------------------------------------------------------------ #
    def artifact_inventory(self, version: str) -> Sequence[ArtifactDescriptor]:
        if version not in self._versions:
            raise KeyError(f"unknown version: {version!r}")
        return self._inventory[version]

    def read_artifacts(
        self, version: str, artifact_ids: Sequence[str]
    ) -> Mapping[str, str]:
        if version not in self._versions:
            raise KeyError(f"unknown version: {version!r}")
        store = self._versions[version]
        return {aid: store[aid] for aid in artifact_ids if aid in store}

    # ------------------------------------------------------------------ #
    # Materialize + edit
    # ------------------------------------------------------------------ #
    def materialize_candidate(
        self, parent_version: str, attempt_id: str
    ) -> CandidateWorkspace:
        if parent_version not in self._versions:
            raise KeyError(f"unknown parent version: {parent_version!r}")
        if attempt_id in self._workspaces:
            raise ValueError(f"attempt_id already in use: {attempt_id!r}")

        # Deep copy the parent's artifacts into an isolated staging dict.
        staging: dict[str, str] = dict(self._versions[parent_version])

        # Materialize a new candidate version derived from the parent name.
        # The version string is opaque to the core; we just need it to be
        # unique and traceable to its parent.
        new_version = f"{parent_version}+{attempt_id}"

        self._workspaces[attempt_id] = staging
        self._versions[new_version] = staging  # same object, keyed by version
        self._lineage[new_version] = (parent_version, attempt_id)

        # Inventory mirrors parent for now; apply_structured_edits will refresh
        # version_hashes in place when content changes.
        parent_descs = self._inventory[parent_version]
        self._inventory[new_version] = tuple(
            ArtifactDescriptor(
                artifact_id=d.artifact_id,
                kind=d.kind,
                format=d.format,
                version_hash=_hash_content(staging[d.artifact_id]),
                readable=d.readable,
                writable=d.writable,
                merge_strategy=d.merge_strategy,
                bindings=d.bindings,
            )
            for d in parent_descs
        )

        return CandidateWorkspace(
            attempt_id=attempt_id,
            version=new_version,
            path=Path(f"<memory:{new_version}>"),
            parent_version=parent_version,
        )

    def apply_structured_edits(
        self, workspace: CandidateWorkspace, edits: Sequence[ArtifactEdit]
    ) -> Mapping[str, str]:
        staging = self._workspaces[workspace.attempt_id]

        # Snapshot of what changed, returned to the core as evidence.
        changed: dict[str, str] = {}

        for edit in edits:
            aid = edit.artifact_id
            if aid not in staging:
                # The core must never request edits outside an adapter-declared
                # write set; raise so a contract bug is loud.
                raise KeyError(f"unknown artifact in edit: {aid!r}")

            operation = edit.operation
            payload = edit.payload

            if operation == "replace":
                new_content = str(payload["content"])
                staging[aid] = new_content
                changed[aid] = new_content
            elif operation == "append":
                sep = "\n" if not staging[aid].endswith("\n") else ""
                new_content = staging[aid] + sep + str(payload["content"])
                staging[aid] = new_content
                changed[aid] = new_content
            else:
                raise ValueError(
                    f"unsupported edit operation {operation!r} on {aid!r}"
                )

            # Refresh inventory version_hash for this artifact.
            descs = list(self._inventory[workspace.version])
            new_descs = [
                (
                    ArtifactDescriptor(
                        artifact_id=d.artifact_id,
                        kind=d.kind,
                        format=d.format,
                        version_hash=_hash_content(staging[d.artifact_id]),
                        readable=d.readable,
                        writable=d.writable,
                        merge_strategy=d.merge_strategy,
                        bindings=d.bindings,
                    )
                    if d.artifact_id == aid
                    else d
                )
                for d in descs
            ]
            self._inventory[workspace.version] = tuple(new_descs)

        return changed

    # ------------------------------------------------------------------ #
    # Rollout + trace
    # ------------------------------------------------------------------ #
    def run_full_rollout(
        self, workspace: CandidateWorkspace, task: EvolutionTask, rollout_id: str
    ) -> object:
        if workspace.attempt_id not in self._workspaces:
            raise KeyError(f"unknown workspace: {workspace.attempt_id!r}")

        staging = self._workspaces[workspace.attempt_id]

        # Build a deterministic fake "agent execution": assemble a flat
        # context string from all artifacts, then produce a one-line output
        # that mentions the task input. The status is "success" unless the
        # task expected a substring we fail to produce.
        context_parts = [f"[{aid}] {content}" for aid, content in staging.items()]
        context = "\n".join(context_parts)

        # The "agent" emits a deterministic output derived from artifacts and
        # the task. If a skill mentions a token the task expects, it appears.
        tokens_in_skills = " ".join(staging.values())
        output = f"answer for {task.task_id}: " + (
            "matched " + task.expected_contract.get("expected_substring", "")
            if task.expected_contract.get("expected_substring")
            and str(task.expected_contract["expected_substring"]) in tokens_in_skills
            else f"generic output using context length={len(context)}"
        )

        status = "success"
        event_id_0 = f"evt-{rollout_id}-0"
        event_id_1 = f"evt-{rollout_id}-1"
        event_id_2 = f"evt-{rollout_id}-2"
        events = (
            TraceEvent(
                event_id=event_id_0,
                kind="state",
                actor_id=None,
                parent_event_id=None,
                payload={"phase": "init", "context_hash": _hash_content(context)},
            ),
            TraceEvent(
                event_id=event_id_1,
                kind="tool",
                actor_id="agent",
                parent_event_id=event_id_0,
                payload={"tool": "retriever", "matched": "expected_substring" in output},
            ),
            TraceEvent(
                event_id=event_id_2,
                kind="model",
                actor_id="agent",
                parent_event_id=event_id_1,
                payload={"phase": "final", "output_len": len(output)},
            ),
        )

        return _RolloutResult(
            workspace=workspace,
            task=task,
            events=events,
            final_output=output,
            status=status,
        )

    def capture_trace(self, rollout_result: object) -> ExecutionTrace:
        if not isinstance(rollout_result, _RolloutResult):
            raise TypeError(
                "FakeAdapter.capture_trace expected a _RolloutResult produced by "
                "this adapter's run_full_rollout"
            )
        rr = rollout_result
        return ExecutionTrace(
            trace_id=f"trace-{uuid.uuid4().hex[:12]}",
            candidate_id=rr.workspace.version,
            task_id=rr.task.task_id,
            events=rr.events,
            final_output=rr.final_output,
            status=rr.status,
            checkpoint_ids=(),  # No checkpoints: replay is unsupported.
        )

    # ------------------------------------------------------------------ #
    # Replay (declared unsupported)
    # ------------------------------------------------------------------ #
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
        raise RuntimeError(
            "FakeAdapter does not support counterfactual replay; the core must "
            "call run_full_rollout instead"
        )
