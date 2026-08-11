"""A deterministic fake editor for tests and offline demos.

The fake editor implements :class:`agent_evolve.core.editor.Editor` by
inspecting the :class:`EditorRequest` and producing a single
``ArtifactEdit`` that targets the highest-blame artifact in the analysis.
The edit content is chosen to "fix" the mechanism by inserting the task's
expected substring (if any) into the artifact.

This is deliberately simple. Real editor implementations will be backed by
LLMs that read current artifact contents, reason about the failure, and
propose structured edits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from agent_evolve.core.contracts import ArtifactEdit
from agent_evolve.core.editor import EditorRequest, EditorResponse


@dataclass(slots=True)
class FakeEditor:
    """Deterministic editor that injects the expected substring."""

    editor_model_id: str = "fake-editor"

    def propose_edit(self, request: EditorRequest) -> EditorResponse:
        # Pick the highest-blame artifact in the analysis.
        blame_graph = request.analysis.blame_graph
        if not blame_graph.nodes:
            # No blame -> no actionable edit. But EditorResponse requires at
            # least one edit, so we emit a no-op replace on the first
            # write_set artifact.
            target = request.write_set[0]
            new_content = request.current_artifacts.get(target, "")
            return EditorResponse(
                rationale="no blame assigned; no-op edit",
                edits=(
                    ArtifactEdit(
                        artifact_id=target,
                        operation="replace",
                        payload={"content": new_content},
                    ),
                ),
                reads=dict(request.current_artifacts),
                writes={target: new_content},
                risks={},
                expected_effects={},
                editor_model_id=self.editor_model_id,
            )

        # Sort nodes by blame descending; pick the top one whose artifacts
        # intersect the write_set.
        sorted_nodes = sorted(
            blame_graph.nodes, key=lambda n: (-n.blame, n.actor_id)
        )
        target = None
        for n in sorted_nodes:
            for aid in n.artifacts:
                if aid in request.write_set:
                    target = aid
                    break
            if target is not None:
                break
        if target is None:
            # No blamed artifact is in the write_set; fall back to first.
            target = request.write_set[0]

        # Construct the new content: append the expected substring to the
        # current content (or replace if no current content).
        expected = request.task.expected_contract.get("expected_substring", "")
        current = request.current_artifacts.get(target, "")
        if expected and str(expected) not in current:
            sep = "\n" if current and not current.endswith("\n") else ""
            new_content = f"{current}{sep}use {expected} here"
        else:
            # No expected substring, or already present; minor tweak.
            new_content = current + " (refined)" if current else "initial"

        return EditorResponse(
            rationale=f"address {request.analysis.mechanism} by editing {target}",
            edits=(
                ArtifactEdit(
                    artifact_id=target,
                    operation="replace",
                    payload={"content": new_content},
                ),
            ),
            reads=dict(request.current_artifacts),
            writes={target: new_content},
            risks={target: "may regress unrelated tasks"},
            expected_effects={target: "expected substring should appear in output"},
            editor_model_id=self.editor_model_id,
        )
