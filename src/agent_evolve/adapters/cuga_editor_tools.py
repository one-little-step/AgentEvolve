"""Tool clusters handed to the CUGA editor agent.

Two layers on purpose:

* ``build_tool_callables`` returns plain functions with NO CUGA dependency, so
  every authorization, evidence and capture rule is unit-testable offline.
* ``build_editor_tools`` wraps those callables with ``tracked_tool`` + ``tool``,
  deferring the SDK import into the function body.

Every tool returns a JSON string. CUGA tools must return strings, and returning
a structured error keeps one failing tool from aborting the agent run. Nothing
here raises into the agent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from agent_evolve.adapters.cuga_editor_evidence import EvidenceView
from agent_evolve.adapters.cuga_editor_state import EditStagingArea
from agent_evolve.core.editor import EditorRequest
from agent_evolve.core.memory import EditMemory

TOOL_APP_NAMES: dict[str, str] = {
    # evidence
    "get_mechanism": "evidence",
    "list_blamed_actors": "evidence",
    "get_task_input": "evidence",
    "list_trace_actors": "evidence",
    "read_trace_events": "evidence",
    # harness
    "list_artifacts": "harness",
    "read_artifact": "harness",
    "stage_replace": "harness",
    "stage_create": "harness",
    "list_staged": "harness",
    "unstage": "harness",
    # history
    "search_edit_history": "history",
    "get_attempt_outcome": "history",
    # parents
    "list_parents": "parents",
    "read_parent_artifact": "parents",
    # rollout
    "list_rollout_tools": "rollout",
    # submit
    "submit_edit_plan": "submit",
}

_MAX_HISTORY_RECORDS = 5


@dataclass(slots=True)
class EditorToolContext:
    """Per-request state the tools close over.

    Bound to one ``propose_edit`` call. Nothing here is global, so concurrent
    editors cannot interfere.
    """

    staging: EditStagingArea
    evidence: EvidenceView
    request: EditorRequest
    adapter: object
    memory: EditMemory
    _plan: dict | None = field(default=None, repr=False)


def submitted_plan(ctx: EditorToolContext) -> dict | None:
    """The finalized plan, or ``None`` if the agent never finalized."""
    return ctx._plan


def _ok(**payload: object) -> str:
    return json.dumps(payload, default=str)


def _err(message: str) -> str:
    return json.dumps({"status": "error", "message": message})


def build_tool_callables(ctx: EditorToolContext) -> dict[str, Callable[..., str]]:
    """Build the tool bodies for one editor request."""

    # ---------------------------------------------------------- evidence
    def get_mechanism() -> str:
        """Return the diagnosed failure mechanism and its severity."""
        return _ok(**ctx.evidence.mechanism())

    def list_blamed_actors() -> str:
        """List actors blamed for the failure, highest blame first, with the artifacts each owns."""
        return json.dumps(list(ctx.evidence.blamed_actors()), default=str)

    def get_task_input() -> str:
        """Return the task input text the failing agent was given."""
        return _ok(input_text=ctx.evidence.task_input())

    def list_trace_actors() -> str:
        """List the distinct actors that appear in the failed run's trace."""
        return json.dumps(list(ctx.evidence.actors()))

    def read_trace_events(
        kind: str = "", actor_id: str = "", limit: int = 50
    ) -> str:
        """Read trace events from the failed run, optionally filtered by kind and actor_id."""
        try:
            events = ctx.evidence.events(
                kind=kind or None,
                actor_id=actor_id or None,
                limit=max(1, min(int(limit), 200)),
            )
        except Exception as exc:  # noqa: BLE001 - never raise into the agent
            return _err(f"read_trace_events failed: {exc}")
        return json.dumps(list(events), default=str)

    # ---------------------------------------------------------- harness
    def list_artifacts() -> str:
        """List the artifact ids you may modify, and the required prefixes for new artifacts."""
        return _ok(
            writable=list(ctx.request.write_set),
            creatable_prefixes=list(ctx.request.creatable_prefixes),
        )

    def read_artifact(artifact_id: str) -> str:
        """Read the current content of one writable artifact."""
        content = ctx.request.current_artifacts.get(artifact_id)
        if content is None:
            return _err(
                f"{artifact_id!r} is not readable; call list_artifacts first"
            )
        return _ok(artifact_id=artifact_id, content=content)

    def stage_replace(artifact_id: str, content: str) -> str:
        """Stage replacement content for an existing artifact. Returns whether it was accepted."""
        outcome = ctx.staging.stage_replace(artifact_id, content)
        return _ok(accepted=outcome.accepted, reason=outcome.reason)

    def stage_create(artifact_id: str, content: str) -> str:
        """Stage a brand-new artifact. The id must start with the creatable prefix."""
        outcome = ctx.staging.stage_create(artifact_id, content)
        return _ok(accepted=outcome.accepted, reason=outcome.reason)

    def list_staged() -> str:
        """List the artifact ids you have staged so far."""
        return _ok(staged=list(ctx.staging.staged_ids()))

    def unstage(artifact_id: str) -> str:
        """Remove one staged edit, freeing its creation slot if it was a creation."""
        outcome = ctx.staging.unstage(artifact_id)
        return _ok(accepted=outcome.accepted, reason=outcome.reason)

    # ---------------------------------------------------------- history
    def search_edit_history() -> str:
        """Search past edit attempts for this issue, so you do not repeat a failed strategy."""
        try:
            records = ctx.memory.retrieve(
                ctx.request.issue_id, max_records=_MAX_HISTORY_RECORDS
            )
        except Exception as exc:  # noqa: BLE001
            return _err(f"search_edit_history failed: {exc}")
        return json.dumps(
            [
                {
                    "attempt_id": r.attempt_id,
                    "artifact_ids": list(r.artifact_ids),
                    "outcome": r.outcome,
                    "summary": r.summary,
                }
                for r in records
            ],
            default=str,
        )

    def get_attempt_outcome(attempt_id: str) -> str:
        """Get the recorded status and summary of one past attempt."""
        try:
            attempt = ctx.memory.get(attempt_id)
        except KeyError:
            return _err(f"unknown attempt_id: {attempt_id!r}")
        return _ok(
            attempt_id=attempt.attempt_id,
            status=attempt.status.value,
            artifact_ids=list(attempt.artifact_ids),
            summary=attempt.sanitized_reasoning,
        )

    # ---------------------------------------------------------- parents
    def list_parents() -> str:
        """List the candidate parents available, marking which is primary, each one's scores, and each one's diagnosed faults."""
        return json.dumps(
            [
                {
                    "candidate_id": p.candidate_id,
                    "is_primary": p.is_primary,
                    "score_summary": dict(p.score_summary),
                    # SV-10: what this parent is weak at, so an edit can target a
                    # mechanism instead of inferring one from a score. Cluster
                    # ids, numbers and trace refs only -- never mechanism prose,
                    # because this payload is a persistence surface.
                    "issues": [
                        {
                            "task_id": i.task_id,
                            "mechanism_cluster_id": i.mechanism_cluster_id,
                            "severity": i.severity,
                            "confidence": i.confidence,
                            "artifact_ids": list(i.writable_artifact_ids),
                            "evidence_refs": list(i.evidence_refs),
                        }
                        for i in p.issues
                    ],
                }
                for p in ctx.request.parents
            ],
            default=str,
        )

    def read_parent_artifact(parent_id: str, artifact_id: str) -> str:
        """Read an artifact from a donor parent, so you can transplant content it already has."""
        parent = next(
            (p for p in ctx.request.parents if p.candidate_id == parent_id), None
        )
        if parent is None:
            return _err(f"unknown parent: {parent_id!r}; call list_parents first")
        try:
            contents = ctx.adapter.read_artifacts(parent.version, (artifact_id,))
        except Exception as exc:  # noqa: BLE001
            return _err(f"read_parent_artifact failed: {exc}")
        # Record only a read that actually returned content: provenance must
        # reflect what the editor used, not what it attempted.
        ctx.staging.record_parent_read(parent_id)
        return _ok(
            parent_id=parent_id,
            artifact_id=artifact_id,
            content=contents[artifact_id],
        )

    # ---------------------------------------------------------- rollout
    def list_rollout_tools() -> str:
        """List the tools the failing rollout agent has, with signature and purpose.

        Use this before claiming a task was impossible: the agent may already own
        a capability its harness never told it about.
        """
        try:
            from agent_evolve.cuga_wrapper.tools import rollout_tool_inventory

            inventory = rollout_tool_inventory()
        except Exception as exc:  # noqa: BLE001 - never raise into the agent
            return _err(f"list_rollout_tools failed: {exc}")
        return _ok(tools=[dict(entry) for entry in inventory], count=len(inventory))

    # ---------------------------------------------------------- submit
    def submit_edit_plan(
        rationale: str, risks: str = "", expected_effect: str = ""
    ) -> str:
        """Finalize your work. Call this exactly once, including when declining to edit.

        Unfinalized staged edits are discarded, so this call is mandatory.
        """
        if not rationale.strip():
            return _ok(
                accepted=False,
                reason="rationale is required, including when declining",
            )
        edits = ctx.staging.edits()
        ctx._plan = {
            "edits": edits,
            "rationale": rationale,
            "risks": risks,
            "expected_effect": expected_effect,
            "declined": not edits,
            "parents_read": ctx.staging.parents_read(),
        }
        return _ok(
            accepted=True,
            staged=list(ctx.staging.staged_ids()),
            declined=not edits,
        )

    return {
        "get_mechanism": get_mechanism,
        "list_blamed_actors": list_blamed_actors,
        "get_task_input": get_task_input,
        "list_trace_actors": list_trace_actors,
        "read_trace_events": read_trace_events,
        "list_artifacts": list_artifacts,
        "read_artifact": read_artifact,
        "stage_replace": stage_replace,
        "stage_create": stage_create,
        "list_staged": list_staged,
        "unstage": unstage,
        "search_edit_history": search_edit_history,
        "get_attempt_outcome": get_attempt_outcome,
        "list_parents": list_parents,
        "read_parent_artifact": read_parent_artifact,
        "list_rollout_tools": list_rollout_tools,
        "submit_edit_plan": submit_edit_plan,
    }


def build_editor_tools(
    ctx: EditorToolContext,
    callables: dict[str, Callable[..., str]] | None = None,
) -> list:
    """Wrap the tool bodies as tracked LangChain tools.

    This is the only place the CUGA SDK is imported from this module, mirroring
    ``cuga_wrapper.tools.build_tools``.

    ``callables`` lets the caller supply already-wrapped bodies (for example
    the agent's call-recording wrappers). Rebuilding from ``ctx`` here instead
    would silently discard that instrumentation, so tool-execution evidence
    would read as empty even on a run where tools really executed.
    """
    from langchain_core.tools import tool

    from cuga import tracked_tool

    built = []
    for name, fn in (callables or build_tool_callables(ctx)).items():
        fn.__name__ = name
        wrapped = tracked_tool(app_name=TOOL_APP_NAMES[name])(fn)
        built.append(tool(wrapped))
    return built
