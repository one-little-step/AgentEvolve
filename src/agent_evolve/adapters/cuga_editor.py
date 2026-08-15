"""CUGA-agent-backed editor implementing the core ``Editor`` protocol.

The whole multi-turn agent loop lives inside ``propose_edit``, so
``agent_evolve.core`` never learns the editor is a CUGA agent.

Isolation (design doc §13): the editor agent is constructed with tracing
detached and no workspace bound, so its own LLM calls cannot enter a rollout
trace and it cannot read a candidate's skills directory. CUGA's singleton
ActivityTracker and global policy DB remain shared in-process; that residual
risk is accepted and guarded by test.
"""
from __future__ import annotations

import functools
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

from agent_evolve.adapters.cuga_editor_evidence import (
    EvidenceView,
    contamination_terms_from,
)
from agent_evolve.adapters.cuga_editor_skills import (
    EDITOR_INSTRUCTIONS,
    EDITOR_SKILLS,
    build_editor_prompt,
)
from agent_evolve.adapters.cuga_editor_state import EditStagingArea
from agent_evolve.adapters.cuga_editor_tools import (
    EditorToolContext,
    build_tool_callables,
    submitted_plan,
)
from agent_evolve.core.contracts import ExecutionTrace
from agent_evolve.core.editor import EditorOutcome, EditorRequest, EditorResponse
from agent_evolve.core.memory import EditMemory


class EditorDeclined(RuntimeError):
    """The editor produced no usable plan.

    Carries the distinct :class:`EditorOutcome` so a caller can tell
    ``no_tool_call`` (the agent did not engage) from ``no_op`` (the agent
    judged no edit warranted). ``repair_once_then_classify`` converts this into
    a recorded non-promotion.
    """

    def __init__(self, outcome: EditorOutcome, message: str) -> None:
        super().__init__(message)
        self.outcome = outcome


def prepare_editor_environment(skills_dir: str | None = None) -> None:
    """Unbind any rollout workspace and point CUGA at the configured model.

    Two independent responsibilities, both required before construction:

    * ``CUGA_FOLDER`` must be cleared. CUGA reads it in the sandbox and in
      ``prepare_node``, so a leftover value from a rollout would hand the
      editor a candidate's skills.
    * The model environment must be configured. Without it CUGA falls back to
      its built-in default (``gpt-4o`` against api.openai.com) and fails with
      "Missing credentials" -- the editor never reaches the model at all.
      ``RuntimeSettings.from_env`` reads CUGA_MODEL/LITELLM_MODEL and the
      matching base URL and key.
    """
    os.environ.pop("CUGA_FOLDER", None)
    from agent_evolve.cuga_wrapper import RuntimeSettings, prepare_cuga_environment

    prepare_cuga_environment()
    RuntimeSettings.from_env().configure_cuga_environment()
    if skills_dir:
        # The constructor argument does not reach every consumer on this build:
        # create_sandbox_tools and prepare_node read CUGA_FOLDER directly.
        os.environ["CUGA_FOLDER"] = str(skills_dir)


def materialize_editor_skills(workspace_dir: Path | str) -> str:
    """Write the editor's own skills into an isolated CUGA workspace.

    Required for the skills to exist at all. ``enable_skills=True`` with
    ``cuga_folder=None`` makes CUGA resolve its skill root to ``<cwd>/.cuga``,
    where it loads whatever unrelated skills a previous run left behind -- a
    live run was observed loading a stale ``web-research`` skill and none of
    the editor's four. Skills only reach the model when they are on disk under
    a folder CUGA is pointed at.
    """
    from agent_evolve.cuga_wrapper import materialize_harness

    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    materialize_harness({"skills": EDITOR_SKILLS}, workspace)
    return str(workspace)


def editor_agent_kwargs(skills_dir: str | None = None) -> dict[str, object]:
    """Construction arguments that keep the editor out of rollout traces.

    ``skills_dir`` must be the folder CONTAINING ``skills/``, not the
    ``skills/`` directory itself: CUGA discovers ``<skills_folder>/skills/**
    /SKILL.md``. Both ``cuga_folder`` and ``skills_folder`` are set because
    CUGA resolves its skill root from ``cuga_folder`` in some paths and reads
    ``skills_folder`` in others.
    """
    kwargs: dict[str, object] = {
        # No callbacks: the GraphEventCollector must never see editor LLM calls,
        # or the editor would pollute the evidence the analyzer reads.
        "callbacks": [],
        "special_instructions": EDITOR_INSTRUCTIONS,
        "enable_skills": bool(skills_dir),
        "auto_load_policies": False,
        # Reset the process-global policy store: a playbook written by any
        # earlier run keeps matching for the editor otherwise.
        "reset_policy_storage": True,
    }
    # cuga_folder must point at the editor's own workspace, never be left None
    # (which resolves to <cwd>/.cuga and picks up stale global skills).
    kwargs["cuga_folder"] = skills_dir
    kwargs["skills_folder"] = skills_dir
    return kwargs


def _evidence_summary(view: EvidenceView) -> str:
    mechanism = view.mechanism()
    actors = ", ".join(
        f"{a['actor_id']} (blame {a['blame']})" for a in view.blamed_actors()
    ) or "none attributed"
    return (
        f"MECHANISM: {mechanism['mechanism']}\n"
        f"SEVERITY: {mechanism['severity']}\n"
        f"BLAMED ACTORS: {actors}\n"
        f"TASK: {view.task_input()}"
    )


def _parent_summary(request: EditorRequest) -> str:
    """State the donor inventory in the prompt.

    Without this the editor has no signal that crossover is even possible: two
    live runs with a donor whose artifact already contained the missing
    capability never called list_parents, because nothing in the prompt said a
    donor existed.
    """
    donors = [p for p in request.parents if not p.is_primary]
    if not donors:
        return "PARENTS: primary only, no donors available."
    described = "; ".join(
        f"{d.candidate_id} (scores {dict(d.score_summary)})" for d in donors
    )
    return (
        f"PARENTS: {len(donors)} donor parent(s) available: {described}. "
        "Inspect a donor's artifact before deciding to refine."
    )


@dataclass(slots=True)
class CugaEditorAgent:
    """Editor backed by a multi-turn CUGA agent."""

    adapter: object
    memory: EditMemory
    # Injected for tests: (tool_callables, prompt) -> agent answer. When None,
    # a real CugaAgent is constructed.
    agent_factory: Callable[[dict, str], str] | None = None
    editor_model_id: str = "cuga-editor-agent"
    trace: ExecutionTrace | None = None
    last_outcome: EditorOutcome = EditorOutcome.UNAVAILABLE
    last_parents_read: tuple[str, ...] = ()
    last_tools_called: tuple[str, ...] = field(default_factory=tuple)
    _active_ctx: EditorToolContext | None = None
    # SDK-reported tool calls from the last real run (independent evidence).
    last_sdk_tool_calls: tuple = ()

    def propose_edit(self, request: EditorRequest) -> EditorResponse:
        ctx = self._build_context(request)
        self._active_ctx = ctx
        callables = build_tool_callables(ctx)
        recorded, names = self._recording_wrapper(callables)
        prompt = build_editor_prompt(
            _evidence_summary(ctx.evidence) + "\n" + _parent_summary(request)
        )

        try:
            self._run_agent(recorded, prompt)
        except Exception as exc:  # noqa: BLE001 - classify, never propagate raw
            self.last_tools_called = tuple(names)
            self.last_outcome = EditorOutcome.UNAVAILABLE
            raise EditorDeclined(
                EditorOutcome.UNAVAILABLE, f"editor agent failed: {exc}"
            ) from exc

        self.last_tools_called = tuple(names)
        plan = submitted_plan(ctx)

        if plan is None:
            # Includes the case where edits were staged but never finalized:
            # unfinalized work is discarded, not silently applied.
            self.last_outcome = EditorOutcome.NO_TOOL_CALL
            self.last_parents_read = ()
            raise EditorDeclined(
                EditorOutcome.NO_TOOL_CALL,
                "editor agent never called submit_edit_plan",
            )

        self.last_parents_read = tuple(plan["parents_read"])

        if plan["declined"]:
            self.last_outcome = EditorOutcome.NO_OP
            raise EditorDeclined(
                EditorOutcome.NO_OP,
                f"editor declined to edit: {plan['rationale']}",
            )

        self.last_outcome = EditorOutcome.VALID
        writes = {
            edit.artifact_id: str(edit.payload.get("content", ""))
            for edit in plan["edits"]
        }
        return EditorResponse(
            rationale=plan["rationale"],
            edits=plan["edits"],
            reads=dict(request.current_artifacts),
            writes=writes,
            risks={"summary": plan["risks"]} if plan["risks"] else {},
            expected_effects=(
                {"summary": plan["expected_effect"]}
                if plan["expected_effect"]
                else {}
            ),
            editor_model_id=self.editor_model_id,
        )

    # -------------------------------------------------------------- #
    # Internals
    # -------------------------------------------------------------- #
    def _build_context(self, request: EditorRequest) -> EditorToolContext:
        pool_created = request.pool_created_count
        staging = EditStagingArea(
            write_set=request.write_set,
            creatable_prefix=request.creatable_prefix,
            pool_created_count=pool_created,
        )
        trace = self.trace or ExecutionTrace(
            trace_id="unavailable",
            candidate_id=request.base_workspace.version,
            task_id=request.task.task_id,
            events=(),
            final_output="",
            status="unavailable",
        )
        evidence = EvidenceView(
            analysis=request.analysis,
            trace=trace,
            task=request.task,
            contamination_terms=contamination_terms_from(request.task),
        )
        return EditorToolContext(
            staging=staging,
            evidence=evidence,
            request=request,
            adapter=self.adapter,
            memory=self.memory,
        )

    @staticmethod
    def _recording_wrapper(
        callables: dict[str, Callable[..., str]],
    ) -> tuple[dict[str, Callable[..., str]], list[str]]:
        names: list[str] = []

        def wrap(name: str, fn: Callable[..., str]) -> Callable[..., str]:
            # functools.wraps carries __doc__ and __wrapped__ across the
            # boundary. Both are load-bearing for the real agent path:
            # LangChain's @tool refuses a body with no docstring, and it derives
            # the args schema from the signature, which inspect.signature can
            # only recover by following __wrapped__ through this *args wrapper.
            @functools.wraps(fn)
            def recorded(*args, **kwargs) -> str:
                names.append(name)
                return fn(*args, **kwargs)

            recorded.__name__ = name
            return recorded

        return {name: wrap(name, fn) for name, fn in callables.items()}, names

    def _run_agent(self, callables: dict, prompt: str) -> str:
        if self.agent_factory is not None:
            return self.agent_factory(callables, prompt)
        return self._run_cuga_agent(callables, prompt)

    def _run_cuga_agent(self, callables: dict, prompt: str) -> str:
        """Construct and run a real CUGA agent. SDK import stays local."""
        import asyncio

        from agent_evolve.adapters.cuga_editor_tools import build_editor_tools
        from cuga import CugaAgent

        skills_dir = materialize_editor_skills(
            Path(tempfile.mkdtemp(prefix="agent-evolve-editor-"))
        )
        prepare_editor_environment(skills_dir)
        kwargs = editor_agent_kwargs(skills_dir)
        # Pass the RECORDED callables: rebuilding from ctx would drop the
        # call-recording wrappers, so a run that really executed tools would
        # still report zero tool calls.
        agent = CugaAgent(
            tools=build_editor_tools(self._active_ctx, callables), **kwargs
        )

        async def run() -> str:
            await agent.initialize()
            # track_tool_calls surfaces the SDK's own aggregated tool-call list,
            # which is independent evidence from our wrapper's ledger.
            result = await agent.invoke(prompt, track_tool_calls=True)
            self.last_sdk_tool_calls = tuple(
                getattr(result, "tool_calls", ()) or ()
            )
            return str(result)

        return asyncio.run(run())
