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
import gc
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
from agent_evolve.core.editor import (
    EditorOutcome,
    EditorRequest,
    EditorResponse,
    ParentContext,
)
from agent_evolve.core.memory import EditMemory
from agent_evolve.core.run_logging import RunLogSink


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


def _fault_lines(parent: ParentContext, limit: int = 4) -> str:
    """This parent's diagnosed faults, worst first.

    Severity is an *attention* signal here, so the ordering matters: the editor
    gets one attempt, and presenting a mild fault before a severe one spends that
    attempt on the lesser problem. Capped at ``limit`` so a parent with a long
    history cannot crowd the blame evidence out of the prompt.

    Only cluster ids, numbers and artifact ids cross into the prompt --
    :class:`Issue` carries no prose, so no mechanism description or task content
    can leak through this text.
    """
    if not parent.issues:
        return "    no diagnosed faults on record"
    ranked = sorted(
        parent.issues, key=lambda i: (-i.severity, i.task_id, i.mechanism_cluster_id)
    )
    lines = [
        f"    - task {i.task_id}, mechanism {i.mechanism_cluster_id}, "
        f"severity {i.severity}, attributable to: "
        f"{', '.join(i.writable_artifact_ids) or 'no writable surface'}"
        for i in ranked[:limit]
    ]
    if len(ranked) > limit:
        lines.append(f"    - ... and {len(ranked) - limit} further fault(s)")
    return "\n".join(lines)


def _parent_summary(request: EditorRequest) -> str:
    """State each parent's diagnosed faults, and the donor inventory, in the prompt.

    Two distinct failures this text prevents, and both were measured rather than
    assumed:

    * **Crossover was undiscoverable.** Two live runs with a donor whose artifact
      already contained the missing capability never called ``list_parents``,
      because nothing in the prompt said a donor existed.
    * **Parent weaknesses were invisible (SV-10).** The faults reached
      ``ParentContext.issues`` and the ``list_parents`` tool, but this text
      rendered parents as scores alone -- ``c-donor (scores {'task-a': 0.9})``.
      ``EDITOR_INSTRUCTIONS`` gates tool use on what the evidence reports, so
      evidence the prompt never mentions is evidence the model never asks for.

    A score says a donor is *better*; its faults say *where it is not*. Both are
    needed before choosing to transplant rather than refine.
    """
    primary = next((p for p in request.parents if p.is_primary), None)
    donors = [p for p in request.parents if not p.is_primary]

    blocks: list[str] = []
    if primary is not None:
        blocks.append(
            f"THIS PARENT ({primary.candidate_id}) HAS THESE DIAGNOSED FAULTS "
            "(worst first):\n" + _fault_lines(primary)
        )

    if not donors:
        blocks.append("PARENTS: primary only, no donors available.")
    else:
        described = "; ".join(
            f"{d.candidate_id} (scores {dict(d.score_summary)})" for d in donors
        )
        donor_blocks = "\n".join(
            f"  {d.candidate_id} known faults:\n{_fault_lines(d)}" for d in donors
        )
        blocks.append(
            f"PARENTS: {len(donors)} donor parent(s) available: {described}. "
            "Inspect a donor's artifact before deciding to refine.\n"
            f"{donor_blocks}"
        )

    return "\n".join(blocks)


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
    #: When set and active, records the prompt, the raw answer, the ordered
    #: tool-call ledger and the terminal outcome. Off by default so a
    #: measurement run writes nothing.
    log_sink: RunLogSink | None = None
    #: D5/TL: optional factory producing the complementary-parent payload
    #: provider for one request. The composition root attaches it after
    #: construction (``attach_complement_provider``), closing over the runner:
    #: the index must be rebuilt from live TS2 state per attempt. ``None``
    #: leaves the tool reporting itself unavailable -- zero cost, zero risk.
    complement_provider_factory: Callable[[EditorRequest], Callable[[int], dict]] | None = None
    #: W4-prime: per-request factory producing the replay provider (keyword
    #: args -> raw report dict). Attached by the composition root once the
    #: facade exists; ``None`` makes the tool report itself unavailable.
    replay_provider_factory: Callable[[EditorRequest], Callable[..., dict]] | None = None

    def propose_edit(self, request: EditorRequest) -> EditorResponse:
        ctx = self._build_context(request)
        self._active_ctx = ctx
        callables = build_tool_callables(ctx)
        recorded, names = self._recording_wrapper(callables)
        prompt = build_editor_prompt(
            _evidence_summary(ctx.evidence) + "\n" + _parent_summary(request)
        )
        self._log(request, {"event": "editor_prompt", "prompt": prompt})

        try:
            answer = self._run_agent(recorded, prompt)
        except Exception as exc:  # noqa: BLE001 - classify, never propagate raw
            self.last_tools_called = tuple(names)
            self.last_outcome = EditorOutcome.UNAVAILABLE
            self._log_outcome(
                request, names, error=f"{type(exc).__name__}: {exc}"
            )
            raise EditorDeclined(
                EditorOutcome.UNAVAILABLE, f"editor agent failed: {exc}"
            ) from exc

        self.last_tools_called = tuple(names)
        self._log(request, {"event": "editor_answer", "answer": str(answer)})
        plan = submitted_plan(ctx)

        if plan is None:
            # Includes the case where edits were staged but never finalized:
            # unfinalized work is discarded, not silently applied.
            self.last_outcome = EditorOutcome.NO_TOOL_CALL
            self.last_parents_read = ()
            self._log_outcome(
                request, names, error="never called submit_edit_plan"
            )
            raise EditorDeclined(
                EditorOutcome.NO_TOOL_CALL,
                "editor agent never called submit_edit_plan",
            )

        self.last_parents_read = tuple(plan["parents_read"])

        if plan["declined"]:
            self.last_outcome = EditorOutcome.NO_OP
            self._log_outcome(request, names, plan=plan)
            raise EditorDeclined(
                EditorOutcome.NO_OP,
                f"editor declined to edit: {plan['rationale']}",
            )

        self.last_outcome = EditorOutcome.VALID
        self._log_outcome(request, names, plan=plan)
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
    def _log(self, request: EditorRequest, record: dict[str, object]) -> None:
        """Best-effort write. Never raises: an observer must not fail an edit.

        A logging failure that propagated would discard a multi-turn agent run
        that has already been paid for, so every error is swallowed -- including
        a sink that does not behave like one.
        """
        sink = self.log_sink
        if sink is None:
            return
        try:
            sink.write_record(
                f"{request.base_workspace.version}__{request.task.task_id}",
                {
                    "candidate_version": request.base_workspace.version,
                    "task_id": request.task.task_id,
                    "issue_id": request.issue_id,
                    "attempt_id": request.base_workspace.attempt_id,
                    **record,
                },
            )
        except Exception:  # noqa: BLE001 - capture is an observer, never a gate
            pass

    def _log_outcome(
        self,
        request: EditorRequest,
        names: list[str],
        *,
        plan: dict | None = None,
        error: str | None = None,
    ) -> None:
        """One terminal record per attempt, on every path including a decline.

        The declined paths are the reason this exists: ``no_op`` and
        ``no_tool_call`` produce no plan and no response, so without this record
        the most informative outcomes were the only ones leaving no artifact.
        """
        record: dict[str, object] = {
            "event": "editor_outcome",
            "outcome": self.last_outcome.value,
            "tools_called": list(names),
            "sdk_tool_calls": [str(c) for c in self.last_sdk_tool_calls],
            "parents_read": list(self.last_parents_read),
        }
        if plan is not None:
            record["declined"] = bool(plan["declined"])
            record["rationale"] = plan["rationale"]
            record["risks"] = plan["risks"]
            record["expected_effect"] = plan["expected_effect"]
            record["artifact_ids"] = [e.artifact_id for e in plan["edits"]]
        if error is not None:
            record["error"] = error
        self._log(request, record)

    def _build_context(self, request: EditorRequest) -> EditorToolContext:
        pool_created = request.pool_created_count
        staging = EditStagingArea(
            write_set=request.write_set,
            creatable_prefixes=request.creatable_prefixes,
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
            complement_provider=(
                self.complement_provider_factory(request)
                if self.complement_provider_factory is not None
                else None
            ),
            replay_provider=(
                self.replay_provider_factory(request)
                if self.replay_provider_factory is not None
                else None
            ),
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
            # `aclose`, NOT `close`: the installed SDK exposes only `aclose`.
            # `finally` so a failed edit -- the case a retry loop repeats -- does
            # not leak the agent's graph and message history. See the 2026-08-19
            # memory-exhaustion report: one agent per propose_edit, never closed.
            try:
                await agent.initialize()
                # track_tool_calls surfaces the SDK's own aggregated tool-call
                # list, which is independent evidence from our wrapper's ledger.
                result = await agent.invoke(prompt, track_tool_calls=True)
                self.last_sdk_tool_calls = tuple(
                    getattr(result, "tool_calls", ()) or ()
                )
                return str(result)
            finally:
                aclose = getattr(agent, "aclose", None)
                if aclose is not None:
                    await aclose()

        try:
            return asyncio.run(run())
        finally:
            del agent
            gc.collect()


def attach_complement_provider(
    agent: CugaEditorAgent,
    provider_factory: Callable[[EditorRequest], Callable[[int], dict]],
) -> None:
    """D5/TL: give this editor the voluntary ``list_complementary_parents`` tool.

    Call once from the composition root, after both runner and editor exist.
    The factory receives each ``EditorRequest``; the INNER callable runs per
    tool invocation, so every attempt reads the live TS2 store rather than a
    stale snapshot. Unattached, the tool reports itself unavailable and costs
    nothing.
    """
    agent.complement_provider_factory = provider_factory


def attach_replay_provider(
    agent: CugaEditorAgent,
    provider_factory: Callable[[EditorRequest], Callable[..., dict]],
) -> None:
    """W4-prime: give this editor the voluntary ``run_replay_experiment`` tool.

    Mirror of :func:`attach_complement_provider`. The factory receives each
    ``EditorRequest``; the INNER callable runs per tool invocation, so every
    experiment reads live state (staged edits + the request's task) rather
    than a snapshot frozen at composition. Unattached, the tool reports itself
    unavailable and costs nothing.
    """
    agent.replay_provider_factory = provider_factory
