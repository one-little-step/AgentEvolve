"""Shared Interface B mechanism: run a CUGA workspace agent over tool callables.

RHO's group diagnoser, candidate optimizer, and preference judge are all
workspace agents in the published implementation. They differ in prompt, tool
set, and instructions -- not in mechanism -- so the mechanism lives here once.

Two layers, mirroring ``cuga_editor_tools``:

* the caller supplies plain callables with NO CUGA dependency, so authorization
  and capture rules stay unit-testable offline, and
* this module wraps them with ``tracked_tool`` + ``tool`` and defers every SDK
  import into the function body.

Tool calls are recorded by wrapping the callables, because the agent's prose is
not evidence that a tool ran, and ``InvokeResult.tool_calls`` is not reliable as
sole ground truth. That is the same tool-execution-over-narration principle the
editor already relies on. ``sdk_tool_calls`` is kept as *independent*
corroboration only; ``tools_called`` is the ledger.

``cuga_editor`` is deliberately NOT refactored onto this helper: the genetic path
works, and changing it under deadline would risk a measured result for a
cosmetic gain.
"""
from __future__ import annotations

import functools
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

#: Prompt framing that CUGA CodeAct tool use empirically depends on. Whether the
#: agent calls a tool at all is a near-deterministic function of this wording:
#: without an explicit "write and execute fenced Python" contract, observed runs
#: emitted no fence at all, the graph never reached the sandbox, and the model
#: narrated that tools were unavailable. CUGA also executes only the FIRST fenced
#: block in a response and silently discards the rest.
#: Default ``special_instructions``.
#:
#: MEASURED ON LIVE ROUNDS, not styled, and not generalisable. Tool invocation
#: on ``azure/gpt-5.6-luna`` is a deterministic function of prompt wording and
#: all-or-nothing per phrasing (see
#: ``reference/cuga_example_wrapper/docs/cuga-integration-learnings.md``).
#:
#: What was actually measured here, and why this long form is kept:
#:
#: * On a trivial one-tool probe, a two-line imperative beat this long form.
#:   That result does NOT transfer: the probe has one tool and no evidence to
#:   read, so it is not representative of the real agents.
#: * On a live RHO round with the real diagnoser and optimizer prompts, THIS
#:   long form produced 2 of 2 observed diagnoses and 3 of 3 distinct
#:   candidates. Replacing it with the two-line form on the same round dropped
#:   the optimizer to 0 of 3, all discarded ``NO_TOOL_CALL``.
#:
#: Prefer the live-round measurement over the probe. Re-measure on a live round
#: before editing this text; a probe-only A/B is not sufficient evidence.
WORKSPACE_AGENT_TOOL_CONTRACT = """\
HOW TO ACT

Write and execute Python code that calls the provided tools, then report the
exact values those calls returned.

* You act ONLY by writing and executing Python code that calls the tools.
  Narrating an intention does nothing; only executed calls count.
* Emit exactly ONE fenced Python block per turn. Only the first fenced block in
  a response is executed and the rest are discarded.
* Put every call you want executed in that single block, then print the results.
* Wait for the execution output before deciding the next step. A missing
  variable means the call did not run; re-issue it rather than concluding the
  tools are unavailable.
* You are never "unable to call a tool in this environment". Every tool listed
  below is registered and callable. If you believe you cannot execute one, you
  have not emitted an executable fenced Python block -- emit one.
* Finish only after you have executed the terminal submit tool.
"""


@dataclass(frozen=True, slots=True)
class WorkspaceAgentRun:
    """The outcome of one workspace-agent invocation.

    ``ok`` means the invocation completed without raising. It does NOT mean the
    agent did useful work: check ``no_tool_call`` for the "model narrated instead
    of acting" outcome, which is an observable result rather than a success.
    """

    answer: str = ""
    tools_called: tuple[str, ...] = ()
    sdk_tool_calls: tuple = ()
    error: str = ""

    @property
    def ok(self) -> bool:
        """True when the invocation completed without raising."""
        return not self.error

    @property
    def no_tool_call(self) -> bool:
        """True when the agent completed but executed no tool at all."""
        return self.ok and not self.tools_called


#: Appended as the LAST thing in every Interface B prompt.
#:
#: Measured: with the same body, a prompt ending on a submission schema produced
#: a complete narration and an empty tool ledger, while the same prompt ending
#: on this directive executed seven tools. Whatever the model reads last decides
#: whether it emits a fenced block, so nothing may be appended after this.
_EXECUTE_NOW = """
BEGIN NOW

Your first response must contain one fenced Python block that calls list_tools()
and the read-only tools you need, and prints their results. Do not describe a
plan -- emit the block. Nothing you write counts until the terminal submit tool
has actually executed.
"""


def tool_roster(callables: Mapping[str, Callable[..., str]]) -> str:
    """Render each tool's exact call signature and one-line purpose.

    CUGA's tool use is strongly prompt-conditioned: the model decides whether to
    emit an executable block at all partly from how concretely the tools are
    presented. Naming every tool with its real signature -- derived from the
    callable, so it cannot drift from what is actually registered -- removes the
    "I am unable to call a tool in this environment" failure mode, where the
    model narrates because it is unsure what it may call.
    """
    rows: list[str] = []
    for name, fn in callables.items():
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            sig = "(...)"
        doc = (inspect.getdoc(fn) or "").strip().splitlines()
        purpose = doc[0] if doc else "no description"
        rows.append(f"  {name}{sig}\n      {purpose}")
    return "\n".join(rows)


def build_list_tools(callables: Mapping[str, Callable[..., str]]) -> Callable[[], str]:
    """Build a ``list_tools`` callable describing ``callables``.

    Injected into every Interface B agent. It gives the model a cheap first
    action that always succeeds, which both confirms the sandbox is live and
    establishes the executed-code pattern before anything load-bearing runs.
    The roster is computed from the real callables, so it cannot go stale.
    """
    roster = tool_roster(callables)

    def list_tools() -> str:
        """List every tool available in this session with its exact signature."""
        return roster

    return list_tools


def recording_wrapper(
    callables: Mapping[str, Callable[..., str]],
) -> tuple[dict[str, Callable[..., str]], list[str]]:
    """Wrap each callable so an actual invocation is recorded by name.

    The returned list is live: it accumulates during the run, so evidence of what
    executed survives a later failure.
    """
    names: list[str] = []

    def wrap(name: str, fn: Callable[..., str]) -> Callable[..., str]:
        # functools.wraps carries __doc__ and __wrapped__ across the boundary.
        # Both are load-bearing on the real agent path: LangChain's @tool refuses
        # a body with no docstring, and it derives the args schema from the
        # signature, which inspect.signature can only recover by following
        # __wrapped__ through this *args wrapper. A bare wrapper tells the model
        # every tool takes no arguments.
        @functools.wraps(fn)
        def recorded(*args: object, **kwargs: object) -> str:
            names.append(name)
            return fn(*args, **kwargs)

        recorded.__name__ = name
        return recorded

    return {name: wrap(name, fn) for name, fn in callables.items()}, names


def build_tracked_tools(
    callables: Mapping[str, Callable[..., str]],
    app_names: Mapping[str, str],
) -> list:
    """Wrap callables as tracked LangChain tools. SDK imports stay local."""
    from cuga import tracked_tool
    from langchain_core.tools import tool

    built = []
    for name, fn in callables.items():
        # @tool derives the tool name the model sees from __name__, which a
        # lambda or a locally-defined body may not carry correctly.
        fn.__name__ = name
        wrapped = tracked_tool(app_name=app_names.get(name, "rho"))(fn)
        built.append(tool(wrapped))
    return built


def prepare_workspace_environment(skills_dir: str | None = None) -> None:
    """Point CUGA at the configured model and unbind any rollout workspace.

    ``CUGA_FOLDER`` is process-global and is read directly by the sandbox and by
    ``prepare_node``, so a leftover value from a rollout would hand this agent a
    candidate's skills. Clearing it is required, and the constructor argument is
    not sufficient on its own.

    Do NOT disable the knowledge engine here. Setting
    ``DYNACONF_KNOWLEDGE__ENABLED=false`` was tried to avoid the engine's
    process-global single-writer lock, and it stopped the agent from calling any
    tool at all -- a verified 7-tool diagnosis run dropped to an empty ledger.
    The knowledge tools are part of the tool surface CodeAct is primed on, so
    removing them changes the model's behaviour, not just the engine's. The
    lock contention is instead avoided by not holding a rollout engine open
    across a workspace-agent call.
    """
    os.environ.pop("CUGA_FOLDER", None)
    from agent_evolve.cuga_wrapper import RuntimeSettings, prepare_cuga_environment

    prepare_cuga_environment()
    RuntimeSettings.from_env().configure_cuga_environment()
    if skills_dir:
        os.environ["CUGA_FOLDER"] = str(skills_dir)


def workspace_agent_kwargs(
    special_instructions: str,
    skills_dir: str | None = None,
) -> dict[str, object]:
    """Construction arguments that keep this agent out of rollout traces.

    ``skills_dir`` must be the folder CONTAINING ``skills/``. Both
    ``cuga_folder`` and ``skills_folder`` are set because CUGA resolves its skill
    root from one in some code paths and the other in others. Never leave
    ``cuga_folder`` as ``None``: it then resolves to ``<cwd>/.cuga`` and loads
    whatever stale skills a previous run left behind.

    No ``temperature`` is ever sent: the endpoint rejects ``0.0`` and reasoning
    models skip it anyway.
    """
    return {
        # No callbacks: the rollout GraphEventCollector must never see this
        # agent's LLM calls, or it would pollute the evidence it reads.
        "callbacks": [],
        "special_instructions": special_instructions,
        "enable_skills": bool(skills_dir),
        "auto_load_policies": False,
        # Policy storage is process-global: a playbook written by an earlier run
        # keeps matching otherwise.
        "reset_policy_storage": True,
        "cuga_folder": skills_dir,
        "skills_folder": skills_dir,
    }


def _run_real_agent(
    callables: Mapping[str, Callable[..., str]],
    prompt: str,
    app_names: Mapping[str, str],
    skills_dir: Path | None,
    special_instructions: str,
) -> tuple[str, tuple]:
    """Construct and run a real CugaAgent. SDK import stays local."""
    import asyncio

    from cuga import CugaAgent

    folder = str(skills_dir) if skills_dir is not None else None
    prepare_workspace_environment(folder)
    # Pass the RECORDED callables: rebuilding them here would drop the
    # call-recording wrappers, so a run that really executed tools would still
    # report zero tool calls.
    agent = CugaAgent(
        tools=build_tracked_tools(callables, app_names),
        **workspace_agent_kwargs(special_instructions, folder),
    )

    async def run() -> tuple[str, tuple]:
        await agent.initialize()
        # One execution only. track_tool_calls surfaces the SDK's own aggregated
        # list, which is independent evidence from our ledger -- not a
        # replacement for it.
        result = await agent.invoke(prompt, track_tool_calls=True)
        return str(result), tuple(getattr(result, "tool_calls", ()) or ())

    return asyncio.run(run())


def run_workspace_agent(
    callables: Mapping[str, Callable[..., str]],
    prompt: str,
    *,
    app_names: Mapping[str, str],
    skills_dir: Path | None = None,
    special_instructions: str | None = None,
    agent_factory: Callable[[dict, str], str] | None = None,
) -> WorkspaceAgentRun:
    """Run one workspace agent, returning failure as data rather than raising.

    A raised exception here would discard a whole round's evidence for one bad
    invocation, so every failure is classified and returned. ``tools_called`` is
    populated even on failure, because what executed already happened.

    ``agent_factory`` is the test seam: it receives the RECORDED callables dict
    and the prompt, and returns the agent's answer string. When ``None``, a real
    ``CugaAgent`` is constructed.

    A ``list_tools`` tool is injected automatically, its roster is inserted into
    the prompt, and a short execute directive is appended LAST. The ordering is
    load bearing: a prompt that ends on a tool listing or a schema was observed
    to produce a full narration -- including a claimed "submitted successfully"
    -- with an empty tool ledger, while the same content ending on an explicit
    "write and execute a fenced block" directive executed seven tools. The model
    acts on whatever it read last, so the roster goes in the middle and the
    directive goes at the end. ``list_tools`` never overrides a caller's own tool
    of that name.
    """
    with_listing = dict(callables)
    with_listing.setdefault("list_tools", build_list_tools(callables))
    recorded, names = recording_wrapper(with_listing)
    prompt = (
        f"{prompt}\n\nTOOLS AVAILABLE IN THIS SESSION "
        f"(exact signatures; every one is registered and callable):\n"
        f"{tool_roster(with_listing)}\n{_EXECUTE_NOW}"
    )
    instructions = (
        WORKSPACE_AGENT_TOOL_CONTRACT
        if special_instructions is None
        else f"{special_instructions}\n\n{WORKSPACE_AGENT_TOOL_CONTRACT}"
    )
    try:
        if agent_factory is not None:
            answer = agent_factory(recorded, prompt)
            sdk_calls: tuple = ()
        else:
            answer, sdk_calls = _run_real_agent(
                recorded, prompt, app_names, skills_dir, instructions
            )
    except Exception as exc:  # noqa: BLE001 - a failure is data, not a crash
        return WorkspaceAgentRun(
            tools_called=tuple(names), error=f"{type(exc).__name__}: {exc}"
        )
    return WorkspaceAgentRun(
        answer=str(answer),
        tools_called=tuple(names),
        sdk_tool_calls=sdk_calls,
    )
