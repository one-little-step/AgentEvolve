"""Editor tool bodies (spec §5).

build_tool_callables returns plain functions with no CUGA dependency, so the
entire authorization, evidence and capture surface is testable offline.

Every tool returns a JSON string: CUGA tools must return strings, and a
structured error string keeps one failing tool from aborting the agent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_evolve.adapters.cuga_adapter import CugaAdapter
from agent_evolve.adapters.cuga_editor_evidence import (
    EvidenceView,
    contamination_terms_from,
)
from agent_evolve.adapters.cuga_editor_state import EditStagingArea
from agent_evolve.adapters.cuga_editor_tools import (
    TOOL_APP_NAMES,
    EditorToolContext,
    build_editor_tools,
    build_tool_callables,
    submitted_plan,
)
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis
from agent_evolve.core.contracts import (
    CandidateWorkspace,
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)
from agent_evolve.core.editor import EditorRequest, ParentContext
from agent_evolve.core.memory import EditMemory
from agent_evolve.cuga_wrapper import CugaWrapper, InMemoryRuntime, RuntimeSettings

_SECRET = "token-a"


def _ctx(**overrides) -> EditorToolContext:
    adapter = CugaAdapter(wrapper=CugaWrapper(InMemoryRuntime(), RuntimeSettings(model="test-model")))
    adapter.register_candidate("v-primary", {"skills/retrieval": "primary body"})
    adapter.register_candidate("v-donor", {"skills/retrieval": "donor body"})

    task = EvolutionTask(
        task_id="task-a",
        input_text="produce the A capability",
        expected_contract={"expected_substring": _SECRET},
    )
    analysis = CausalAnalysis(
        mechanism="skill never loaded",
        severity=0.9,
        score=0.0,
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="call_model", blame=1.0, artifacts=("skills/retrieval",)),)
        ),
    )
    trace = ExecutionTrace(
        trace_id="t-1",
        candidate_id="cand-1",
        task_id="task-a",
        events=(
            TraceEvent(
                event_id="graph:1", kind="llm_call", actor_id="call_model",
                parent_event_id=None, payload={"messages_ref": "a" * 64},
            ),
            TraceEvent(
                event_id="graph:2", kind="tool_call", actor_id="sandbox",
                parent_event_id="graph:1",
                payload={"name": "run_command", "result": "exit 0"},
            ),
        ),
        final_output=f"answer {_SECRET}",
        status="completed",
    )
    request = EditorRequest(
        base_workspace=CandidateWorkspace("att-1", "v-primary", Path("."), "v0"),
        task=task,
        analysis=analysis,
        issue_id="issue-1",
        write_set=("skills/retrieval",),
        current_artifacts={"skills/retrieval": "primary body"},
        creatable_prefixes=("skills/generated-",),
        parents=(
            ParentContext("cand-1", "v-primary", True, {"task-a": 0.0}),
            ParentContext("cand-2", "v-donor", False, {"task-a": 0.9}),
        ),
    )
    ctx = EditorToolContext(
        staging=EditStagingArea(
            write_set=request.write_set,
            creatable_prefixes=request.creatable_prefixes,
        ),
        evidence=EvidenceView(
            analysis=analysis, trace=trace, task=task,
            contamination_terms=contamination_terms_from(task),
        ),
        request=request,
        adapter=adapter,
        memory=EditMemory(),
    )
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


def _tools(ctx: EditorToolContext | None = None):
    ctx = ctx if ctx is not None else _ctx()
    return ctx, build_tool_callables(ctx)


# ------------------------------------------------------------------ #
# shape
# ------------------------------------------------------------------ #
def test_every_tool_returns_a_json_string() -> None:
    _, tools = _tools()
    for name, fn in tools.items():
        if name in {"get_mechanism", "list_blamed_actors", "get_task_input",
                    "list_trace_actors", "list_artifacts", "list_staged",
                    "list_parents"}:
            out = fn()
            assert isinstance(out, str), name
            json.loads(out)


def test_every_tool_has_a_cluster_assignment() -> None:
    _, tools = _tools()
    assert set(tools) <= set(TOOL_APP_NAMES)
    for name in tools:
        assert TOOL_APP_NAMES[name]


def test_expected_cluster_names() -> None:
    assert set(TOOL_APP_NAMES.values()) == {
        "evidence", "harness", "history", "parents", "submit", "rollout",
        "replay",
    }


# ------------------------------------------------------------------ #
# rollout cluster
# ------------------------------------------------------------------ #
def test_list_rollout_tools_reports_every_tool_the_rollout_agent_owns() -> None:
    """A capability absent from the prompt is a capability the agent will not
    use, and the editor had no way at all to learn the rollout tool inventory.

    Expected names are derived from the wrapper module rather than duplicated,
    so a tool added or removed there cannot silently drift out of this view.
    """
    from agent_evolve.cuga_wrapper import tools as rollout_tools

    _, tools = _tools()
    payload = json.loads(tools["list_rollout_tools"]())

    expected = [func.__name__ for func in rollout_tools._RAW_TOOLS]
    assert [entry["name"] for entry in payload["tools"]] == sorted(expected)
    assert payload["count"] == len(expected)


def test_list_rollout_tools_carries_a_signature_and_purpose_per_tool() -> None:
    """A bare name does not tell the editor how to invoke a tool; the signature
    and the tool's own one-line docstring do."""
    _, tools = _tools()
    payload = json.loads(tools["list_rollout_tools"]())

    calculator = next(e for e in payload["tools"] if e["name"] == "calculator")
    assert calculator["signature"] == "calculator(expression: str) -> str"
    assert calculator["purpose"] == (
        "Evaluate a mathematical expression safely and return its value."
    )


def test_list_rollout_tools_needs_no_cuga_sdk() -> None:
    """The inventory is read from the plain tool functions, not from
    ``build_tools``, which imports the SDK and would make the whole offline
    editor test surface depend on a live CUGA install."""
    _, tools = _tools()

    payload = json.loads(tools["list_rollout_tools"]())

    assert payload["tools"]


# ------------------------------------------------------------------ #
# evidence cluster
# ------------------------------------------------------------------ #
def test_get_mechanism_returns_mechanism_and_severity() -> None:
    _, tools = _tools()
    payload = json.loads(tools["get_mechanism"]())
    assert payload["mechanism"] == "skill never loaded"


def test_get_task_input_returns_input_text() -> None:
    _, tools = _tools()
    assert json.loads(tools["get_task_input"]())["input_text"] == (
        "produce the A capability"
    )


def test_read_trace_events_strips_blob_refs() -> None:
    _, tools = _tools()
    events = json.loads(tools["read_trace_events"]())
    llm = next(e for e in events if e["kind"] == "llm_call")
    assert "messages_ref" not in llm["payload"]


def test_no_tool_output_contains_the_expected_value() -> None:
    """Leak audit across every readable tool."""
    ctx, tools = _tools()
    blob = ""
    for name in ("get_mechanism", "list_blamed_actors", "get_task_input",
                 "list_trace_actors", "list_artifacts", "list_parents"):
        blob += tools[name]()
    blob += tools["read_trace_events"]()
    blob += tools["read_artifact"]("skills/retrieval")
    blob += tools["read_parent_artifact"]("cand-2", "skills/retrieval")
    assert _SECRET not in blob


def test_no_tool_output_contains_the_final_output() -> None:
    _, tools = _tools()
    blob = tools["read_trace_events"]() + tools["get_mechanism"]()
    assert "answer " not in blob


# ------------------------------------------------------------------ #
# harness cluster
# ------------------------------------------------------------------ #
def test_read_artifact_returns_current_content() -> None:
    _, tools = _tools()
    assert json.loads(tools["read_artifact"]("skills/retrieval"))["content"] == (
        "primary body"
    )


def test_read_artifact_rejects_unknown_id() -> None:
    _, tools = _tools()
    payload = json.loads(tools["read_artifact"]("skills/absent"))
    assert payload["status"] == "error"


def test_stage_replace_accepts_authorized_write() -> None:
    ctx, tools = _tools()
    payload = json.loads(tools["stage_replace"]("skills/retrieval", "new"))
    assert payload["accepted"] is True
    assert ctx.staging.staged_ids() == ("skills/retrieval",)


def test_stage_replace_returns_rejection_without_raising() -> None:
    _, tools = _tools()
    payload = json.loads(tools["stage_replace"]("policies/x", "new"))
    assert payload["accepted"] is False
    assert "authorized write set" in payload["reason"]


def test_stage_create_enforces_namespace() -> None:
    _, tools = _tools()
    payload = json.loads(tools["stage_create"]("generated/x", "body"))
    assert payload["accepted"] is False


def test_stage_create_accepts_namespaced_id() -> None:
    _, tools = _tools()
    payload = json.loads(tools["stage_create"]("skills/generated-x", "body"))
    assert payload["accepted"] is True


def test_list_staged_reflects_staging() -> None:
    _, tools = _tools()
    tools["stage_replace"]("skills/retrieval", "new")
    assert json.loads(tools["list_staged"]())["staged"] == ["skills/retrieval"]


def test_unstage_removes_an_edit() -> None:
    _, tools = _tools()
    tools["stage_replace"]("skills/retrieval", "new")
    tools["unstage"]("skills/retrieval")
    assert json.loads(tools["list_staged"]())["staged"] == []


# ------------------------------------------------------------------ #
# parents cluster
# ------------------------------------------------------------------ #
def test_list_parents_marks_the_primary() -> None:
    _, tools = _tools()
    parents = json.loads(tools["list_parents"]())
    primary = [p for p in parents if p["is_primary"]]
    assert [p["candidate_id"] for p in primary] == ["cand-1"]


def test_read_parent_artifact_returns_donor_content() -> None:
    _, tools = _tools()
    payload = json.loads(tools["read_parent_artifact"]("cand-2", "skills/retrieval"))
    assert payload["content"] == "donor body"


def test_read_parent_artifact_records_provenance() -> None:
    ctx, tools = _tools()
    tools["read_parent_artifact"]("cand-2", "skills/retrieval")
    assert ctx.staging.parents_read() == ("cand-2",)


def test_read_parent_artifact_rejects_unknown_parent() -> None:
    ctx, tools = _tools()
    payload = json.loads(tools["read_parent_artifact"]("cand-99", "skills/retrieval"))
    assert payload["status"] == "error"
    assert ctx.staging.parents_read() == ()


# ------------------------------------------------------------------ #
# submit cluster
# ------------------------------------------------------------------ #
def test_submitted_plan_is_none_before_finalizing() -> None:
    ctx, _ = _tools()
    assert submitted_plan(ctx) is None


def test_submit_captures_the_staged_plan() -> None:
    ctx, tools = _tools()
    tools["stage_replace"]("skills/retrieval", "new")
    payload = json.loads(tools["submit_edit_plan"]("because evidence", "some risk", "fix"))
    assert payload["accepted"] is True
    plan = submitted_plan(ctx)
    assert plan is not None
    assert [e.artifact_id for e in plan["edits"]] == ["skills/retrieval"]
    assert plan["rationale"] == "because evidence"


def test_submit_with_nothing_staged_is_an_explicit_decline() -> None:
    ctx, tools = _tools()
    payload = json.loads(tools["submit_edit_plan"]("no change warranted", "", ""))
    assert payload["accepted"] is True
    plan = submitted_plan(ctx)
    assert plan is not None
    assert plan["edits"] == ()
    assert plan["declined"] is True


def test_submit_requires_a_rationale_when_declining() -> None:
    ctx, tools = _tools()
    payload = json.loads(tools["submit_edit_plan"]("", "", ""))
    assert payload["accepted"] is False
    assert submitted_plan(ctx) is None


def test_submit_is_idempotent_last_call_wins() -> None:
    ctx, tools = _tools()
    tools["stage_replace"]("skills/retrieval", "first")
    tools["submit_edit_plan"]("first rationale", "", "")
    tools["stage_replace"]("skills/retrieval", "second")
    tools["submit_edit_plan"]("second rationale", "", "")
    plan = submitted_plan(ctx)
    assert plan["rationale"] == "second rationale"
    assert plan["edits"][0].payload["content"] == "second"


def test_module_imports_without_cuga_installed() -> None:
    """build_tool_callables must not require the SDK."""
    import agent_evolve.adapters.cuga_editor_tools as mod

    assert not hasattr(mod, "tracked_tool")


def test_every_tool_callable_has_a_docstring() -> None:
    """LangChain's @tool raises without one, and the model reads it as the
    tool's description when deciding whether to call it.

    This failed only at live-run time ("Function must have a docstring if
    description not provided"), after every offline test passed, so it is
    pinned here where it costs nothing to catch.
    """
    ctx = _ctx()
    callables = build_tool_callables(ctx)
    missing = [name for name, fn in callables.items() if not fn.__doc__]
    assert missing == []
    # Guards the loop above against passing vacuously if the tool dict ever
    # shrinks: every declared cluster member must actually be present.
    assert set(callables) == set(TOOL_APP_NAMES)


def test_submit_edit_plan_docstring_states_the_finalize_requirement() -> None:
    """The mandatory-finalize contract must reach the model through the tool
    description, not only through the system instructions."""
    doc = build_tool_callables(_ctx())["submit_edit_plan"].__doc__ or ""
    lowered = doc.lower()
    assert "once" in lowered
    assert "declin" in lowered


def test_build_editor_tools_uses_supplied_callables_not_fresh_ones() -> None:
    """The real-agent path must not discard the caller's instrumentation.

    _run_cuga_agent wraps every tool body in a call recorder, then hands the
    wrapped dict to build_editor_tools. An implementation that rebuilds from
    ctx instead reports zero tool calls on a run where tools really executed --
    observed live, where get_mechanism ran but the ledger stayed empty.
    """
    ctx = _ctx()
    calls: list[str] = []

    def recorded() -> str:
        """Recording stand-in for a tool body."""
        calls.append("get_mechanism")
        return "{}"

    supplied = {"get_mechanism": recorded}
    built = build_editor_tools(ctx, supplied)

    assert len(built) == 1
    built[0].invoke({})
    assert calls == ["get_mechanism"], "supplied callable was not the one invoked"
