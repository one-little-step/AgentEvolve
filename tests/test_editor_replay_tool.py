"""W4-prime: the voluntary ``run_replay_experiment`` editor tool + its wiring.

Mirrors ``list_complementary_parents`` exactly (checkmarks 73/81): the tool is
registered in a cluster, unavailable-by-default until the composition root
attaches a provider factory, never raises into the agent. What is pinned here:

1. The tool is registered and reports ``unavailable`` at zero cost when no
   provider is attached.
2. The tool threads the editor's STAGED edits as the mutation and passes the
   caller's ``resume``/``gate_enabled`` through to the provider.
3. ``attach_replay_provider`` wires the per-request factory; building it is
   lazy, and each invocation reads live state.
4. A raising provider becomes an error string, never an exception.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.adapters.cuga_editor import attach_replay_provider  # noqa: E402
from agent_evolve.adapters.cuga_editor_tools import (  # noqa: E402
    TOOL_APP_NAMES,
    EditorToolContext,
    build_tool_callables,
)
from agent_evolve.core.blame import CausalAnalysis  # noqa: E402
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.editor import EditorRequest  # noqa: E402


class BlameGraphShim:
    def __init__(self) -> None:
        from agent_evolve.core.blame import BlameGraph, BlameNode

        self.graph = BlameGraph(
            nodes=(BlameNode(actor_id="planner", blame=1.0, artifacts=("skills/s",)),)
        )


def _analysis() -> CausalAnalysis:
    return CausalAnalysis(
        mechanism="the planner failed", severity=0.7, score=0.0,
        blame_graph=BlameGraphShim().graph,
    )


def _tool_ctx(provider=None) -> EditorToolContext:
    from agent_evolve.adapters.cuga_editor_evidence import EvidenceView
    from agent_evolve.adapters.cuga_editor_state import EditStagingArea
    from agent_evolve.core.contracts import (
        CandidateWorkspace,
        EvolutionTask,
        ExecutionTrace,
    )

    task = EvolutionTask(task_id="task-a", input_text="x", expected_contract={})
    analysis = _analysis()
    trace = ExecutionTrace(
        trace_id="tr", candidate_id="v1", task_id="task-a",
        events=(), final_output="", status="success",
    )
    request = EditorRequest(
        base_workspace=CandidateWorkspace("att", "v1", Path("."), "v0"),
        task=task,
        analysis=analysis,
        issue_id="i-1",
        write_set=("skills/s",),
        current_artifacts={},
        parents=(),
    )
    return EditorToolContext(
        staging=EditStagingArea(
            write_set=("skills/s",),
            creatable_prefixes=(),
            pool_created_count=0,
        ),
        evidence=EvidenceView(
            analysis=analysis, trace=trace, task=task, contamination_terms=()
        ),
        request=request,
        adapter=object(),
        memory=None,  # type: ignore[arg-type]
        replay_provider=provider,
    )


def test_tool_registered_in_a_cluster() -> None:
    assert TOOL_APP_NAMES.get("run_replay_experiment") is not None


def test_tool_without_provider_reports_unavailable_at_zero_cost() -> None:
    tools = build_tool_callables(_tool_ctx(None))
    decoded = json.loads(tools["run_replay_experiment"]())
    assert decoded["status"] == "unavailable"


def test_tool_threads_staged_edits_and_params_to_provider() -> None:
    seen: dict = {}

    def provider(**kwargs):
        seen.update(kwargs)
        return {"status": "ok", "taped_calls": 3}

    ctx = _tool_ctx(provider)
    ctx.staging.stage_replace("skills/s", "new content")

    tools = build_tool_callables(ctx)
    decoded = json.loads(tools["run_replay_experiment"](resume=2, gate_enabled=False))

    assert decoded["status"] == "ok"
    assert seen["resume"] == 2
    assert seen["gate_enabled"] is False
    assert seen["artifacts"] == {"skills/s": "new content"}


def test_tool_converts_raising_provider_to_error_string() -> None:
    def boom(**kwargs):
        raise RuntimeError("tape exploded")

    tools = build_tool_callables(_tool_ctx(boom))
    decoded = json.loads(tools["run_replay_experiment"]())
    assert decoded["status"] == "error"
    assert "tape exploded" in decoded["message"]


def test_attach_sets_the_factory_on_the_agent() -> None:
    from agent_evolve.adapters.cuga_editor import CugaEditorAgent

    agent = CugaEditorAgent.__new__(CugaEditorAgent)
    factory = lambda request: (lambda **kw: {"status": "ok"})  # noqa: E731

    attach_replay_provider(agent, factory)

    assert agent.replay_provider_factory is factory


def test_factory_is_lazy_and_reads_live_state_per_call() -> None:
    from agent_evolve.adapters.cuga_editor import CugaEditorAgent

    calls = []

    def facade_run(**kwargs):
        calls.append(kwargs)
        return {"status": "ok"}

    def factory(request):
        # Building the provider must NOT run anything.
        return lambda **kw: facade_run(task_id=request.task.task_id, **kw)

    agent = CugaEditorAgent.__new__(CugaEditorAgent)
    attach_replay_provider(agent, factory)

    assert calls == [], "building the factory must stay lazy"

    request = SimpleNamespace(task=SimpleNamespace(task_id="task-live-1"))
    provider = agent.replay_provider_factory(request)
    assert calls == [], "building the provider must stay lazy"

    provider(resume=1)
    assert calls == [{"task_id": "task-live-1", "resume": 1}]


# ---------------------------------------------------------------------- #
# Composition root: the wiring must be called where the editor is born
# ---------------------------------------------------------------------- #
def test_pipeline_composition_root_calls_wire_editor_replays() -> None:
    import ast
    import inspect

    from agent_evolve import pipeline as pipeline_module

    tree = ast.parse(inspect.getsource(pipeline_module.build_live_stack))

    wired = [
        call
        for node in ast.walk(tree)
        for call in [node]
        if isinstance(call, ast.Call)
        and getattr(call.func, "id", "") == "wire_editor_replays"
    ]
    assert wired, (
        "wire_editor_replays is never called in build_live_stack -- the replay "
        "tool would be permanently unavailable in production (W4 regression)"
    )
