"""CugaEditorAgent.propose_edit with a stubbed agent (spec §4, §7, §10).

The stub invokes a scripted tool sequence, so the whole classification and
capture path is testable without the SDK or a network call.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_evolve.adapters.cuga_adapter import CugaAdapter
from agent_evolve.adapters.cuga_editor import CugaEditorAgent, EditorDeclined
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis
from agent_evolve.core.contracts import (
    CandidateWorkspace,
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)
from agent_evolve.core.editor import (
    EditorOutcome,
    EditorRequest,
    ParentContext,
    repair_once_then_classify,
)
from agent_evolve.core.memory import EditMemory
from agent_evolve.cuga_wrapper import CugaWrapper, InMemoryRuntime, RuntimeSettings


class ScriptedAgent:
    """Calls a fixed sequence of tools, then returns prose.

    Mirrors the real contract: the prose answer is irrelevant and must be
    ignored by propose_edit.
    """

    def __init__(self, script, answer="I have finished my analysis."):
        self.script = script
        self.answer = answer
        self.called: list[str] = []

    def run(self, tools: dict, prompt: str) -> str:
        for name, args in self.script:
            self.called.append(name)
            tools[name](*args)
        return self.answer


def _adapter() -> CugaAdapter:
    adapter = CugaAdapter(wrapper=CugaWrapper(InMemoryRuntime(), RuntimeSettings(model="test-model")))
    adapter.register_candidate("v-primary", {"skills/retrieval": "primary body"})
    adapter.register_candidate("v-donor", {"skills/retrieval": "donor body"})
    return adapter


def _request() -> EditorRequest:
    task = EvolutionTask(task_id="task-a", input_text="do A",
                         expected_contract={"expected_substring": "token-a"})
    analysis = CausalAnalysis(
        mechanism="skill never loaded", severity=0.9, score=0.0,
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="call_model", blame=1.0,
                             artifacts=("skills/retrieval",)),)
        ),
    )
    return EditorRequest(
        base_workspace=CandidateWorkspace("att-1", "v-primary", Path("."), "v0"),
        task=task,
        analysis=analysis,
        issue_id="issue-1",
        write_set=("skills/retrieval",),
        current_artifacts={"skills/retrieval": "primary body"},
        creatable_prefix="skills/generated-",
        parents=(
            ParentContext("cand-1", "v-primary", True, {"task-a": 0.0}),
            ParentContext("cand-2", "v-donor", False, {"task-a": 0.9}),
        ),
    )


def _editor(script, answer="done") -> tuple[CugaEditorAgent, ScriptedAgent]:
    stub = ScriptedAgent(script, answer)
    editor = CugaEditorAgent(
        adapter=_adapter(),
        memory=EditMemory(),
        agent_factory=lambda tools, prompt: stub.run(tools, prompt),
    )
    return editor, stub


_TRACE = ExecutionTrace(
    trace_id="t-1", candidate_id="cand-1", task_id="task-a",
    events=(TraceEvent(event_id="graph:1", kind="llm_call",
                       actor_id="call_model", parent_event_id=None, payload={}),),
    final_output="", status="completed",
)


# ------------------------------------------------------------------ #
# happy path
# ------------------------------------------------------------------ #
def test_propose_edit_returns_the_staged_plan() -> None:
    editor, _ = _editor([
        ("get_mechanism", ()),
        ("read_artifact", ("skills/retrieval",)),
        ("stage_replace", ("skills/retrieval", "improved body")),
        ("submit_edit_plan", ("addresses the mechanism",)),
    ])
    response = editor.propose_edit(_request())
    assert [e.artifact_id for e in response.edits] == ["skills/retrieval"]
    assert response.edits[0].payload["content"] == "improved body"
    assert response.rationale == "addresses the mechanism"
    assert editor.last_outcome is EditorOutcome.VALID


def test_prose_answer_is_ignored() -> None:
    editor, _ = _editor(
        [("stage_replace", ("skills/retrieval", "x")),
         ("submit_edit_plan", ("r",))],
        answer='{"edits": [{"artifact_id": "skills/HACKED"}]}',
    )
    response = editor.propose_edit(_request())
    assert [e.artifact_id for e in response.edits] == ["skills/retrieval"]


def test_editor_model_id_is_reported() -> None:
    editor, _ = _editor([("stage_replace", ("skills/retrieval", "x")),
                         ("submit_edit_plan", ("r",))])
    response = editor.propose_edit(_request())
    assert response.editor_model_id == "cuga-editor-agent"


def test_tools_called_are_recorded() -> None:
    editor, _ = _editor([("get_mechanism", ()),
                         ("stage_replace", ("skills/retrieval", "x")),
                         ("submit_edit_plan", ("r",))])
    editor.propose_edit(_request())
    assert "get_mechanism" in editor.last_tools_called


# ------------------------------------------------------------------ #
# provenance
# ------------------------------------------------------------------ #
def test_parents_read_are_recorded_for_provenance() -> None:
    editor, _ = _editor([
        ("read_parent_artifact", ("cand-2", "skills/retrieval")),
        ("stage_replace", ("skills/retrieval", "donor body")),
        ("submit_edit_plan", ("transplanted from donor",)),
    ])
    editor.propose_edit(_request())
    assert editor.last_parents_read == ("cand-2",)


def test_unread_donors_are_not_recorded() -> None:
    editor, _ = _editor([("stage_replace", ("skills/retrieval", "x")),
                         ("submit_edit_plan", ("r",))])
    editor.propose_edit(_request())
    assert editor.last_parents_read == ()


# ------------------------------------------------------------------ #
# outcome taxonomy (spec §10)
# ------------------------------------------------------------------ #
def test_never_calling_submit_is_no_tool_call() -> None:
    editor, _ = _editor([("get_mechanism", ())])
    with pytest.raises(EditorDeclined) as excinfo:
        editor.propose_edit(_request())
    assert excinfo.value.outcome is EditorOutcome.NO_TOOL_CALL
    assert editor.last_outcome is EditorOutcome.NO_TOOL_CALL


def test_staging_without_finalizing_is_no_tool_call() -> None:
    """Staged-but-unfinalized work is discarded, not silently applied."""
    editor, _ = _editor([("stage_replace", ("skills/retrieval", "x"))])
    with pytest.raises(EditorDeclined) as excinfo:
        editor.propose_edit(_request())
    assert excinfo.value.outcome is EditorOutcome.NO_TOOL_CALL


def test_explicit_decline_is_no_op_not_no_tool_call() -> None:
    editor, _ = _editor([("submit_edit_plan", ("evidence does not justify a change",))])
    with pytest.raises(EditorDeclined) as excinfo:
        editor.propose_edit(_request())
    assert excinfo.value.outcome is EditorOutcome.NO_OP
    assert editor.last_outcome is EditorOutcome.NO_OP


def test_agent_error_is_unavailable() -> None:
    def exploding_factory(tools, prompt):
        raise RuntimeError("CUGA execution failed")

    editor = CugaEditorAgent(
        adapter=_adapter(), memory=EditMemory(),
        agent_factory=exploding_factory,
    )
    with pytest.raises(EditorDeclined) as excinfo:
        editor.propose_edit(_request())
    assert excinfo.value.outcome is EditorOutcome.UNAVAILABLE


# ------------------------------------------------------------------ #
# integration with the core repair protocol
# ------------------------------------------------------------------ #
def test_repair_protocol_treats_a_decline_as_a_non_promotion() -> None:
    editor, _ = _editor([("submit_edit_plan", ("declining",))])
    result = repair_once_then_classify(editor, _request())
    assert result.status == "malformed"
    assert result.response is None
    # The distinct outcome survives for the caller to record.
    assert editor.last_outcome is EditorOutcome.NO_OP


def test_repair_protocol_passes_a_valid_plan_through_unchanged() -> None:
    editor, _ = _editor([("stage_replace", ("skills/retrieval", "x")),
                         ("submit_edit_plan", ("r",))])
    result = repair_once_then_classify(editor, _request())
    assert result.status == "valid"
    assert result.correction_requests == 0


# ------------------------------------------------------------------ #
# isolation (spec §13)
# ------------------------------------------------------------------ #
def test_editor_agent_construction_detaches_tracing() -> None:
    """The editor's own LLM calls must never enter a rollout trace."""
    from agent_evolve.adapters.cuga_editor import editor_agent_kwargs

    kwargs = editor_agent_kwargs()
    assert kwargs["callbacks"] == []
    assert kwargs["cuga_folder"] is None


def test_editor_agent_construction_binds_no_workspace(monkeypatch) -> None:
    import os

    from agent_evolve.adapters.cuga_editor import prepare_editor_environment

    monkeypatch.setenv("CUGA_FOLDER", "/some/rollout/workspace")
    prepare_editor_environment()
    assert "CUGA_FOLDER" not in os.environ


def test_recording_wrapper_preserves_docstring_and_signature() -> None:
    """The real-agent path feeds wrapped bodies to LangChain's @tool.

    @tool raises without a docstring and derives its args schema from the
    signature, which inspect.signature can only recover through __wrapped__.
    A bare *args wrapper broke both -- twice, live, after the offline suite
    passed -- because agent_factory tests never touch the decorator.
    """
    import inspect

    def stage_replace(artifact_id: str, content: str) -> str:
        """Stage replacement content for an existing artifact."""
        return "{}"

    wrapped, _names = CugaEditorAgent._recording_wrapper(
        {"stage_replace": stage_replace}
    )
    recorded = wrapped["stage_replace"]

    assert recorded.__doc__, "docstring lost: LangChain @tool would reject this"
    params = list(inspect.signature(recorded).parameters)
    assert params == ["artifact_id", "content"], (
        "signature lost: LangChain would build an empty args schema"
    )
