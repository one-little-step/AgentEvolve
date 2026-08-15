"""Multi-parent editor request fields and outcome taxonomy (spec §7, §10)."""
from __future__ import annotations

import pytest

from agent_evolve.core.blame import BlameGraph, CausalAnalysis
from agent_evolve.core.contracts import CandidateWorkspace, EvolutionTask
from agent_evolve.core.editor import (
    EditorOutcome,
    EditorRequest,
    ParentContext,
)
from pathlib import Path


def _request(**kwargs) -> EditorRequest:
    defaults = dict(
        base_workspace=CandidateWorkspace("att-1", "v1", Path("."), "v0"),
        task=EvolutionTask(task_id="t", input_text="i"),
        analysis=CausalAnalysis(
            mechanism="m", severity=0.5, score=0.0,
            blame_graph=BlameGraph(nodes=()),
        ),
        issue_id="issue-1",
        write_set=("skills/a",),
    )
    defaults.update(kwargs)
    return EditorRequest(**defaults)


def test_parents_defaults_to_empty() -> None:
    assert _request().parents == ()


def test_creatable_prefix_defaults_to_disabled() -> None:
    assert _request().creatable_prefix == ""


def test_pool_created_count_defaults_to_zero() -> None:
    assert _request().pool_created_count == 0


def test_request_accepts_parent_context() -> None:
    primary = ParentContext(
        candidate_id="cand-1", version="v1", is_primary=True,
        score_summary={"task-a": 0.5},
    )
    donor = ParentContext(
        candidate_id="cand-2", version="v2", is_primary=False,
        score_summary={"task-a": 0.9},
    )
    request = _request(parents=(primary, donor))
    assert [p.candidate_id for p in request.parents] == ["cand-1", "cand-2"]


def test_request_rejects_more_than_one_primary_parent() -> None:
    a = ParentContext(candidate_id="c1", version="v1", is_primary=True, score_summary={})
    b = ParentContext(candidate_id="c2", version="v2", is_primary=True, score_summary={})
    with pytest.raises(ValueError, match="exactly one primary parent"):
        _request(parents=(a, b))


def test_request_rejects_parents_without_a_primary() -> None:
    a = ParentContext(candidate_id="c1", version="v1", is_primary=False, score_summary={})
    with pytest.raises(ValueError, match="exactly one primary parent"):
        _request(parents=(a,))


def test_current_artifacts_subset_guard_is_unchanged() -> None:
    """The existing write_set guard must survive the extension."""
    with pytest.raises(ValueError, match="outside write_set"):
        _request(current_artifacts={"skills/b": "x"})


def test_outcome_distinguishes_no_tool_call_from_no_op() -> None:
    assert EditorOutcome.NO_TOOL_CALL != EditorOutcome.NO_OP
    assert EditorOutcome.NO_TOOL_CALL.value == "no_tool_call"
    assert EditorOutcome.NO_OP.value == "no_op"
