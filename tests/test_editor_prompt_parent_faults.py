"""SV-10 — the editor must be *told* about its parents' diagnosed faults.

Delivering evidence on `EditorRequest` is necessary and not sufficient. A live
CUGA editor decides which tools to call from the prompt text, and
``EDITOR_INSTRUCTIONS`` gates the parent tools explicitly:

    "When the evidence reports that donor parents are available, call
     list_parents and read_parent_artifact ... BEFORE you decide to refine."

``_parent_summary`` is the sole parent-facing prompt text, and it previously
rendered parents as **scores only** -- ``c-donor (scores {'task-a': 0.9})``. So a
parent's mechanism ids, severities and target surfaces reached the tool layer but
never the prompt, and nothing told the model the evidence existed.

That failure mode is not hypothetical here. ``_parent_summary``'s own docstring
records it:

    "two live runs with a donor whose artifact already contained the missing
     capability never called list_parents, because nothing in the prompt said a
     donor existed."

The same reasoning applies to faults: a tool that renders `issues` is dead weight
if the prompt never mentions them. These tests assert on the **prompt string the
model receives**, which is built deterministically before any network call -- so
this is offline-provable and needs no proxy capture.

What is still *not* proven offline, and is recorded as such: whether a live model,
once told, actually calls `list_parents` and changes its edit. That needs the
interception proxy and a captured request body.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_evolve.adapters.cuga_editor import _parent_summary
from agent_evolve.adapters.cuga_editor_skills import EDITOR_INSTRUCTIONS
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis
from agent_evolve.core.contracts import CandidateWorkspace, EvolutionTask
from agent_evolve.core.editor import EditorRequest, ParentContext
from agent_evolve.core.issues import Issue


def _issue(
    task_id: str,
    mechanism: str,
    *,
    severity: float,
    artifacts: tuple[str, ...] = ("skills/retrieval",),
) -> Issue:
    return Issue(
        issue_id=f"{task_id}:{mechanism}",
        task_id=task_id,
        mechanism_cluster_id=mechanism,
        severity=severity,
        confidence=0.8,
        entropy=0.0,
        coverage_need=0.0,
        pareto_relevance=0.0,
        raw_quality=0.0,
        embedding=(),
        writable_artifact_ids=artifacts,
        evidence_refs=("trace-1",),
        lineage="base-v0",
        entropy_tier="skip",
    )


def _request(*parents: ParentContext) -> EditorRequest:
    """A real ``EditorRequest``, so dataclass validation is exercised too."""
    return EditorRequest(
        base_workspace=CandidateWorkspace("att-1", "v1", Path("."), "v0"),
        task=EvolutionTask(
            task_id="task-a",
            input_text="do A",
            expected_contract={"expected_substring": "token-a"},
        ),
        analysis=CausalAnalysis(
            mechanism="skill never loaded",
            severity=0.9,
            score=0.0,
            blame_graph=BlameGraph(
                nodes=(
                    BlameNode(
                        actor_id="call_model",
                        blame=1.0,
                        artifacts=("skills/retrieval",),
                    ),
                )
            ),
        ),
        issue_id="issue-1",
        write_set=("skills/retrieval",),
        current_artifacts={"skills/retrieval": "primary body"},
        parents=parents,
    )


def _primary(*issues: Issue) -> ParentContext:
    return ParentContext(
        candidate_id="c-primary",
        version="v1",
        is_primary=True,
        score_summary={"task-a": 0.2},
        issues=issues,
    )


def _donor(*issues: Issue) -> ParentContext:
    return ParentContext(
        candidate_id="c-donor",
        version="v2",
        is_primary=False,
        score_summary={"task-a": 0.9},
        issues=issues,
    )


# --------------------------------------------------------------------------- #
# The primary's own weaknesses must appear in the prompt
# --------------------------------------------------------------------------- #


def test_prompt_states_the_primary_diagnosed_faults() -> None:
    """The editor is asked to improve this parent; it must be told what is wrong."""
    text = _parent_summary(
        _request(_primary(_issue("task-a", "task-a:c0", severity=0.9)))
    )
    assert "task-a:c0" in text, "the primary's mechanism id never reaches the prompt"
    assert "0.9" in text, "the primary's severity never reaches the prompt"
    assert "skills/retrieval" in text, "the target surface never reaches the prompt"


def test_prompt_ranks_primary_faults_by_severity() -> None:
    """Severity is an attention signal, so the worst fault must lead.

    The editor gets one attempt; presenting a 0.2 fault before a 0.9 one spends
    that attempt on the lesser problem.
    """
    text = _parent_summary(
        _request(
            _primary(
                _issue("task-a", "task-a:c0", severity=0.2),
                _issue("task-b", "task-b:c1", severity=0.9),
            )
        )
    )
    assert text.index("task-b:c1") < text.index("task-a:c0"), (
        "faults are not ordered by severity; the editor sees the milder one first"
    )


def test_prompt_survives_a_parent_with_no_diagnosis() -> None:
    """No diagnosis is a legitimate state and must not fabricate evidence."""
    text = _parent_summary(_request(_primary()))
    assert text
    assert "task-a:c0" not in text


# --------------------------------------------------------------------------- #
# Donor faults, so a transplant is informed rather than blind
# --------------------------------------------------------------------------- #


def test_prompt_states_donor_faults_alongside_scores() -> None:
    """A donor's score says it is better; its faults say where it is not."""
    text = _parent_summary(
        _request(
            _primary(_issue("task-a", "task-a:c0", severity=0.9)),
            _donor(_issue("task-b", "task-b:c1", severity=0.3)),
        )
    )
    assert "c-donor" in text
    assert "task-b:c1" in text, "the donor's diagnosed fault never reaches the prompt"


def test_donor_availability_signal_is_preserved() -> None:
    """Do not regress the existing fix that made crossover discoverable.

    The docstring on ``_parent_summary`` records two live runs where the editor
    never called ``list_parents`` because the prompt did not announce a donor.
    Adding fault text must not remove that announcement.
    """
    text = _parent_summary(
        _request(_primary(), _donor(_issue("task-b", "task-b:c1", severity=0.3)))
    )
    assert "donor" in text.lower()
    assert "c-donor" in text

    none_text = _parent_summary(_request(_primary()))
    assert "no donors" in none_text.lower()


# --------------------------------------------------------------------------- #
# The instructions must tell the model the evidence is there
# --------------------------------------------------------------------------- #


def test_instructions_direct_the_editor_to_the_fault_evidence() -> None:
    """The prompt gates tool use on the evidence text; it must name faults.

    ``EDITOR_INSTRUCTIONS`` already says to call ``list_parents`` "when the
    evidence reports that donor parents are available". Fault evidence needs the
    same explicit hook, or the model has no reason to read it.
    """
    lowered = EDITOR_INSTRUCTIONS.lower()
    assert "list_parents" in lowered
    assert any(
        phrase in lowered
        for phrase in ("diagnosed fault", "known weakness", "parent's faults")
    ), "the instructions never mention parent fault evidence"


# --------------------------------------------------------------------------- #
# The persistence rule still holds at the prompt boundary
# --------------------------------------------------------------------------- #


def test_prompt_carries_no_mechanism_prose() -> None:
    """The prompt is a persistence surface; only ids and numbers may cross.

    ``Issue`` has no prose field, so this holds by construction. Asserted so a
    later change that starts interpolating ``mechanism_description`` into the
    prompt fails here rather than silently leaking task content.
    """
    text = _parent_summary(
        _request(_primary(_issue("task-a", "task-a:c0", severity=0.9)))
    )
    for banned in ("expected_substring", "token-a", "expected answer"):
        assert banned not in text
