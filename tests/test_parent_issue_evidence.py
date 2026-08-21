"""SV-10 — a parent's diagnosed faults must reach the editor.

**The register's stated fix direction is wrong on both halves, and measuring it
changed the fix.** ``docs/SEVERE-OPEN-ISSUES.md:1047`` prescribes: *"keep
``(task, mechanism)`` in ``ParentContext``, use ``weighted_score()``"*. Executing
the production path shows both instructions are dead ends:

* ``weighted_score()`` returns ``self.mean`` **exactly**. It is
  ``mean * severity * confidence``, and no caller anywhere in ``src/`` passes
  ``severity=`` or ``confidence=`` to ``ScoreProvenance``; all four production
  sites omit them, the class is frozen, and there is no ``replace``/``**kwargs``
  path. Switching the projection to it is a no-op that looks like a fix.
* **Keeping the mechanism key surfaces a placeholder.** Five of the six
  pool-write paths pass the *constant* ``self.mechanism_cluster_id``
  (``orchestrator.py`` lines 1416, 1440, 1562, 1871, 1950; field default ``"c0"``
  at :952). Only ``:345`` passes a clusterer-assigned id, and it sits in
  ``run_iteration``, which has zero production callers. So the score tensor's
  mechanism dimension is ``"mechanism-default"``/``"c0"`` for every candidate.

The real defect is **discard, not lossy projection**. ``run_attempt`` builds the
parent's *full* diagnosed fault set at ``orchestrator.py:2015``, keeps
``selected[0]`` at :2031, and passes that single issue plus one analysis into
``propose_edits``. Every other fault the analyzer diagnosed for that parent --
already paid for with real rollouts and real analyzer calls -- is thrown away
before the editor is asked to fix the parent.

``Issue`` already carries exactly what SV-10 asks for, per parent, with a
*genuinely written* severity (the diagnoser's ``CausalAnalysis.severity``, kind
(A), not the never-written ``ScoreProvenance.severity``, kind (B)):

    mechanism_cluster_id, severity, confidence, evidence_refs,
    writable_artifact_ids

So the fix routes evidence that already exists: **zero new rollouts, zero new
model calls.**

Two properties this must preserve, because breaking either would be worse than
the defect:

* **No prose crosses the boundary.** ``Issue`` carries no prose field at all --
  no ``mechanism_description``, no ``recurring_failure_mode``. The editor prompt
  is a persistence surface under ``AGENTS.md``, so carrying cluster ids and
  numbers rather than mechanism text satisfies the rule *by construction*, and
  the register's open question ("confirm whether ``recurring_failure_mode`` text
  is safe to persist at all") does not need answering to close this.
* **Same-task faults must not collide.** The old projection keyed a dict on
  ``task_id`` alone, so two mechanisms on one task silently overwrote each other
  and the editor saw whichever came last in iteration order.

Also folded in by user decision: ``run_attempt`` drew ``select_parent()``
**twice** per attempt (``build_issues()``:1490 and ``run_attempt()``:2033).
``select_parent`` consumes ``rng.random()``, so the draws are independent and the
parent whose faults were diagnosed could differ from the parent materialized and
edited. One draw per attempt, so the evidence and the workspace describe the same
entry. (The deeper sampling redesign is deferred.)

Assertions are behavioural: what the editor actually receives on the production
path, and how many distinct parents one attempt observes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_evolve.core.contracts import EvolutionCandidate
from agent_evolve.core.editor import EditorRequest, ParentContext
from agent_evolve.core.fake_editor import FakeEditor
from agent_evolve.core.issues import Issue

from test_phase_6_orchestrator import _CLUSTER, _record, _runner, _task  # type: ignore

_TASKS = (_task("task-a"), _task("task-b"))


class _RecordingEditor:
    """A ``FakeEditor`` that retains every request it was handed.

    SV-10 is a claim about what the editor *receives*, so the test must read the
    delivered request rather than a stand-in for it.
    """

    editor_model_id = "recording-editor"

    def __init__(self) -> None:
        self.requests: list[EditorRequest] = []
        self._inner = FakeEditor()

    def propose_edit(self, request: EditorRequest):
        self.requests.append(request)
        return self._inner.propose_edit(request)


def _issue(
    task_id: str,
    mechanism: str,
    *,
    severity: float,
    confidence: float = 0.8,
    artifacts: tuple[str, ...] = ("wisdom",),
    evidence: tuple[str, ...] = ("trace-1",),
) -> Issue:
    return Issue(
        issue_id=f"{task_id}:{mechanism}",
        task_id=task_id,
        mechanism_cluster_id=mechanism,
        severity=severity,
        confidence=confidence,
        entropy=0.0,
        coverage_need=0.0,
        pareto_relevance=0.0,
        raw_quality=0.0,
        embedding=(),
        writable_artifact_ids=artifacts,
        evidence_refs=evidence,
        lineage="base-v0",
        entropy_tier="skip",
    )


# --------------------------------------------------------------------------- #
# 1. ParentContext must be able to carry diagnosed faults at all
# --------------------------------------------------------------------------- #


def test_parent_context_carries_issue_evidence() -> None:
    """The delivery surface exists and retains every fault handed to it."""
    issues = (
        _issue("task-a", "task-a:c0", severity=0.9),
        _issue("task-b", "task-b:c1", severity=0.4),
    )
    parent = ParentContext(
        candidate_id="c1",
        version="v1",
        is_primary=True,
        score_summary={"task-a": 0.2},
        issues=issues,
    )
    assert len(parent.issues) == 2
    assert {i.mechanism_cluster_id for i in parent.issues} == {
        "task-a:c0",
        "task-b:c1",
    }


def test_parent_context_issues_defaults_empty() -> None:
    """A parent with no diagnosis is representable, and is not an error."""
    parent = ParentContext(candidate_id="c1", version="v1", is_primary=True)
    assert parent.issues == ()


def test_same_task_mechanisms_do_not_collide() -> None:
    """Two faults on ONE task must both survive.

    This is the specific loss in the old ``{t_id: cell.mean}`` projection: a dict
    keyed on ``task_id`` alone silently kept whichever mechanism came last.
    """
    issues = (
        _issue("task-a", "task-a:c0", severity=0.9),
        _issue("task-a", "task-a:c1", severity=0.3),
    )
    parent = ParentContext(
        candidate_id="c1", version="v1", is_primary=True, issues=issues
    )
    same_task = [i for i in parent.issues if i.task_id == "task-a"]
    assert len(same_task) == 2, "same-task mechanisms collapsed into one"
    assert {i.severity for i in same_task} == {0.9, 0.3}


# --------------------------------------------------------------------------- #
# 2. The production path must actually populate it
# --------------------------------------------------------------------------- #


def test_editor_receives_parent_faults_beyond_the_worked_issue() -> None:
    """The parent's OTHER diagnosed faults must reach the editor.

    The worked issue always reaches it via ``request.issue_id``. SV-10 is about
    the ones discarded at ``orchestrator.py:2031``: the editor is asked to
    improve a parent while being shown one of its several known faults.
    """
    editor = _RecordingEditor()
    runner = _runner(seed=0)
    runner.editor = editor

    outcome = runner.run_attempt(_TASKS)
    assert outcome is not None
    assert editor.requests, "editor was never called"

    request = editor.requests[0]
    primary = next(p for p in request.parents if p.is_primary)

    assert primary.issues, (
        "the primary parent reached the editor with no diagnosed faults; "
        "run_attempt discarded them at orchestrator.py:2031"
    )
    # Both tasks fail for a fresh base, so the parent has a fault on each.
    assert {i.task_id for i in primary.issues} == {"task-a", "task-b"}


def test_parent_faults_carry_severity_and_attribution() -> None:
    """Evidence must be actionable: which surface, how bad, on what basis."""
    editor = _RecordingEditor()
    runner = _runner(seed=0)
    runner.editor = editor
    runner.run_attempt(_TASKS)

    primary = next(p for p in editor.requests[0].parents if p.is_primary)
    assert primary.issues

    for issue in primary.issues:
        assert issue.writable_artifact_ids, "no surface named for this fault"
        assert issue.evidence_refs, "fault carries no trace-backed evidence"
        assert 0.0 <= issue.severity <= 1.0
        assert issue.mechanism_cluster_id


def test_parent_evidence_carries_no_prose() -> None:
    """No mechanism prose may cross into the editor request.

    The editor prompt is a persistence surface (``AGENTS.md``). ``Issue`` has no
    prose field, so this holds by construction -- asserted so a later widening of
    the payload to include ``mechanism_description`` fails loudly here.
    """
    editor = _RecordingEditor()
    runner = _runner(seed=0)
    runner.editor = editor
    runner.run_attempt(_TASKS)

    primary = next(p for p in editor.requests[0].parents if p.is_primary)
    for issue in primary.issues:
        for banned in ("mechanism_description", "recurring_failure_mode"):
            assert not hasattr(issue, banned), (
                f"{banned} crossed into the editor request; carry cluster ids "
                "and numbers, never mechanism prose"
            )


def test_donor_parents_also_carry_their_faults() -> None:
    """Donor evidence is what makes a transplant informed rather than blind.

    Without it the editor can see a donor scored better but not *why*, which is
    the difference between directed crossover and copying.
    """
    editor = _RecordingEditor()
    runner = _runner(seed=0)
    runner.editor = editor

    # Give a second candidate winning-cell evidence so select_parents can offer
    # it as a donor. The version stays "base-v0" because the pool keys on
    # candidate_id while the adapter must still be able to materialize it.
    runner.pool.add_candidate(
        EvolutionCandidate(
            candidate_id="cand-donor", version="base-v0", artifact_hashes={}
        )
    )
    _record(runner.pool, "cand-donor", "task-a", 0.9)

    runner.run_attempt(_TASKS)
    request = editor.requests[0]
    # Every parent offered must expose the same evidence shape; a donor with no
    # diagnosis yields an empty tuple, never a missing attribute.
    for parent in request.parents:
        assert isinstance(parent.issues, tuple)


# --------------------------------------------------------------------------- #
# 3. One parent draw per attempt
# --------------------------------------------------------------------------- #


def test_attempt_draws_one_parent(monkeypatch) -> None:
    """The parent diagnosed and the parent edited must be the same entry.

    ``select_parent`` consumes ``rng.random()``, so two calls in one attempt are
    independent draws. With a multi-candidate pool they can disagree, and the
    editor is then shown one parent's faults while writing into another's
    workspace.
    """
    runner = _runner(seed=0)

    # Populate the pool so parent_frequencies has mass on several candidates:
    # with only base present every draw trivially agrees and the test is vacuous.
    for i in range(3):
        cid = f"cand-{i}"
        runner.pool.add_candidate(
            EvolutionCandidate(
                candidate_id=cid, version="base-v0", artifact_hashes={}
            )
        )
        _record(runner.pool, cid, "task-a", 0.5 + i * 0.1)

    calls: list[str] = []
    original = type(runner).select_parent

    def _counting(self):
        entry = original(self)
        calls.append(entry.candidate_id)
        return entry

    monkeypatch.setattr(type(runner), "select_parent", _counting)
    runner.run_attempt(_TASKS)

    assert len(calls) == 1, (
        f"run_attempt drew a parent {len(calls)} times ({calls}); "
        "independent draws can diagnose parent A and edit parent B"
    )
