"""Tests for the editor protocol, validation, and acceptance rules."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_evolve.core.blame import empty_analysis
from agent_evolve.core.contracts import (
    ArtifactEdit,
    CandidateWorkspace,
    EditPlan,
    EvolutionTask,
    ExpectedEffect,
)
from agent_evolve.core.editor import (
    AcceptanceDecision,
    EditorRequest,
    EditorResponse,
    FocusedValidationReport,
    ProtectedFloor,
    ValidationKind,
    ValidationPlanner,
    ValidationProbe,
    ValidationResult,
    build_attempt,
    decide_acceptance,
    lineage_of,
    record_attempt,
    repair_once_then_classify,
    validate_editor_plan,
)
from agent_evolve.core.errors import WriteAuthorizationError
from agent_evolve.core.memory import (
    AttemptStatus,
    EditMemory,
)
from pathlib import Path


def _workspace(attempt: str = "att-1", parent: str = "base-v0") -> CandidateWorkspace:
    return CandidateWorkspace(
        attempt_id=attempt, version=f"{parent}+{attempt}", path=Path("."), parent_version=parent
    )


def _task(tid: str = "t1") -> EvolutionTask:
    return EvolutionTask(task_id=tid, input_text="x")


def _edit(aid: str = "skills/r1") -> ArtifactEdit:
    return ArtifactEdit(artifact_id=aid, operation="replace", payload={"content": "x"})


# ---------------------------------------------------------------------- #
# EditorRequest
# ---------------------------------------------------------------------- #
def test_request_rejects_empty_write_set():
    with pytest.raises(ValueError):
        EditorRequest(
            base_workspace=_workspace(),
            task=_task(),
            analysis=__import__("agent_evolve.core.blame", fromlist=["empty_analysis"]).empty_analysis(),
            issue_id="i1",
            write_set=(),
        )


def test_request_rejects_current_artifacts_outside_write_set():
    with pytest.raises(ValueError):
        EditorRequest(
            base_workspace=_workspace(),
            task=_task(),
            analysis=__import__("agent_evolve.core.blame", fromlist=["empty_analysis"]).empty_analysis(),
            issue_id="i1",
            write_set=("skills/a",),
            current_artifacts={"skills/b": "x"},
        )


def test_request_rejects_empty_issue_id():
    with pytest.raises(ValueError):
        EditorRequest(
            base_workspace=_workspace(),
            task=_task(),
            analysis=__import__("agent_evolve.core.blame", fromlist=["empty_analysis"]).empty_analysis(),
            issue_id="",
            write_set=("skills/a",),
        )


# ---------------------------------------------------------------------- #
# EditorResponse
# ---------------------------------------------------------------------- #
def test_response_rejects_empty_edits():
    with pytest.raises(ValueError):
        EditorResponse(
            rationale="r", edits=(), reads={}, writes={}, risks={}, expected_effects={}
        )


def test_artifact_edit_rejects_empty_artifact_id_at_construction() -> None:
    with pytest.raises(ValidationError, match="artifact_id"):
        ArtifactEdit(artifact_id="", operation="replace", payload={})


def test_response_rejects_writes_with_denied_key():
    with pytest.raises(ValueError):
        EditorResponse(
            rationale="r",
            edits=(_edit(),),
            reads={},
            writes={"expected_answer": "42"},
            risks={},
            expected_effects={},
        )


# ---------------------------------------------------------------------- #
# ValidationResult
# ---------------------------------------------------------------------- #
def test_validation_result_rejects_invalid_score():
    with pytest.raises(ValueError):
        ValidationResult(
            kind=ValidationKind.ORIGIN, task_id="t", score=1.5, trace_id="x", passed=True
        )


# ---------------------------------------------------------------------- #
# FocusedValidationReport
# ---------------------------------------------------------------------- #
def _vr(kind: ValidationKind, task_id: str, score: float, passed: bool | None = None) -> ValidationResult:
    if passed is None:
        passed = score >= 0.5
    return ValidationResult(kind=kind, task_id=task_id, score=score, trace_id="t", passed=passed)


def test_report_origin_passed_default_true_when_empty():
    r = FocusedValidationReport(origin=(), worked=(), regression=())
    assert r.origin_passed is True
    assert r.worked_passed is True
    assert r.regression_violated is False


def test_report_origin_passed_false_when_any_failed():
    r = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "t", 0.9), _vr(ValidationKind.ORIGIN, "t2", 0.1, passed=False)),
        worked=(),
        regression=(),
    )
    assert r.origin_passed is False


def test_report_regression_violated_when_any_failed():
    r = FocusedValidationReport(
        origin=(),
        worked=(),
        regression=(_vr(ValidationKind.REGRESSION, "t", 0.1, passed=False),),
    )
    assert r.regression_violated is True


def test_weighted_net_gain_default_weights():
    r = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "t", 0.8),),
        worked=(_vr(ValidationKind.WORKED, "t", 0.6),),
        regression=(_vr(ValidationKind.REGRESSION, "t", 0.9),),
        generalization=(_vr(ValidationKind.GENERALIZATION, "t", 0.5),),
    )
    # The regression probe PASSED, so it is not charged: it exists to confirm the
    # edit broke nothing, and passing it is evidence for the edit, not against.
    # 1.0*0.8 + 0.5*0.6 + 0 + 0.25*0.5 = 0.8 + 0.3 + 0.125 = 1.225
    assert r.weighted_net_gain() == pytest.approx(1.225)


def test_a_perfect_edit_is_not_rejected_by_its_own_passing_regression_probes():
    """The defect that made evolution arithmetically inert at >=2 tasks.

    With ``-1.0 * score`` applied to every regression probe, each PASSING probe
    subtracted up to 1.0. An edit that fixed its origin case and regressed
    nothing hit 1.0 - 1.0 = 0.0 as soon as a second task existed, so no edit
    could ever be accepted and every self-improvement delta was exactly zero for
    arithmetic reasons rather than agent quality.
    """
    for probe_count in range(0, 4):
        r = FocusedValidationReport(
            origin=(_vr(ValidationKind.ORIGIN, "t0", 1.0),),
            worked=(),
            regression=tuple(
                _vr(ValidationKind.REGRESSION, f"r{i}", 1.0)
                for i in range(probe_count)
            ),
        )
        gain = r.weighted_net_gain()
        decision = decide_acceptance(r)
        assert gain > 0.0, (
            f"{probe_count} passing regression probes made gain {gain}; "
            "passing probes must never push a good edit negative"
        )
        assert decision.accepted is True, (
            f"a perfect edit was rejected with {probe_count} passing probes"
        )


def test_failing_regression_probe_still_penalizes_net_gain():
    """The penalty must survive the fix, or real regressions become free.

    Acceptance is NOT asserted here: ``decide_acceptance`` deliberately tolerates
    a small regression when net gain stays positive (see
    ``test_decide_acceptance_allows_small_regression_when_net_gain_positive``).
    This test pins the magnitude only.
    """
    clean = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "t0", 1.0),),
        worked=(),
        regression=(_vr(ValidationKind.REGRESSION, "r0", 1.0),),
    )
    regressed = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "t0", 1.0),),
        worked=(),
        regression=(_vr(ValidationKind.REGRESSION, "r0", 0.1, passed=False),),
    )
    assert regressed.weighted_net_gain() < clean.weighted_net_gain()


def test_passing_regression_probes_are_free_at_every_probe_count():
    """Pins the exact arithmetic, not just its sign.

    ``run_evolution`` used to warn that multi-task runs were arithmetically
    inert because passing probes were charged -1.0 each. The values are pinned
    here so a regression to a per-probe charge is caught as a number rather than
    rediscovered as an inert run, and so the CLI's diagnostic text cannot drift
    back to claiming inertness.
    """
    for probe_count in (0, 1, 2, 3, 5):
        r = FocusedValidationReport(
            origin=(_vr(ValidationKind.ORIGIN, "t0", 1.0),),
            worked=(),
            regression=tuple(
                _vr(ValidationKind.REGRESSION, f"r{i}", 1.0)
                for i in range(probe_count)
            ),
        )
        assert r.weighted_net_gain() == pytest.approx(1.0), (
            f"{probe_count} passing regression probes changed the gain; passing "
            "probes are evidence for the edit and must cost exactly nothing"
        )


@pytest.mark.parametrize(
    "probe_score,expected_gain",
    [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)],
)
def test_a_failing_regression_probe_is_charged_one_minus_its_score(
    probe_score: float, expected_gain: float
):
    """The charge is the size of the fall, so a worse regression costs more.

    Pinning the exact value keeps the penalty from being quietly flattened to a
    constant, which would make a collapse to 0.1 cost the same as a dip to 0.9.
    """
    r = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "t0", 1.0),),
        worked=(),
        regression=(
            _vr(ValidationKind.REGRESSION, "r0", probe_score, passed=False),
        ),
    )

    assert r.weighted_net_gain() == pytest.approx(expected_gain)


def test_worse_regression_costs_more_than_a_mild_one():
    """The penalty scales with how far the probe fell."""
    mild = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "t0", 1.0),),
        worked=(),
        regression=(_vr(ValidationKind.REGRESSION, "r0", 0.9, passed=False),),
    )
    severe = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "t0", 1.0),),
        worked=(),
        regression=(_vr(ValidationKind.REGRESSION, "r0", 0.1, passed=False),),
    )
    assert severe.weighted_net_gain() < mild.weighted_net_gain()


# ---------------------------------------------------------------------- #
# Acceptance
# ---------------------------------------------------------------------- #
def test_decide_acceptance_accepts_when_all_pass_and_gain_positive():
    r = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "t", 0.8),),
        worked=(_vr(ValidationKind.WORKED, "t", 0.7),),
        regression=(_vr(ValidationKind.REGRESSION, "t", 0.9),),
    )
    d = decide_acceptance(r)
    assert d.accepted is True
    assert d.status == AttemptStatus.ACCEPTED


def test_decide_acceptance_rejects_when_origin_fails():
    r = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "t", 0.2, passed=False),),
        worked=(),
        regression=(),
    )
    d = decide_acceptance(r)
    assert d.accepted is False
    assert d.status == AttemptStatus.REJECTED


def test_decide_acceptance_allows_small_regression_when_net_gain_positive():
    """Per architecture: small regressions allowed when net gain > 0 and no floor.

    Probe ``score`` is the task score (higher is better) and real producers set
    ``passed = score >= 0.5`` (`orchestrator.py:486,1735`). A *small* regression
    is therefore a probe that dipped just below the bar, e.g. 0.45 -- not 0.1,
    which is a collapse. The original values (0.1 with a comment reading
    ``(-1.0)*0.1 = -0.1``) only looked small because the old formula charged the
    score itself, so the worse a probe scored the less it cost.
    """
    r = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "t", 0.9),),
        worked=(),
        regression=(_vr(ValidationKind.REGRESSION, "t", 0.45, passed=False),),
    )
    # net gain = 1.0*0.9 + (-1.0)*(1-0.45) = 0.9 - 0.55 = 0.35 > 0
    assert r.weighted_net_gain() == pytest.approx(0.35)
    d = decide_acceptance(r)
    assert d.accepted is True
    assert d.status == AttemptStatus.ACCEPTED


def test_decide_acceptance_rejects_regression_when_net_gain_negative():
    """When the regression cost overwhelms the gain, reject as REGRESSION."""
    r = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "t", 0.2, passed=True),),
        worked=(),
        regression=(_vr(ValidationKind.REGRESSION, "t", 0.1, passed=False),),
    )
    # net gain = 1.0*0.2 + (-1.0)*0.1 = 0.1 > 0; still accepted.
    # Make it negative: origin 0.05, regression 0.5.
    r2 = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "t", 0.05, passed=True),),
        worked=(),
        regression=(_vr(ValidationKind.REGRESSION, "t", 0.5, passed=False),),
    )
    # net gain = 1.0*0.05 + (-1.0)*0.5 = -0.45 < 0; regression_violated True.
    d = decide_acceptance(r2)
    assert d.accepted is False
    assert d.status == AttemptStatus.REGRESSION


def test_decide_acceptance_rejects_when_protected_floor_violated():
    r = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "tA", 0.9),),
        worked=(),
        regression=(),
    )
    # Floor on tB which has no probe in this report: floor is checked against
    # results for that task. With no tB results, the floor is not violated
    # (the helper only checks tasks that DO have results).
    floor = ProtectedFloor(task_id="tA", mechanism_cluster_id="c0", min_score=0.95)
    d = decide_acceptance(r, protected_floors=(floor,))
    assert d.accepted is False
    assert d.status == AttemptStatus.REGRESSION
    assert floor in d.protected_floors_violated


def test_decide_acceptance_rejects_when_net_gain_below_threshold():
    """A gain that does not clear the threshold is rejected.

    The regression probe here PASSED, so it contributes nothing (it is evidence
    the edit broke nothing). The gain therefore comes from the origin probe
    alone, and the threshold is set above it. Previously this test relied on a
    passing regression probe cancelling the origin exactly -- the arithmetic that
    made acceptance impossible once a second task existed.
    """
    r = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "t", 0.1, passed=True),),
        worked=(),
        regression=(_vr(ValidationKind.REGRESSION, "t", 0.1, passed=True),),
    )
    assert r.weighted_net_gain() == pytest.approx(0.1)
    d = decide_acceptance(r, net_gain_threshold=0.5)
    assert d.accepted is False


# ---------------------------------------------------------------------- #
# ValidationPlanner
# ---------------------------------------------------------------------- #
def test_planner_emits_origin_worked_regression_only_by_default():
    p = ValidationPlanner(
        origin_task=_task("tA"),
        worked_tasks=(_task("tW"),),
        regression_tasks=(_task("tR"),),
        generalization_tasks=(_task("tG"),),
    )
    probes = p.build_probes()
    kinds = {pr.kind for pr in probes}
    assert kinds == {ValidationKind.ORIGIN, ValidationKind.WORKED, ValidationKind.REGRESSION}


def test_planner_emits_generalization_when_flagged():
    p = ValidationPlanner(
        origin_task=_task("tA"),
        generalization_tasks=(_task("tG"),),
        emit_generalization_probes=True,
    )
    probes = p.build_probes()
    assert any(pr.kind == ValidationKind.GENERALIZATION for pr in probes)


def test_planner_origin_always_emitted():
    p = ValidationPlanner(origin_task=_task("tA"))
    probes = p.build_probes()
    assert len(probes) == 1
    assert probes[0].kind == ValidationKind.ORIGIN


# ---------------------------------------------------------------------- #
# build_attempt + record_attempt
# ---------------------------------------------------------------------- #
def test_build_attempt_assembles_record():
    response = EditorResponse(
        rationale="fix retrieval",
        edits=(_edit(),),
        reads={},
        writes={"skills/r1": "new content"},
        risks={},
        expected_effects={},
        editor_model_id="fake-editor",
    )
    r = FocusedValidationReport(
        origin=(_vr(ValidationKind.ORIGIN, "tA", 0.9),),
        worked=(),
        regression=(),
    )
    d = decide_acceptance(r)
    att = build_attempt(
        attempt_id="att-1",
        candidate_id="c1",
        issue_id="issue-1",
        response=response,
        evidence_refs=("trace-1",),
        history_refs=(),
        report=r,
        decision=d,
    )
    assert att.attempt_id == "att-1"
    assert att.artifact_ids == ("skills/r1",)
    assert att.status == AttemptStatus.ACCEPTED
    assert att.validation_summary["origin:tA"] == "pass"
    assert att.validation_summary["decision"] == "accepted"


def test_record_attempt_scopes_retry_budget():
    memory = EditMemory()
    response = EditorResponse(
        rationale="r",
        edits=(_edit(),),
        reads={},
        writes={},
        risks={},
        expected_effects={},
    )
    r = FocusedValidationReport(origin=(), worked=(), regression=())
    d = decide_acceptance(r)
    att = build_attempt(
        attempt_id="a1",
        candidate_id="c1",
        issue_id="i1",
        response=response,
        evidence_refs=(),
        history_refs=(),
        report=r,
        decision=d,
    )
    ws = _workspace()
    record_attempt(memory, att, ws)
    assert memory.retry_budget.remaining("i1", "skills/r1", "base-v0") == 2


def test_lineage_of_defaults_to_parent_version():
    ws = _workspace(parent="base-v0")
    assert lineage_of(ws) == "base-v0"


def test_lineage_of_uses_sorted_parents_when_provided():
    ws = _workspace(parent="base-v0")
    assert lineage_of(ws, ["c2-v0", "c1-v0"]) == "c1-v0|c2-v0"


# ---------------------------------------------------------------------- #
# Authorization boundary + repair protocol
# ---------------------------------------------------------------------- #
def plan_targeting(artifact_id: str, read_requests: tuple[str, ...] = ()) -> EditPlan:
    return EditPlan(
        attempt_id="att-1",
        issue_fingerprint="fp-1",
        read_requests=read_requests,
        authorized_writes=(artifact_id,),
        edits=(ArtifactEdit(artifact_id=artifact_id, operation="replace", payload={"content": "x"}),),
        rationale="fix",
        risks=(),
        expected_effect=ExpectedEffect(mechanism_cluster_refs=("cluster-1",)),
    )


def _editor_request() -> EditorRequest:
    return EditorRequest(
        base_workspace=_workspace(),
        task=_task(),
        analysis=empty_analysis(),
        issue_id="i1",
        write_set=("skills/a",),
        current_artifacts={"skills/a": "x"},
    )


class MalformedEditor:
    """An editor that always produces an empty-edit reply (contract rejects)."""

    editor_model_id = "malformed-editor"

    def propose_edit(self, request: EditorRequest) -> EditorResponse:
        return EditorResponse(
            rationale="r", edits=(), reads={}, writes={}, risks={}, expected_effects={}
        )


class RecoveringEditor:
    """Returns malformed output until it receives a correction request."""

    editor_model_id = "recovering-editor"

    def __init__(self) -> None:
        self.proposals = 0

    def propose_edit(self, request: EditorRequest) -> EditorResponse:
        self.proposals += 1
        if not request.correction_request:
            raise ValueError("edits is required (cannot be empty)")
        return EditorResponse(
            rationale="repaired",
            edits=(
                ArtifactEdit(
                    artifact_id="skills/a", operation="replace", payload={"content": "y"}
                ),
            ),
            reads={},
            writes={},
            risks={},
            expected_effects={},
        )


def test_editor_plan_rejects_edit_outside_authorized_write_set() -> None:
    with pytest.raises(WriteAuthorizationError):
        validate_editor_plan(plan_targeting("artifact-outside"), authorized_writes=("artifact-inside",))


def test_second_malformed_response_returns_recordable_non_promotion() -> None:
    result = repair_once_then_classify(MalformedEditor(), _editor_request())
    assert result.status == "malformed"
    assert result.correction_requests == 1
    assert result.response is None


def test_valid_editor_plan_passes_authorization() -> None:
    plan = plan_targeting("skills/a", read_requests=("skills/a",))
    assert (
        validate_editor_plan(
            plan,
            readable=frozenset({"skills/a"}),
            authorized_writes=frozenset({"skills/a"}),
        )
        is plan
    )


def test_editor_plan_rejects_unreadable_read_request() -> None:
    with pytest.raises(WriteAuthorizationError):
        validate_editor_plan(
            plan_targeting("skills/a", read_requests=("skills/b",)),
            readable=frozenset({"skills/a"}),
            authorized_writes=frozenset({"skills/a"}),
        )


def test_repair_once_then_valid_recovers() -> None:
    editor = RecoveringEditor()
    result = repair_once_then_classify(editor, _editor_request())
    assert result.status == "valid"
    assert result.correction_requests == 1
    assert result.response is not None
    assert editor.proposals == 2
