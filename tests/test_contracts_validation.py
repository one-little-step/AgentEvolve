"""Strict boundary-contract validation required before core orchestration.

These are new Pydantic models beside the prototype runtime dataclasses.  They
lock the persisted/boundary shape without making an incompatible migration of
existing in-memory call sites in this remediation increment.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_evolve.core.contracts import (
    ArtifactEdit,
    ArtifactMergeDecision,
    AttemptRecord,
    EditPlan,
    ExpectedEffect,
    MergeProvenance,
    MemoryRecord,
    RedactionReport,
    ScoreCell,
    ValidationCase,
    ValidationResult,
)


def score_cell_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "candidate_id": "candidate-1",
        "task_id": "task-1",
        "mechanism_cluster_id": "cluster-1",
        "score": 0.75,
        "severity": 0.8,
        "confidence": 0.9,
        "stability": 0.5,
        "rollout_count": 2,
        "rollout_ids": ("rollout-1", "rollout-2"),
        "verdict_refs": ("verdict-1",),
        "artifact_versions": {"artifact-1": "sha256:abc"},
        "evaluator_id": "judge-v1",
        "coverage": "evaluated",
        "coverage_reason": None,
    }
    values.update(changes)
    return values


def attempt_record_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "attempt_id": "attempt-1",
        "snapshot_version": "snapshot-1",
        "parent_candidate_id": "candidate-1",
        "result_candidate_id": None,
        "status": "rejected",
        "issue_fingerprint": "issue-1",
        "task_refs": ("task-1",),
        "mechanism_cluster_refs": ("cluster-1",),
        "read_set": ("artifact-1",),
        "write_set": ("artifact-1",),
        "workspace_sealed": False,
        "hashes_before": {"artifact-1": "sha256:abcdef"},
        "hashes_after": {},
        "analysis_refs": ("analysis-1",),
        "verdict_refs": ("verdict-1",),
        "memory_refs": ("memory-1",),
        "validation_result_ref": "validation-1",
        "rationale_summary": "bounded sanitized summary",
        "risk_summary": "bounded sanitized risk",
        "budget_usage": {},
        "retry_state": {},
        "timestamps": {},
    }
    values.update(changes)
    return values


def test_score_cell_rejects_zero_rollouts() -> None:
    with pytest.raises(ValidationError, match="rollout_count"):
        ScoreCell(**score_cell_values(rollout_count=0, rollout_ids=()))


def test_score_cell_rejects_empty_cluster_id() -> None:
    with pytest.raises(ValidationError, match="mechanism_cluster_id"):
        ScoreCell(**score_cell_values(mechanism_cluster_id=""))


def test_score_cell_rejects_empty_verdict_refs() -> None:
    with pytest.raises(ValidationError, match="verdict_refs"):
        ScoreCell(**score_cell_values(verdict_refs=()))


def test_score_cell_rejects_empty_artifact_versions() -> None:
    with pytest.raises(ValidationError, match="artifact_versions"):
        ScoreCell(**score_cell_values(artifact_versions={}))


def test_score_cell_rejects_malformed_artifact_versions() -> None:
    with pytest.raises(ValidationError, match="artifact_versions"):
        ScoreCell(**score_cell_values(artifact_versions={"artifact-1": "not-a-hash"}))


@pytest.mark.parametrize(
    "changes",
    (
        {"rollout_ids": ("", "rollout-2")},
        {"verdict_refs": ("",)},
        {"artifact_versions": {"": "sha256:abcdef"}},
    ),
)
def test_score_cell_rejects_blank_references_and_artifact_ids(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ScoreCell(**score_cell_values(**changes))


def test_score_cell_rejects_single_rollout_stability() -> None:
    with pytest.raises(ValidationError, match="stability"):
        ScoreCell(**score_cell_values(
            rollout_count=1,
            rollout_ids=("rollout-1",),
            stability=0.5,
        ))


def test_score_cell_requires_stability_after_one_rollout() -> None:
    with pytest.raises(ValidationError, match="stability"):
        ScoreCell(**score_cell_values(stability=None))


def test_score_cell_requires_reason_for_unavailable_coverage() -> None:
    with pytest.raises(ValidationError, match="coverage_reason"):
        ScoreCell(**score_cell_values(coverage="unavailable", coverage_reason=None))


def test_score_cell_accepts_reason_for_excluded_coverage() -> None:
    cell = ScoreCell(**score_cell_values(
        coverage="excluded",
        coverage_reason="outside-evaluation-scope",
    ))

    assert cell.coverage_reason == "outside-evaluation-scope"


def test_score_cell_requires_reason_for_excluded_coverage() -> None:
    with pytest.raises(ValidationError, match="coverage_reason"):
        ScoreCell(**score_cell_values(coverage="excluded", coverage_reason=None))


@pytest.mark.parametrize("coverage", ("unavailable", "excluded"))
@pytest.mark.parametrize("coverage_reason", ("", "   "))
def test_score_cell_rejects_blank_non_evaluated_coverage_reason(
    coverage: str, coverage_reason: str
) -> None:
    with pytest.raises(ValidationError, match="coverage_reason"):
        ScoreCell(**score_cell_values(
            coverage=coverage,
            coverage_reason=coverage_reason,
        ))


def test_score_cell_rejects_reason_for_evaluated_coverage() -> None:
    with pytest.raises(ValidationError, match="coverage_reason"):
        ScoreCell(**score_cell_values(coverage_reason="not-applicable"))


@pytest.mark.parametrize(
    ("changes", "field"),
    (
        ({"rollout_count": 2, "rollout_ids": ("rollout-1",)}, "rollout_ids"),
        ({"rollout_ids": ("rollout-1", "rollout-1")}, "rollout_ids"),
    ),
)
def test_score_cell_rejects_provenance_relations(
    changes: dict[str, object], field: str
) -> None:
    with pytest.raises(ValidationError, match=field):
        ScoreCell(**score_cell_values(**changes))


@pytest.mark.parametrize(
    ("field", "value"),
    tuple(
        (field, value)
        for field in ("score", "severity", "confidence", "stability")
        for value in (-0.1, 1.1)
    ),
)
def test_score_cell_rejects_out_of_range_score_values(
    field: str, value: float
) -> None:
    with pytest.raises(ValidationError, match=field):
        ScoreCell(**score_cell_values(**{field: value}))


@pytest.mark.parametrize(
    "changes",
    (
        {"score": "0.75"},
        {"rollout_ids": ["rollout-1", "rollout-2"]},
    ),
)
def test_score_cell_accepts_standard_pydantic_coercion(changes: dict[str, object]) -> None:
    assert ScoreCell(**score_cell_values(**changes)).candidate_id == "candidate-1"


@pytest.mark.parametrize(
    "status",
    ("accepted", "rejected", "no_op", "malformed", "exhausted", "unavailable"),
)
def test_attempt_record_accepts_only_mandated_terminal_statuses(status: str) -> None:
    result_candidate_id = "candidate-2" if status == "accepted" else None
    validation_result_ref = "validation-1" if status in {"accepted", "rejected"} else None

    record = AttemptRecord(**attempt_record_values(
        status=status,
        result_candidate_id=result_candidate_id,
        validation_result_ref=validation_result_ref,
    ))

    assert record.status == status


def test_attempt_record_rejects_unknown_terminal_status() -> None:
    with pytest.raises(ValidationError, match="status"):
        AttemptRecord(**attempt_record_values(status="in_progress"))


def test_accepted_attempt_requires_result_candidate() -> None:
    with pytest.raises(ValidationError, match="result_candidate_id"):
        AttemptRecord(**attempt_record_values(status="accepted", validation_result_ref="validation-1"))


def test_accepted_attempt_requires_validation_result() -> None:
    with pytest.raises(ValidationError, match="validation_result_ref"):
        AttemptRecord(**attempt_record_values(status="accepted", result_candidate_id="candidate-2", validation_result_ref=None))


def test_non_accepted_attempt_rejects_result_candidate() -> None:
    with pytest.raises(ValidationError, match="result_candidate_id"):
        AttemptRecord(**attempt_record_values(status="rejected", result_candidate_id="candidate-2"))


def test_rejected_attempt_requires_validation_result() -> None:
    with pytest.raises(ValidationError, match="validation_result_ref"):
        AttemptRecord(**attempt_record_values(validation_result_ref=None))


def test_rejected_attempt_requires_evidence_references() -> None:
    with pytest.raises(ValidationError, match="memory_refs"):
        AttemptRecord(**attempt_record_values(memory_refs=()))


def test_unavailable_attempt_allows_empty_evidence_references() -> None:
    record = AttemptRecord(**attempt_record_values(
        status="unavailable",
        validation_result_ref=None,
        analysis_refs=(),
        verdict_refs=(),
        memory_refs=(),
    ))

    assert record.status == "unavailable"


def test_unsealed_attempt_rejects_hashes_after() -> None:
    with pytest.raises(ValidationError, match="hashes_after"):
        AttemptRecord(**attempt_record_values(hashes_after={"artifact-1": "sha256:abcdef"}))


def test_sealed_attempt_requires_hashes_after() -> None:
    with pytest.raises(ValidationError, match="hashes_after"):
        AttemptRecord(**attempt_record_values(workspace_sealed=True))


@pytest.mark.parametrize(
    "changes",
    (
        {"hashes_before": {"artifact-1": "not-a-hash"}},
        {
            "workspace_sealed": True,
            "hashes_after": {"artifact-1": "not-a-hash"},
        },
    ),
)
def test_attempt_record_rejects_malformed_hash_mappings(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="hash mappings"):
        AttemptRecord(**attempt_record_values(**changes))


@pytest.mark.parametrize(
    "changes",
    (
        {"task_refs": ("",)},
        {"mechanism_cluster_refs": ("",)},
        {"read_set": ("",)},
        {"write_set": ("",)},
        {"hashes_before": {"": "sha256:abcdef"}},
        {"workspace_sealed": True, "hashes_after": {"": "sha256:abcdef"}},
        {"analysis_refs": ("",)},
        {"verdict_refs": ("",)},
        {"memory_refs": ("",)},
    ),
)
def test_attempt_record_rejects_blank_nested_ids(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AttemptRecord(**attempt_record_values(**changes))


def test_edit_plan_rejects_edit_outside_authorized_writes() -> None:
    with pytest.raises(ValidationError, match="authorized_writes"):
        EditPlan(
            attempt_id="attempt-1",
            issue_fingerprint="issue-1",
            read_requests=("artifact-read",),
            authorized_writes=("artifact-write",),
            edits=(
                ArtifactEdit(
                    artifact_id="artifact-read",
                    operation="replace",
                    payload={},
                ),
            ),
            rationale="bounded sanitized rationale",
            risks=(),
            expected_effect=ExpectedEffect(mechanism_cluster_refs=("cluster-1",)),
        )


def test_artifact_edit_requires_non_empty_identifiers() -> None:
    with pytest.raises(ValidationError, match="artifact_id"):
        ArtifactEdit(artifact_id="", operation="replace", payload={})


def test_artifact_edit_rejects_empty_operation() -> None:
    with pytest.raises(ValidationError, match="operation"):
        ArtifactEdit(artifact_id="artifact-write", operation="", payload={})


def test_expected_effect_requires_mechanism_cluster_references() -> None:
    with pytest.raises(ValidationError, match="mechanism_cluster_refs"):
        ExpectedEffect(mechanism_cluster_refs=())


def test_edit_plan_rejects_blank_nested_artifact_ids() -> None:
    for changes in (
        {"read_requests": ("",)},
        {"authorized_writes": ("",)},
    ):
        values: dict[str, object] = {
            "attempt_id": "attempt-1",
            "issue_fingerprint": "issue-1",
            "read_requests": ("artifact-read",),
            "authorized_writes": ("artifact-write",),
            "edits": (
                ArtifactEdit(
                    artifact_id="artifact-write",
                    operation="replace",
                    payload={},
                ),
            ),
            "rationale": "bounded sanitized rationale",
            "expected_effect": ExpectedEffect(mechanism_cluster_refs=("cluster-1",)),
        }
        values.update(changes)
        with pytest.raises(ValidationError):
            EditPlan(**values)


def test_expected_effect_rejects_blank_mechanism_cluster_reference() -> None:
    with pytest.raises(ValidationError):
        ExpectedEffect(mechanism_cluster_refs=("",))


def test_edit_plan_requires_actual_edits() -> None:
    with pytest.raises(ValidationError, match="edits"):
        EditPlan(
            attempt_id="attempt-1",
            issue_fingerprint="issue-1",
            read_requests=("artifact-read",),
            authorized_writes=("artifact-write",),
            edits=(),
            rationale="bounded sanitized rationale",
            risks=(),
            expected_effect=ExpectedEffect(mechanism_cluster_refs=("cluster-1",)),
        )


def validation_case_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "case_id": "case-1",
        "outcome": "passed",
    }
    values.update(changes)
    return values


def validation_result_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "origin_cases": (ValidationCase(**validation_case_values()),),
        "worked_cases": (),
        "regression_cases": (),
        "generalization_cases": (),
        "primary_gain": 0.2,
        "weighted_net_gain": 0.1,
        "protected_floor_outcome": "satisfied",
        "decision": "accept",
        "decision_reason": "origin evidence improved",
        "unavailable_cases": (),
    }
    values.update(changes)
    return values


def test_validation_result_rejects_accept_with_violated_protected_floor() -> None:
    with pytest.raises(ValidationError, match="protected_floor_outcome"):
        ValidationResult(**validation_result_values(
            protected_floor_outcome="violated",
            decision="accept",
        ))


def test_validation_result_requires_origin_cases() -> None:
    with pytest.raises(ValidationError, match="origin_cases"):
        ValidationResult(**validation_result_values(origin_cases=()))


def test_validation_result_rejects_available_case_in_unavailable_cases() -> None:
    with pytest.raises(ValidationError, match="unavailable_cases"):
        ValidationResult(**validation_result_values(
            unavailable_cases=(
                ValidationCase(**validation_case_values(outcome="passed")),
            ),
        ))


def test_validation_case_rejects_unknown_outcome() -> None:
    with pytest.raises(ValidationError, match="outcome"):
        ValidationCase(**validation_case_values(outcome="unknown"))


def test_merge_provenance_requires_distinct_parent_candidates() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        MergeProvenance(
            merge_id="merge-1",
            ancestor_candidate_id="candidate-1",
            left_candidate_id="candidate-1",
            right_candidate_id="candidate-2",
            child_admitted=False,
            child_candidate_id=None,
            artifact_decisions=(_artifact_decision(),),
            complementarity=0.5,
            eligibility_checks={"comparable": True},
        )


def test_artifact_merge_decision_requires_ancestor_hash_when_inherited() -> None:
    with pytest.raises(ValidationError, match="resulting_hash"):
        _artifact_decision(resulting_hash="sha256:fedcba")


def merge_decision_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "artifact_id": "artifact-1",
        "ancestor_hash": "sha256:abcdef",
        "left_hash": "sha256:123abc",
        "right_hash": "sha256:456def",
        "resulting_hash": "sha256:123abc",
        "inheritance": "left",
        "evidence_score_left": 0.4,
        "evidence_score_right": 0.3,
        "decision_reason": "left evidence is stronger",
        "refinement_request_ref": None,
        "operation_emitted": True,
    }
    values.update(changes)
    return values


def merge_provenance_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "merge_id": "merge-1",
        "ancestor_candidate_id": "candidate-1",
        "left_candidate_id": "candidate-2",
        "right_candidate_id": "candidate-3",
        "child_admitted": False,
        "child_candidate_id": None,
        "artifact_decisions": (ArtifactMergeDecision(**merge_decision_values()),),
        "complementarity": 0.5,
        "eligibility_checks": {"comparable": True},
    }
    values.update(changes)
    return values


def memory_record_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "memory_record_id": "memory-1",
        "attempt_id": "attempt-1",
        "artifact_ids": ("artifact-1",),
        "issue_fingerprint": "issue-1",
        "outcome": "rejected",
        "summary": "bounded sanitized summary",
        "evidence_refs": ("evidence-1",),
        "redaction_report": RedactionReport(rule_hits=(), truncations=0),
    }
    values.update(changes)
    return values


def test_refined_merge_requires_refinement_request() -> None:
    with pytest.raises(ValidationError, match="refinement_request_ref"):
        ArtifactMergeDecision(**merge_decision_values(
            inheritance="refined",
            refinement_request_ref=None,
        ))


def test_shared_merge_requires_equal_parent_hashes() -> None:
    with pytest.raises(ValidationError, match="shared inheritance"):
        ArtifactMergeDecision(**merge_decision_values(inheritance="shared"))


def test_ancestor_result_cannot_emit_operation() -> None:
    with pytest.raises(ValidationError, match="operation_emitted"):
        ArtifactMergeDecision(**merge_decision_values(
            resulting_hash="sha256:abcdef",
            operation_emitted=True,
        ))


def test_admitted_merge_requires_child_candidate() -> None:
    with pytest.raises(ValidationError, match="child_candidate_id"):
        MergeProvenance(**merge_provenance_values(
            child_admitted=True,
            child_candidate_id=None,
        ))


def test_unadmitted_merge_rejects_child_candidate() -> None:
    with pytest.raises(ValidationError, match="child_candidate_id"):
        MergeProvenance(**merge_provenance_values(child_candidate_id="candidate-4"))


def test_merge_provenance_rejects_blank_child_candidate_id() -> None:
    with pytest.raises(ValidationError):
        MergeProvenance(**merge_provenance_values(
            child_admitted=True,
            child_candidate_id="",
        ))


def test_memory_record_forbids_raw_prompt() -> None:
    with pytest.raises(ValidationError, match="raw_prompt"):
        MemoryRecord(**memory_record_values(), raw_prompt="secret")


def test_memory_record_accepts_attempt_outcome_and_redaction_report() -> None:
    record = MemoryRecord(**memory_record_values())

    assert record.outcome == "rejected"
    assert record.redaction_report.truncations == 0


@pytest.mark.parametrize("field", ("ancestor_hash", "left_hash", "right_hash", "resulting_hash"))
def test_merge_decision_rejects_non_content_hash(field: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactMergeDecision(**merge_decision_values(**{field: "not-a-hash"}))


def test_memory_record_rejects_blank_artifact_or_evidence_reference() -> None:
    with pytest.raises(ValidationError):
        MemoryRecord(**memory_record_values(artifact_ids=("",)))
    with pytest.raises(ValidationError):
        MemoryRecord(**memory_record_values(evidence_refs=("",)))


def _artifact_decision(**changes: object) -> ArtifactMergeDecision:
    values: dict[str, object] = {
        "artifact_id": "artifact-1",
        "ancestor_hash": "sha256:abcdef",
        "left_hash": "sha256:123abc",
        "right_hash": "sha256:456def",
        "resulting_hash": "sha256:abcdef",
        "inheritance": "ancestor",
        "evidence_score_left": 0.4,
        "evidence_score_right": 0.3,
        "decision_reason": "tie retained the ancestor",
        "refinement_request_ref": None,
        "operation_emitted": False,
    }
    values.update(changes)
    return ArtifactMergeDecision(**values)
