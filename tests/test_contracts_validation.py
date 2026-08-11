"""Strict boundary-contract validation required before core orchestration.

These are new Pydantic models beside the prototype runtime dataclasses.  They
lock the persisted/boundary shape without making an incompatible migration of
existing in-memory call sites in this remediation increment.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_evolve.core.contracts import (
    ArtifactMergeDecision,
    AttemptRecord,
    EditPlan,
    MergeProvenance,
    ScoreCell,
)


def _score_cell(**changes: object) -> ScoreCell:
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
    }
    values.update(changes)
    return ScoreCell(**values)


def _attempt_record(**changes: object) -> AttemptRecord:
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
        "hashes_before": {"artifact-1": "sha256:before"},
        "hashes_after": {},
        "analysis_refs": ("analysis-1",),
        "verdict_refs": ("verdict-1",),
        "memory_refs": (),
        "validation_result_ref": "validation-1",
        "rationale_summary": "bounded sanitized summary",
        "risk_summary": "bounded sanitized risk",
        "budget_usage": {},
        "retry_state": {},
        "timestamps": {},
    }
    values.update(changes)
    return AttemptRecord(**values)


def test_score_cell_rejects_zero_rollouts() -> None:
    with pytest.raises(ValidationError, match="rollout_count"):
        _score_cell(rollout_count=0, rollout_ids=())


def test_score_cell_rejects_empty_mechanism_cluster_id() -> None:
    with pytest.raises(ValidationError, match="mechanism_cluster_id"):
        _score_cell(mechanism_cluster_id="")


def test_score_cell_rejects_single_rollout_stability() -> None:
    with pytest.raises(ValidationError, match="stability"):
        _score_cell(
            rollout_count=1,
            rollout_ids=("rollout-1",),
            stability=0.5,
        )


@pytest.mark.parametrize(
    "status",
    ("accepted", "rejected", "no_op", "malformed", "exhausted", "unavailable"),
)
def test_attempt_record_accepts_only_mandated_terminal_statuses(status: str) -> None:
    result_candidate_id = "candidate-2" if status == "accepted" else None
    validation_result_ref = "validation-1" if status in {"accepted", "rejected"} else None

    record = _attempt_record(
        status=status,
        result_candidate_id=result_candidate_id,
        validation_result_ref=validation_result_ref,
    )

    assert record.status == status


def test_attempt_record_rejects_unknown_terminal_status() -> None:
    with pytest.raises(ValidationError, match="status"):
        _attempt_record(status="in_progress")


def test_accepted_attempt_requires_result_candidate_and_validation() -> None:
    with pytest.raises(ValidationError, match="result_candidate_id"):
        _attempt_record(status="accepted", validation_result_ref="validation-1")


def test_non_accepted_attempt_cannot_claim_a_result_candidate() -> None:
    with pytest.raises(ValidationError, match="result_candidate_id"):
        _attempt_record(status="rejected", result_candidate_id="candidate-2")


def test_edit_plan_rejects_an_edit_outside_authorized_writes() -> None:
    with pytest.raises(ValidationError, match="authorized_writes"):
        EditPlan(
            attempt_id="attempt-1",
            issue_fingerprint="issue-1",
            read_requests=("artifact-read",),
            authorized_writes=("artifact-write",),
            edit_targets=("artifact-read",),
            rationale="bounded sanitized rationale",
        )


def test_merge_provenance_requires_distinct_parent_candidates() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        MergeProvenance(
            merge_id="merge-1",
            ancestor_candidate_id="candidate-1",
            left_candidate_id="candidate-1",
            right_candidate_id="candidate-2",
            child_candidate_id=None,
            artifact_decisions=(_artifact_decision(),),
            complementarity=0.5,
            eligibility_checks={"comparable": True},
        )


def test_artifact_merge_decision_requires_ancestor_hash_when_inherited() -> None:
    with pytest.raises(ValidationError, match="resulting_hash"):
        _artifact_decision(resulting_hash="sha256:other")


def _artifact_decision(**changes: object) -> ArtifactMergeDecision:
    values: dict[str, object] = {
        "artifact_id": "artifact-1",
        "ancestor_hash": "sha256:ancestor",
        "left_hash": "sha256:left",
        "right_hash": "sha256:right",
        "resulting_hash": "sha256:ancestor",
        "inheritance": "ancestor",
        "evidence_score_left": 0.4,
        "evidence_score_right": 0.3,
        "decision_reason": "tie retained the ancestor",
    }
    values.update(changes)
    return ArtifactMergeDecision(**values)
