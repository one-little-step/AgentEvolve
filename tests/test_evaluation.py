"""Tests for four-category validation plans and acceptance rules."""
from __future__ import annotations

from agent_evolve.core.contracts import ValidationCase, ValidationResult
from agent_evolve.core.evaluation import (
    AcceptanceDecision,
    build_validation_plan,
    decide_acceptance,
    summarize_cases,
)

origin_case = ValidationCase(case_id="origin-1", outcome="passed")


def result(
    *,
    primary_gain: float = 0.4,
    weighted_net_gain: float = 0.2,
    protected_floor_outcome: str = "satisfied",
) -> ValidationResult:
    return ValidationResult(
        origin_cases=(origin_case,),
        worked_cases=(),
        regression_cases=(),
        generalization_cases=(),
        primary_gain=primary_gain,
        weighted_net_gain=weighted_net_gain,
        protected_floor_outcome=protected_floor_outcome,
        decision="reject" if protected_floor_outcome == "violated" else "accept",
        decision_reason="test",
        unavailable_cases=(),
    )


def test_deferred_generalization_is_explicitly_unverified() -> None:
    plan = build_validation_plan(origin_case, written_artifacts=("artifact-1",), probe_mode="deferred")
    assert plan.generalization_status == "generalization_unverified"
    assert plan.generalization_cases == ()


def test_protected_floor_forces_rejection_despite_positive_gain() -> None:
    decision = decide_acceptance(result(primary_gain=0.4, weighted_net_gain=0.2, protected_floor_outcome="violated"))
    assert decision.decision == "reject"


def test_unavailable_case_is_not_counted_as_passing() -> None:
    assert summarize_cases((ValidationCase(case_id="x", outcome="unavailable"),)).passed == 0


def test_positive_gain_accepts() -> None:
    decision = decide_acceptance(result(primary_gain=0.4, weighted_net_gain=0.2))
    assert decision.decision == "accept"
    assert decision.reason == "validated_gain"


def test_non_positive_primary_gain_rejects() -> None:
    decision = decide_acceptance(result(primary_gain=0.0, weighted_net_gain=0.2))
    assert decision.decision == "reject"
    assert decision.reason == "primary_gain_not_positive"


def test_non_positive_weighted_net_gain_rejects() -> None:
    decision = decide_acceptance(result(primary_gain=0.4, weighted_net_gain=-0.1))
    assert decision.decision == "reject"
    assert decision.reason == "weighted_net_gain_not_positive"


def test_summarize_counts_failed_and_unavailable_separately() -> None:
    cases = (
        ValidationCase(case_id="p", outcome="passed"),
        ValidationCase(case_id="f", outcome="failed"),
        ValidationCase(case_id="u", outcome="unavailable"),
    )
    summary = summarize_cases(cases)
    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.unavailable == 1


def test_acceptance_decision_is_a_frozen_dataclass() -> None:
    decision = AcceptanceDecision("accept", "validated_gain")
    assert decision.decision == "accept"
    assert decision.reason == "validated_gain"
