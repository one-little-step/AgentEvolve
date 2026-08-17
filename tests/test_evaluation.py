"""Tests for four-category validation plans and acceptance rules."""
from __future__ import annotations

from agent_evolve.core.contracts import (
    EvolutionTask,
    ExecutionTrace,
    ValidationCase,
    ValidationResult,
)
from agent_evolve.core.evaluation import (
    CONTRACT_GRADER_NAME,
    AcceptanceDecision,
    ContractScorer,
    build_validation_plan,
    decide_acceptance,
    summarize_cases,
    tally_scores,
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


# --------------------------------------------------------------------------- #
# non-answer detection reaches the tally the delta is computed from
# --------------------------------------------------------------------------- #


def _trace(final_output: str, *, status: str = "success") -> ExecutionTrace:
    return ExecutionTrace(
        trace_id="t-1",
        candidate_id="c-1",
        task_id="task-1",
        status=status,
        final_output=final_output,
        events=(),
    )


def _task() -> EvolutionTask:
    return EvolutionTask(
        task_id="task-1",
        input_text="How many bird species are on camera simultaneously?",
        expected_contract={"expected_substring": "3"},
    )


def test_a_give_up_rollout_is_unscorable_rather_than_a_failing_contract_score() -> None:
    """Trace 8ec45ba0: 'I'm sorry, but I couldn't access the video or a reliable transcript...'

    The contract grader would report 0.0 because '3' is absent. That is a
    failure-to-match the rollout never attempted, and it belongs outside the
    denominator.
    """
    score = ContractScorer().score_rollout(
        _task(),
        _trace(
            "I\u2019m sorry, but I couldn\u2019t access the video or a "
            "reliable transcript to verify the answer."
        ),
    )
    assert not score.scorable
    assert "non-answer" in score.reason


def test_a_wrong_but_committed_rollout_is_still_scored_as_wrong() -> None:
    """The anti-regression at the tally boundary.

    If this became unscorable, every wrong answer could leave the denominator
    and the reported delta would rise without the agent improving.
    """
    score = ContractScorer().score_rollout(_task(), _trace("The answer is 42"))
    assert score.scorable
    assert not score.passed


def test_a_correct_rollout_is_still_scored_as_passing() -> None:
    """Detection must not disturb the positive case."""
    score = ContractScorer().score_rollout(
        _task(), _trace("The highest number of bird species is 3.")
    )
    assert score.scorable
    assert score.passed


def test_the_tally_excludes_non_answers_from_the_denominator() -> None:
    """A non-answer must be visible in the tally and absent from pass_rate.

    One pass and one non-answer is 1/1 = 100%, not 1/2 = 50%: the second
    rollout produced nothing to measure.
    """
    scorer = ContractScorer()
    scores = (
        scorer.score_rollout(_task(), _trace("The count is 3.")),
        scorer.score_rollout(_task(), _trace("I\u2019m unable to execute the tool call in this turn.")),
    )
    tally = tally_scores(scores, grader_name=CONTRACT_GRADER_NAME)

    assert tally.evaluated == 1
    assert tally.passed == 1
    assert tally.attempted == 2
    assert tally.unscorable == 1
    assert tally.pass_rate == 1.0


def test_pass_rate_is_none_when_every_rollout_is_a_non_answer() -> None:
    """Nothing committed means nothing to score -- not a 0% pass rate."""
    scorer = ContractScorer()
    scores = tuple(
        scorer.score_rollout(
            _task(), _trace("I\u2019m unable to execute the tool call in this turn.")
        )
        for _ in range(3)
    )
    tally = tally_scores(scores, grader_name=CONTRACT_GRADER_NAME)

    assert tally.evaluated == 0
    assert tally.unscorable == 3
    assert tally.pass_rate is None


def test_the_tally_summary_states_the_unscorable_count() -> None:
    """An operator must see the exclusion, or an inflated denominator recurs."""
    scorer = ContractScorer()
    scores = (
        scorer.score_rollout(_task(), _trace("The count is 3.")),
        scorer.score_rollout(_task(), _trace("")),
    )
    tally = tally_scores(scores, grader_name=CONTRACT_GRADER_NAME)

    assert "unscorable=1" in tally.summary
