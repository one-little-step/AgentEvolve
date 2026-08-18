"""Tests for the shared Interface B pairwise preference judge.

The judge decides candidate ranking, so it is the selection signal of the whole
loop. Two properties are load-bearing and tested hardest:

* the score is SIGNED and oriented ``baseline -> candidate``, so which side won
  survives aggregation, and
* a constant preference for whichever trajectory sits in the "candidate" slot is
  cancelled by the symmetric (swapped-slot) comparison rather than being read as
  a real improvement.

Ground truth is supplied when the split has it; the test splits carry a
placeholder regex of ``(?i)\\?`` which matches any question mark and must never
be treated as an answer.

Every test injects a fake ``agent_factory``. No test makes a network call.
"""
from __future__ import annotations

import json

from agent_evolve.adapters.cuga_preference_judge import (
    APP_NAMES,
    JUDGE_INSTRUCTIONS,
    MAX_SCORE,
    MIN_SCORE,
    PLACEHOLDER_REGEXES,
    STATUS_INVALID_SCORE,
    STATUS_NO_SUBMIT,
    STATUS_NO_TOOL_CALL,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    PreferenceJudge,
    PreferenceSummary,
    PreferenceVerdict,
    aggregate_preferences,
    is_placeholder_regex,
)
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace, TraceEvent


def _task(expected: str | None = r"(?i)\b17\b") -> EvolutionTask:
    contract: dict = {}
    if expected is not None:
        contract["expected_regex"] = expected
    return EvolutionTask(
        task_id="gaia-1",
        input_text="how many albums",
        expected_contract=contract,
    )


def _trace(trace_id: str, output: str, candidate_id: str) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        candidate_id=candidate_id,
        task_id="gaia-1",
        events=(
            TraceEvent(
                event_id="e1",
                kind="llm_call_end",
                actor_id="call_model",
                parent_event_id=None,
                payload={"text": "working"},
            ),
        ),
        final_output=output,
        status="success",
    )


def _baseline() -> ExecutionTrace:
    return _trace("t-base", "I could not determine it.", "base")


def _candidate() -> ExecutionTrace:
    return _trace("t-cand", "17", "cand-1")


def _submitting(score: float, rationale: str = "r"):
    """A fake agent that reads both slots then submits ``score``."""

    def factory(callables: dict, prompt: str) -> str:
        callables["get_task"]()
        callables["read_baseline"]()
        callables["read_candidate"]()
        callables["submit_preference"](score=score, rationale=rationale)
        return "done"

    return factory


# --------------------------------------------------------------- sign convention
def test_positive_score_favors_the_candidate() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["submit_preference"](score=0.8, rationale="candidate executed a tool")
        return "done"

    verdict = PreferenceJudge(agent_factory=factory).compare(
        _task(), _baseline(), _candidate()
    )

    assert isinstance(verdict, PreferenceVerdict)
    assert verdict.available is True
    assert verdict.score == 0.8
    assert verdict.winner == "candidate"
    assert verdict.status == STATUS_OK


def test_negative_score_favors_the_baseline() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["submit_preference"](score=-0.5, rationale="candidate regressed")
        return "done"

    verdict = PreferenceJudge(agent_factory=factory).compare(
        _task(), _baseline(), _candidate()
    )

    assert verdict.score == -0.5
    assert verdict.winner == "baseline"


def test_zero_score_is_a_tie() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["submit_preference"](score=0.0, rationale="indistinguishable")
        return "done"

    verdict = PreferenceJudge(agent_factory=factory).compare(
        _task(), _baseline(), _candidate()
    )

    assert verdict.winner == "tie"
    assert verdict.available is True


def test_score_bounds_are_the_signed_unit_interval() -> None:
    assert (MIN_SCORE, MAX_SCORE) == (-1.0, 1.0)


# --------------------------------------------------------------- ground truth
def test_ground_truth_is_supplied_when_present() -> None:
    seen: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        seen.append(callables["get_task"]())
        callables["submit_preference"](score=0.5, rationale="r")
        return "done"

    verdict = PreferenceJudge(agent_factory=factory).compare(
        _task(), _baseline(), _candidate()
    )

    assert verdict.gt_available is True
    assert "17" in seen[0]


def test_expected_regex_is_labelled_as_a_pattern_not_a_literal() -> None:
    """A judge told ``expected_answer: (?i)\\b17\\b`` may string-match the pattern."""
    seen: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        seen.append(callables["get_task"]())
        callables["submit_preference"](score=0.5, rationale="r")
        return "done"

    PreferenceJudge(agent_factory=factory).compare(_task(), _baseline(), _candidate())

    payload = json.loads(seen[0])
    assert payload["expected_answer_kind"] == "regex"


def test_placeholder_regex_is_not_treated_as_ground_truth() -> None:
    seen: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        seen.append(callables["get_task"]())
        callables["submit_preference"](score=0.5, rationale="r")
        return "done"

    verdict = PreferenceJudge(agent_factory=factory).compare(
        _task(expected=r"(?i)\?"), _baseline(), _candidate()
    )

    assert verdict.gt_available is False
    payload = json.loads(seen[0])
    assert payload.get("expected_answer") in (None, "")


def test_placeholder_set_contains_the_known_stub() -> None:
    assert r"(?i)\?" in PLACEHOLDER_REGEXES


def test_vacuous_patterns_are_placeholders_whatever_their_inline_flags() -> None:
    assert is_placeholder_regex(r"(?i)\?") is True
    assert is_placeholder_regex(r"\?") is True
    assert is_placeholder_regex("(?is) .* ") is True
    assert is_placeholder_regex(r"(?i)\b17\b") is False


def test_absent_ground_truth_falls_back_to_process_comparison() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["submit_preference"](score=0.3, rationale="better process")
        return "done"

    verdict = PreferenceJudge(agent_factory=factory).compare(
        _task(expected=None), _baseline(), _candidate()
    )

    assert verdict.gt_available is False
    assert verdict.available is True


def test_prompt_states_ground_truth_is_absent_and_forbids_inventing_it() -> None:
    prompts: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        prompts.append(prompt)
        callables["submit_preference"](score=0.0, rationale="r")
        return "done"

    PreferenceJudge(agent_factory=factory).compare(
        _task(expected=None), _baseline(), _candidate()
    )

    assert "NO GROUND TRUTH IS AVAILABLE" in prompts[0]
    assert "invent" in prompts[0]


def test_prompt_states_ground_truth_is_available_when_it_is() -> None:
    prompts: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        prompts.append(prompt)
        callables["submit_preference"](score=0.0, rationale="r")
        return "done"

    PreferenceJudge(agent_factory=factory).compare(_task(), _baseline(), _candidate())

    assert "GROUND TRUTH IS AVAILABLE" in prompts[0]
    assert "NO GROUND TRUTH IS AVAILABLE" not in prompts[0]


# --------------------------------------------------------------- prompt quality
def test_prompt_refuses_length_fluency_and_slot_as_evidence() -> None:
    """Length/verbosity/position bias are the known failure modes of an LLM judge."""
    prompts: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        prompts.append(prompt)
        callables["submit_preference"](score=0.0, rationale="r")
        return "done"

    PreferenceJudge(agent_factory=factory).compare(_task(), _baseline(), _candidate())

    text = prompts[0].lower()
    assert "length" in text
    assert "confident" in text
    assert "slot" in text


def test_instructions_forbid_crediting_a_side_for_its_label() -> None:
    # Whitespace-normalized: the prompt is hard-wrapped for readability, and a
    # line break inside a sentence is not a change in meaning.
    lowered = " ".join(JUDGE_INSTRUCTIONS.lower().split())
    assert "impartial" in lowered
    assert "because it is new" in lowered
    assert "swapped" in lowered


def test_instructions_are_passed_as_special_instructions() -> None:
    """The tool contract must be prefixed by the judge's role, not replaced."""
    captured: dict = {}

    def fake_run(callables, prompt, **kwargs):
        captured.update(kwargs)
        from agent_evolve.adapters.cuga_workspace_agent import WorkspaceAgentRun

        callables["submit_preference"](score=0.0, rationale="r")
        return WorkspaceAgentRun(answer="done", tools_called=("submit_preference",))

    import agent_evolve.adapters.cuga_preference_judge as mod

    original = mod.run_workspace_agent
    mod.run_workspace_agent = fake_run
    try:
        PreferenceJudge(agent_factory=lambda c, p: "x").compare(
            _task(), _baseline(), _candidate()
        )
    finally:
        mod.run_workspace_agent = original

    assert captured["special_instructions"] == JUDGE_INSTRUCTIONS


def test_every_tool_has_a_docstring_and_a_real_signature() -> None:
    """LangChain's @tool raises without a docstring and builds an empty args schema
    without a signature, silently telling the model every tool takes no args."""
    import inspect

    seen: dict = {}

    def factory(callables: dict, prompt: str) -> str:
        seen.update(callables)
        callables["submit_preference"](score=0.0, rationale="r")
        return "done"

    PreferenceJudge(agent_factory=factory).compare(_task(), _baseline(), _candidate())

    # list_tools is injected by run_workspace_agent for every Interface B agent.
    assert set(seen) == set(APP_NAMES) | {"list_tools"}
    for name, fn in seen.items():
        assert (fn.__doc__ or "").strip(), f"{name} has no docstring"
        inspect.signature(fn)  # must not raise
    assert "score" in inspect.signature(seen["submit_preference"]).parameters


# --------------------------------------------------------------- failure modes
def test_out_of_range_score_is_unavailable_not_a_tie() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["submit_preference"](score=7.0, rationale="r")
        return "done"

    verdict = PreferenceJudge(agent_factory=factory).compare(
        _task(), _baseline(), _candidate()
    )

    assert verdict.available is False
    assert verdict.winner != "tie"
    assert "score" in verdict.error
    assert verdict.status == STATUS_INVALID_SCORE


def test_non_numeric_score_is_unavailable() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["submit_preference"](score="much better", rationale="r")
        return "done"

    verdict = PreferenceJudge(agent_factory=factory).compare(
        _task(), _baseline(), _candidate()
    )

    assert verdict.available is False
    assert verdict.status == STATUS_INVALID_SCORE


def test_never_submitting_is_unavailable_not_a_tie() -> None:
    def factory(callables: dict, prompt: str) -> str:
        return "I prefer the candidate."  # narration only

    verdict = PreferenceJudge(agent_factory=factory).compare(
        _task(), _baseline(), _candidate()
    )

    assert verdict.available is False
    assert verdict.status == STATUS_NO_TOOL_CALL


def test_reading_but_not_submitting_is_no_submit() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["read_baseline"]()
        callables["read_candidate"]()
        return "the candidate looks better"

    verdict = PreferenceJudge(agent_factory=factory).compare(
        _task(), _baseline(), _candidate()
    )

    assert verdict.available is False
    assert verdict.status == STATUS_NO_SUBMIT


def test_agent_failure_is_unavailable() -> None:
    def factory(callables: dict, prompt: str) -> str:
        raise RuntimeError("judge exploded")

    verdict = PreferenceJudge(agent_factory=factory).compare(
        _task(), _baseline(), _candidate()
    )

    assert verdict.available is False
    assert verdict.status == STATUS_UNAVAILABLE
    assert "judge exploded" in verdict.error


def test_a_second_submit_is_rejected_and_the_first_verdict_stands() -> None:
    replies: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        callables["submit_preference"](score=0.4, rationale="first")
        replies.append(callables["submit_preference"](score=-0.9, rationale="second"))
        return "done"

    verdict = PreferenceJudge(agent_factory=factory).compare(
        _task(), _baseline(), _candidate()
    )

    assert verdict.score == 0.4
    assert verdict.rationale == "first"
    assert "already" in replies[0]


# --------------------------------------------------------------- evidence access
def test_both_trajectories_are_readable() -> None:
    seen: list[dict] = []

    def factory(callables: dict, prompt: str) -> str:
        seen.append(json.loads(callables["read_baseline"]()))
        seen.append(json.loads(callables["read_candidate"]()))
        callables["submit_preference"](score=0.1, rationale="r")
        return "done"

    PreferenceJudge(agent_factory=factory).compare(_task(), _baseline(), _candidate())

    assert seen[0]["final_output"] == "I could not determine it."
    assert seen[1]["final_output"] == "17"


def test_verdict_records_whether_both_slots_were_inspected() -> None:
    lazy = PreferenceJudge(
        agent_factory=lambda c, p: c["submit_preference"](score=0.5, rationale="r")
    ).compare(_task(), _baseline(), _candidate())
    thorough = PreferenceJudge(agent_factory=_submitting(0.5)).compare(
        _task(), _baseline(), _candidate()
    )

    assert lazy.inspected_both is False
    assert thorough.inspected_both is True


def test_tools_called_ledger_is_recorded_on_the_verdict() -> None:
    verdict = PreferenceJudge(agent_factory=_submitting(0.5)).compare(
        _task(), _baseline(), _candidate()
    )

    assert verdict.tools_called == (
        "get_task",
        "read_baseline",
        "read_candidate",
        "submit_preference",
    )


# --------------------------------------------------------------- position bias
def test_symmetric_comparison_cancels_a_constant_pro_candidate_slot_bias() -> None:
    """A judge that always favours the candidate SLOT has learned nothing."""

    def factory(callables: dict, prompt: str) -> str:
        callables["read_baseline"]()
        callables["read_candidate"]()
        callables["submit_preference"](score=0.6, rationale="the candidate looks new")
        return "done"

    verdict = PreferenceJudge(agent_factory=factory).compare_symmetric(
        _task(), _baseline(), _candidate()
    )

    assert verdict.available is True
    assert verdict.score == 0.0
    assert verdict.winner == "tie"
    assert verdict.position_bias == 0.6
    assert verdict.comparisons == 2
    assert verdict.orientation == "symmetric"


def test_symmetric_comparison_preserves_a_content_driven_preference() -> None:
    """Scoring on what the trajectory actually says survives the swap."""

    def factory(callables: dict, prompt: str) -> str:
        cand = json.loads(callables["read_candidate"]())
        score = 0.8 if cand["final_output"] == "17" else -0.8
        callables["submit_preference"](score=score, rationale="answered or not")
        return "done"

    verdict = PreferenceJudge(agent_factory=factory).compare_symmetric(
        _task(), _baseline(), _candidate()
    )

    assert verdict.score == 0.8
    assert verdict.winner == "candidate"
    assert verdict.position_bias == 0.0


def test_symmetric_comparison_preserves_a_content_driven_regression() -> None:
    def factory(callables: dict, prompt: str) -> str:
        cand = json.loads(callables["read_candidate"]())
        score = -0.6 if cand["final_output"] == "17" else 0.6
        callables["submit_preference"](score=score, rationale="r")
        return "done"

    verdict = PreferenceJudge(agent_factory=factory).compare_symmetric(
        _task(), _baseline(), _candidate()
    )

    assert verdict.score == -0.6
    assert verdict.winner == "baseline"


def test_symmetric_second_pass_swaps_which_slot_holds_which_trajectory() -> None:
    slots: list[tuple[str, str]] = []

    def factory(callables: dict, prompt: str) -> str:
        base = json.loads(callables["read_baseline"]())["final_output"]
        cand = json.loads(callables["read_candidate"]())["final_output"]
        slots.append((base, cand))
        callables["submit_preference"](score=0.0, rationale="r")
        return "done"

    PreferenceJudge(agent_factory=factory).compare_symmetric(
        _task(), _baseline(), _candidate()
    )

    assert slots[0] == ("I could not determine it.", "17")
    assert slots[1] == ("17", "I could not determine it.")


def test_symmetric_is_unavailable_when_either_direction_fails() -> None:
    calls: list[int] = []

    def factory(callables: dict, prompt: str) -> str:
        calls.append(1)
        if len(calls) == 1:
            callables["submit_preference"](score=0.5, rationale="r")
            return "done"
        raise RuntimeError("reverse pass exploded")

    verdict = PreferenceJudge(agent_factory=factory).compare_symmetric(
        _task(), _baseline(), _candidate()
    )

    assert verdict.available is False
    assert verdict.winner == "unavailable"
    assert "reverse" in verdict.error


def test_symmetric_rationale_carries_both_directions() -> None:
    verdict = PreferenceJudge(agent_factory=_submitting(0.2, "because tools ran")).compare_symmetric(
        _task(), _baseline(), _candidate()
    )

    assert verdict.rationale.count("because tools ran") == 2
    assert "forward" in verdict.rationale
    assert "reversed" in verdict.rationale


# --------------------------------------------------------------- aggregation
def test_aggregate_excludes_unavailable_verdicts_rather_than_tying_them() -> None:
    summary = aggregate_preferences(
        [
            PreferenceVerdict(task_id="a", score=0.8, winner="candidate", available=True),
            PreferenceVerdict(task_id="b", score=0.4, winner="candidate", available=True),
            PreferenceVerdict(task_id="c", error="boom"),
        ]
    )

    assert isinstance(summary, PreferenceSummary)
    assert summary.available == 2
    assert summary.unavailable == 1
    assert summary.mean_score == 0.6  # not 0.4, which averaging in a tie would give
    assert summary.candidate_wins == 2
    assert summary.decided is True


def test_aggregate_of_all_unavailable_verdicts_is_undecided() -> None:
    summary = aggregate_preferences([PreferenceVerdict(task_id="a", error="boom")])

    assert summary.decided is False
    assert summary.mean_score == 0.0
    assert summary.available == 0


def test_aggregate_counts_signed_outcomes_separately() -> None:
    summary = aggregate_preferences(
        [
            PreferenceVerdict(task_id="a", score=0.5, winner="candidate", available=True),
            PreferenceVerdict(task_id="b", score=-0.5, winner="baseline", available=True),
            PreferenceVerdict(task_id="c", score=0.0, winner="tie", available=True),
        ]
    )

    assert (summary.candidate_wins, summary.baseline_wins, summary.ties) == (1, 1, 1)
    assert summary.mean_score == 0.0


def test_aggregate_of_nothing_is_undecided() -> None:
    assert aggregate_preferences([]).decided is False
