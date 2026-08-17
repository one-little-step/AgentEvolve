"""Tests for non-answer detection: a give-up is unscorable, not a wrong answer.

Every "detected" string in this file was extracted verbatim from a real
``causal-trace.json:final_output`` under ``data/traces/`` (235 traces walked).
The trace id is cited in each docstring so a reviewer can re-read the source.

The anti-regression tests matter more than the detection tests. A false positive
silently removes a genuinely-wrong answer from the denominator, which inflates
the reported pass rate -- the exact corruption this module exists to prevent.
"""
from __future__ import annotations

from agent_evolve.core.non_answer import (
    NON_ANSWER_EMPTY,
    NON_ANSWER_INABILITY,
    NON_ANSWER_NARRATION,
    NON_ANSWER_RESTATEMENT,
    classify_non_answer,
    is_non_answer,
)


# --------------------------------------------------------------------------- #
# empty / whitespace-only
# --------------------------------------------------------------------------- #


def test_empty_output_is_unscorable() -> None:
    """Trace 383b5801 reports status=success with final_output=''.

    A successful status with no text is the harness claiming an answer it did
    not produce; scoring it as wrong would credit an outage as a wrong answer.
    """
    verdict = classify_non_answer("")
    assert verdict.is_non_answer
    assert verdict.category == NON_ANSWER_EMPTY


def test_whitespace_only_output_is_unscorable() -> None:
    """Whitespace is indistinguishable from empty for measurement purposes."""
    assert classify_non_answer("   \n\t  \n ").category == NON_ANSWER_EMPTY


def test_none_output_is_unscorable() -> None:
    """A missing final_output must not raise on the scoring path."""
    assert is_non_answer(None)


# --------------------------------------------------------------------------- #
# explicit inability
# --------------------------------------------------------------------------- #


def test_explicit_inability_to_load_a_skill_is_unscorable() -> None:
    """Trace 8becce68 (also 98ed2154, 116cdc4b): tool call unavailable.

    Observed 4x across data/traces. The agent states it could not act at all,
    so there is no claim to grade.
    """
    text = (
        "I\u2019m unable to load the status-report skill because the required "
        "tool call was not available in the execution environment."
    )
    verdict = classify_non_answer(text)
    assert verdict.is_non_answer
    assert verdict.category == NON_ANSWER_INABILITY


def test_apology_with_no_committed_value_is_unscorable() -> None:
    """Trace tiny5-baseline-a/24a44dc1: 'I'm sorry, but I wasn't able to complete the computation.'

    Present in all three tiny5 baseline replicates (a/b/c), so it is a stable
    give-up shape rather than a one-off.
    """
    text = "I\u2019m sorry, but I wasn\u2019t able to complete the computation."
    assert classify_non_answer(text).category == NON_ANSWER_INABILITY


def test_inability_to_execute_a_tool_call_is_unscorable() -> None:
    """Trace 19f5417b (also 17cb583b, b97e4954): 'unable to execute the tool call'."""
    text = "I\u2019m unable to execute the tool call in this turn."
    assert classify_non_answer(text).category == NON_ANSWER_INABILITY


def test_inability_stated_after_narration_is_unscorable() -> None:
    """Trace 4e2ac9b4: narration line, then an explicit refusal to guess.

    The give-up is the *last* line. Detection must read the final committed
    segment, not the opening plan, or every multi-line rollout escapes.
    """
    text = (
        "I\u2019ll search for the exact BBC Earth upload now, then verify the "
        "bird species from the result details.\n"
        "I\u2019m unable to access the web-search tool in this session, so I "
        "can\u2019t reliably verify the species without guessing."
    )
    assert classify_non_answer(text).category == NON_ANSWER_INABILITY


def test_bare_unable_to_retrieve_is_unscorable() -> None:
    """Trace 0cb88c5a: 'Unable to retrieve the Wikipedia data needed to verify the count.'

    A first-person pronoun is not required for a give-up to be a give-up.
    """
    text = "Unable to retrieve the Wikipedia data needed to verify the count."
    assert classify_non_answer(text).category == NON_ANSWER_INABILITY


def test_no_access_claim_is_unscorable() -> None:
    """The 'I don't have access to' shape, required by the task specification."""
    text = "I don\u2019t have access to the requested database."
    assert classify_non_answer(text).category == NON_ANSWER_INABILITY


def test_was_unable_to_determine_is_unscorable() -> None:
    """The 'I was unable to determine' shape, required by the task specification."""
    assert is_non_answer("I was unable to determine the value.")


# --------------------------------------------------------------------------- #
# pure narration
# --------------------------------------------------------------------------- #


def test_forward_looking_narration_alone_is_unscorable() -> None:
    """'I would need to...' commits to nothing; there is no answer to grade."""
    text = "I would need to access the source table before I could answer."
    assert classify_non_answer(text).category == NON_ANSWER_NARRATION


def test_let_me_try_narration_alone_is_unscorable() -> None:
    """'Let me try...' is a plan, not a result."""
    assert classify_non_answer("Let me try searching the archive again.").category == (
        NON_ANSWER_NARRATION
    )


def test_narration_followed_by_a_real_answer_is_scored() -> None:
    """Trace 60ace227 opens with a plan and still commits a value.

    Narration is only a non-answer when it is the *whole* output. This is the
    single most likely over-detection, so it is pinned explicitly.
    """
    text = (
        "I\u2019ll check the footage timestamps first.\n"
        "The highest number of bird species on camera simultaneously is 3."
    )
    assert not is_non_answer(text)


# --------------------------------------------------------------------------- #
# restatement of the task
# --------------------------------------------------------------------------- #


def test_output_that_only_restates_the_task_is_unscorable() -> None:
    """Echoing the question commits to no answer, so nothing can be measured."""
    question = "How many studio albums did Mercedes Sosa publish between 2000 and 2009?"
    assert classify_non_answer(question, question=question).category == (
        NON_ANSWER_RESTATEMENT
    )


def test_restatement_detection_requires_the_question_to_be_supplied() -> None:
    """Without the question there is no evidence of restatement; stay silent.

    Guessing restatement from shape alone would be a precision loss.
    """
    question = "How many studio albums did Mercedes Sosa publish between 2000 and 2009?"
    assert not is_non_answer(question)


def test_an_answer_that_quotes_the_task_then_answers_is_scored() -> None:
    """A restatement plus a committed value is a real answer."""
    question = "How many studio albums did Mercedes Sosa publish between 2000 and 2009?"
    text = f"{question} She published 4 studio albums."
    assert not is_non_answer(text, question=question)


# --------------------------------------------------------------------------- #
# anti-regression: a wrong answer must stay wrong
# --------------------------------------------------------------------------- #


def test_a_wrong_but_committed_answer_is_still_scored() -> None:
    """'The answer is 42' when 7 was expected is WRONG, not unscorable.

    Over-detection here would delete genuine failures from the denominator and
    fake a self-improvement delta. This is the critical guarantee.
    """
    assert not is_non_answer("The answer is 42")


def test_a_correct_answer_is_unaffected() -> None:
    """Trace 60ace227: a plain committed answer must never be flagged."""
    assert not is_non_answer(
        "The highest number of bird species on camera simultaneously is 3."
    )


def test_hedged_answer_that_still_commits_a_value_is_scored() -> None:
    """Trace 86cb8405 hedges about verification but commits '**519 at-bats**'.

    This is the highest-risk real case in data/traces: the text contains BOTH
    'I'm unable to complete the source check' AND 'the answer is **519
    at-bats**'. A committed value outranks surrounding hedging, otherwise a
    gradeable answer is silently removed from the denominator.
    """
    text = (
        "The search results conflict with the earlier claim: the StatMuse "
        "snippet points to Roy White (519 AB), so I\u2019m checking the "
        "detailed 1977 Yankees batting table before answering.\n"
        "I\u2019m unable to complete the source check because the browsing "
        "tool is not returning the detailed table in this run. The available "
        "search result indicates the answer is **519 at-bats**, for **Roy "
        "White**, but I can\u2019t reliably confirm the walks-leader "
        "identification from the retrieved data."
    )
    assert not is_non_answer(text)


def test_a_short_token_answer_is_scored() -> None:
    """'done' is the most common final_output in data/traces (122 of 235).

    It is a committed (if terse) output from the offline harness. Treating
    short outputs as non-answers would empty the offline denominator.
    """
    assert not is_non_answer("done")


def test_an_answer_mentioning_inability_of_a_third_party_is_scored() -> None:
    """Only the agent's own give-up counts; a factual sentence is gradeable."""
    assert not is_non_answer("The 1977 roster was unable to field a full team.")


def test_a_numeric_answer_with_units_is_scored() -> None:
    """Trace-observed committed value: '0.1777 m3' (3 occurrences)."""
    assert not is_non_answer("The fish bag\u2019s calculated volume was 0.1777 m\u00b3.")


# --------------------------------------------------------------------------- #
# reason text
# --------------------------------------------------------------------------- #


def test_the_verdict_explains_why_it_is_unscorable() -> None:
    """An unscorable rollout must be auditable, not just excluded silently."""
    verdict = classify_non_answer("I\u2019m unable to execute the tool call in this turn.")
    assert verdict.reason
    assert "non-answer" in verdict.reason


def test_a_scorable_answer_carries_no_category_and_no_reason() -> None:
    """The negative case must be unambiguous at the call site."""
    verdict = classify_non_answer("The answer is 42")
    assert not verdict.is_non_answer
    assert verdict.category == ""
    assert verdict.reason == ""
