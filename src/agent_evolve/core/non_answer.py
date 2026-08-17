"""Non-answer detection: a rollout that gives up is unscorable, not wrong.

Why this module exists
----------------------
``BenchmarkRunResult`` and :class:`~agent_evolve.core.evaluation.ScoreTally`
already separate "no answer" from "wrong answer" for *execution* failures: a
dead worker yields ``pass_rate is None`` rather than a 0% pass rate. This module
extends that same discipline one step further, to the case where the worker
*did* return text but the text is not an answer -- an apology, a statement of
inability, pure narration, or an echo of the task.

Grading such an output against ground truth records a failure-to-match. That
inflates the denominator with rollouts that never committed to a claim, and the
resulting pass rate then measures tool availability rather than agent skill. In
the Gaia replay dataset ``gaia_l1_validation__baseline__20260813_035541``, 10 of
42 recorded answers end in an explicit statement of inability; all 10 currently
count as regex failures.

Precision over recall -- deliberately
-------------------------------------
The two error directions are **not** symmetric:

* A **false negative** (a give-up graded as wrong) leaves the pre-existing,
  already-published behaviour in place. It understates the pass rate.
* A **false positive** (a genuinely wrong answer classified as unscorable)
  deletes a real failure from the denominator and therefore *flatters* the
  reported result. It manufactures a self-improvement delta out of nothing.

Every rule below is therefore tuned for high precision and is allowed to miss
ambiguous cases:

1. Only the **final committed segment** (the last non-blank line) is examined.
   Real rollouts narrate a plan first and answer last; judging the whole blob
   would flag any rollout that merely *mentioned* a difficulty.
2. A **committed value outranks all hedging.** If the final segment names an
   answer ("the answer is ...", "final answer: ..."), the output is scorable no
   matter how much apology surrounds it. Trace
   ``86cb8405-177b-4f65-8097-8210e1e5a5a0`` is exactly this shape: it says both
   "I'm unable to complete the source check" and "the answer is **519
   at-bats**". It is a gradeable answer.
3. Inability phrases must appear **at the start** of the final segment, so a
   factual sentence that happens to contain "unable to" is untouched.
4. Restatement is only detected when the caller supplies the question. Guessing
   it from shape alone would trade precision for nothing.

Patterns are grounded in observed data, not imagination: all 235
``causal-trace.json`` files under ``data/traces/`` were scanned and clustered by
final segment before these rules were written. 30 of 235 are flagged.

This module is agent-neutral: it imports nothing from ``cuga`` or any adapter,
and nothing from :mod:`agent_evolve.benchmarks`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "NON_ANSWER_EMPTY",
    "NON_ANSWER_INABILITY",
    "NON_ANSWER_NARRATION",
    "NON_ANSWER_RESTATEMENT",
    "NonAnswerVerdict",
    "classify_non_answer",
    "final_committed_segment",
    "is_non_answer",
]

#: Category names. Stable strings: they are recorded in tallies and run reports,
#: so an operator can tell *which* kind of non-answer dominated a run.
NON_ANSWER_EMPTY = "empty_output"
NON_ANSWER_INABILITY = "explicit_inability"
NON_ANSWER_NARRATION = "narration_without_answer"
NON_ANSWER_RESTATEMENT = "task_restatement"

#: Straight quote, right single quote, and modifier letter apostrophe. Real
#: rollout text uses U+2019, so an ASCII-only pattern silently matches nothing.
_AP = r"['\u2019\u02bc]"

#: A named answer. Checked FIRST and it wins: hedging around a committed value
#: does not make the value ungradeable.
_COMMITTED_VALUE = re.compile(
    rf"(?:"
    rf"the\s+answer\s+is"
    rf"|final\s+answer"
    rf"|answer\s*[:=]"
    rf"|my\s+answer\s+is"
    rf")",
    re.IGNORECASE,
)

#: Explicit inability, anchored to the start of the final committed segment.
#: Each alternative was observed in ``data/traces`` unless marked otherwise.
_INABILITY = re.compile(
    rf"^\W*(?:"
    rf"i\s*{_AP}?\s*m\s+sorry"  # "I'm sorry, but I wasn't able to ..."
    rf"|i\s+am\s+sorry"
    rf"|sorry\b"
    rf"|i\s+apolog|apologies\b"  # spec-required, not observed
    rf"|i\s*{_AP}?\s*m\s+(?:unable|not\s+able)"  # "I'm unable to execute ..."
    rf"|i\s+am\s+(?:unable|not\s+able)"
    rf"|i\s+cannot|i\s+can\s*{_AP}?\s*t|i\s+can\s+not"  # "I can't verify ..."
    rf"|i\s+could\s*n\s*{_AP}?\s*t|i\s+could\s+not"  # "I couldn't verify ..."
    rf"|i\s+was\s*n\s*{_AP}?\s*t\s+able|i\s+was\s+not\s+able"
    rf"|i\s+was\s+unable"  # "I was unable to determine ..."
    rf"|i\s+have\s*n\s*{_AP}?\s*t\s+been\s+able|i\s+have\s+not\s+been\s+able"
    rf"|i\s+do\s*n\s*{_AP}?\s*t\s+have\s+access|i\s+do\s+not\s+have\s+access"
    rf"|i\s+failed\s+to\b"
    rf"|unable\s+to\b"  # "Unable to retrieve the Wikipedia data ..."
    rf")",
    re.IGNORECASE,
)

#: Forward-looking intent, anchored. Zero incidence in the 235 on-disk traces
#: (no trace ends on a narration line), so this rule is kept deliberately narrow
#: -- it additionally requires the absence of any digit, because "I'll go with
#: 42" is a committed answer wearing narration's clothes.
_NARRATION = re.compile(
    rf"^\W*(?:"
    rf"i\s+would\s+need"
    rf"|i\s+will\s+need"
    rf"|let\s+me\s+(?:try|check|search|look|start|begin)"
    rf"|i\s*{_AP}?\s*ll\s+(?:try|check|search|look|start|begin|verify|retrieve|compute|locate|use)"
    rf"|i\s+will\s+(?:try|check|search|look|start|begin|verify|retrieve|compute|locate|use)"
    rf"|next,?\s+i\s*{_AP}?\s*(?:ll|m)\b"
    rf"|i\s*{_AP}?\s*m\s+(?:checking|searching|looking|going\s+to|about\s+to)"
    rf"|searching\s+(?:for|the)\b"
    rf")",
    re.IGNORECASE,
)

_DIGIT = re.compile(r"\d")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class NonAnswerVerdict:
    """Whether one output is a non-answer, which kind, and why.

    ``category`` and ``reason`` are empty exactly when ``is_non_answer`` is
    false, so a call site cannot mistake a scorable answer for a classified one.
    """

    is_non_answer: bool
    category: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.is_non_answer:
            if not self.category:
                raise ValueError("a non-answer verdict requires a category")
            if not self.reason:
                raise ValueError("a non-answer verdict requires a reason")
        else:
            if self.category or self.reason:
                raise ValueError(
                    "a scorable verdict must carry no category and no reason: "
                    "'not an answer' and 'an answer' are different facts"
                )


_SCORABLE = NonAnswerVerdict(is_non_answer=False)


def final_committed_segment(text: str) -> str:
    """The last non-blank line: what the rollout actually left on the table.

    Rollouts narrate their plan and then answer, so the closing line is the
    committed position. Reading the whole blob instead would flag every rollout
    that merely described an obstacle en route to a correct answer.
    """
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().strip(".?!:;,").casefold()


def classify_non_answer(
    text: str | None, *, question: str | None = None
) -> NonAnswerVerdict:
    """Classify ``text`` as a non-answer, or report it as scorable.

    High precision by design: an ambiguous output is reported scorable. A false
    positive removes a genuinely-wrong answer from the denominator and inflates
    the reported pass rate, which is strictly worse than a missed give-up.

    ``question`` is optional and enables restatement detection only.
    """
    if text is None or not text.strip():
        return NonAnswerVerdict(
            is_non_answer=True,
            category=NON_ANSWER_EMPTY,
            reason="non-answer: output was empty or whitespace only",
        )

    segment = final_committed_segment(text)

    # A named answer outranks every other signal, including its own hedging.
    if _COMMITTED_VALUE.search(segment):
        return _SCORABLE

    if _INABILITY.match(segment):
        return NonAnswerVerdict(
            is_non_answer=True,
            category=NON_ANSWER_INABILITY,
            reason=(
                "non-answer: the rollout closed by stating it could not "
                "produce an answer"
            ),
        )

    if _NARRATION.match(segment) and not _DIGIT.search(segment):
        return NonAnswerVerdict(
            is_non_answer=True,
            category=NON_ANSWER_NARRATION,
            reason=(
                "non-answer: the rollout closed on stated intent without "
                "committing to an answer"
            ),
        )

    if question and _normalize(text) == _normalize(question):
        return NonAnswerVerdict(
            is_non_answer=True,
            category=NON_ANSWER_RESTATEMENT,
            reason="non-answer: the output only restates the task",
        )

    return _SCORABLE


def is_non_answer(text: str | None, *, question: str | None = None) -> bool:
    """Convenience predicate. See :func:`classify_non_answer` for the rules."""
    return classify_non_answer(text, question=question).is_non_answer
