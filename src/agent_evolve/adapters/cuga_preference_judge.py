"""Interface B: signed pairwise preference between a baseline and a candidate.

Shared by the RHO stage and the existing genetic stage. Implemented as a full
CUGA-SDK workspace agent, matching the published implementation
(``RHO_agents_context.md:165-192``): the judge gets an evaluation workspace
holding the task, both trajectories, and tools to inspect them before committing
to a verdict. A cheaper structured call was considered and rejected, for fidelity
to the paper, with the cost accepted.

Why the sign matters
--------------------
The score is SIGNED and describes the transition ``baseline -> candidate``:
positive favours the candidate, negative favours the baseline, zero is a genuine
tie. Collapsing this to a boolean would throw away *by how much* and *which side*
won, which is exactly the information selection needs.

Why symmetry matters
--------------------
This judge is the selection signal of the whole loop, so its two known failure
modes are not cosmetic:

* **position / slot bias** -- an LLM judge shown "baseline" and "candidate" tends
  to reward whichever side is framed as the new one. A constant slot preference
  is indistinguishable from a real improvement in a single-direction comparison.
* **sycophancy toward fluent narration** -- a longer, more confident-sounding
  trajectory reads as better even when it never committed an answer.

``JUDGE_INSTRUCTIONS`` and the prompt attack both directly, and
:meth:`PreferenceJudge.compare_symmetric` attacks position bias structurally:
it runs the comparison twice with the two trajectories in swapped slots and
reports ``(forward - reversed) / 2`` as the score and ``(forward + reversed) / 2``
as the measured ``position_bias``. A judge that always says "+0.6 for the
candidate slot" scores 0.0 with a bias of 0.6 -- its bias becomes an observable
instead of a silent inflation of every candidate.

Ground truth
------------
Task metadata including ground truth IS supplied when available. This is a
deliberate, recorded deviation from the ``AGENTS.md`` no-labels rule (spec
section 7): containment is by prompting rather than by a hard firewall.

Not every split has ground truth. ``gaia_l1_test.json`` and
``gaia_l1_test_tiny10.json`` carry a single shared regex of ``(?i)\\?`` for every
task, which matches any question mark and so passes vacuously. That is a
placeholder, not an answer, and is filtered here so the judge is never told a
stub is ground truth. When GT is absent the judge is told so explicitly and told
not to invent one.

Ground truth also arrives as a *regex*, not a literal, so it is labelled
``expected_answer_kind: "regex"``. A judge handed ``(?i)\\b17\\b`` and told it was
an answer would happily reward a trajectory that printed the pattern.

Failure is observable
---------------------
An unparseable, out-of-range, or never-submitted verdict is UNAVAILABLE and
contributes nothing to a candidate's average. Defaulting it to a tie would
silently pull every candidate toward its baseline, and a judge that fails often
would look like a loop that produces no improvement.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from agent_evolve.adapters.cuga_workspace_agent import run_workspace_agent
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace

JUDGE_MODEL_ID = "cuga-preference-judge"

MIN_SCORE = -1.0
MAX_SCORE = 1.0

#: Number of decimals kept on aggregated scores. Float addition makes
#: ``(0.8 + 0.4) / 2`` differ from ``0.6``, which would make otherwise identical
#: candidates compare unequal.
_ROUND = 6

# ------------------------------------------------------------------ statuses
#: A verdict was produced and is usable.
STATUS_OK = "ok"
#: The agent narrated but executed no tool at all.
STATUS_NO_TOOL_CALL = "no_tool_call"
#: The agent used the workspace but never committed a verdict.
STATUS_NO_SUBMIT = "no_submit"
#: A verdict was committed but the score is not a number in [-1, 1].
STATUS_INVALID_SCORE = "invalid_score"
#: The invocation itself failed.
STATUS_UNAVAILABLE = "unavailable"

#: Winner vocabulary. ``"unavailable"`` is NOT a tie.
WINNER_CANDIDATE = "candidate"
WINNER_BASELINE = "baseline"
WINNER_TIE = "tie"
WINNER_UNAVAILABLE = "unavailable"

ORIENTATION_FORWARD = "forward"
ORIENTATION_REVERSED = "reversed"
ORIENTATION_SYMMETRIC = "symmetric"

#: Regexes that are stubs rather than ground truth. ``(?i)\?`` matches any
#: question mark, so every answer "passes" -- treating it as GT would make the
#: judge confidently wrong on an entire split.
PLACEHOLDER_REGEXES = frozenset({r"(?i)\?", r"\?", "?", r"(?i)\\?", ".*", ".+"})

#: Normalised cores that match essentially anything.
_VACUOUS_CORES = frozenset({r"\?", "?", ".*", ".+", ".*?", ".+?", "", "^", "$", "^$"})

#: Mutually unrelated probes. A pattern that matches all of them carries no
#: information about the answer.
_PROBES = ("?", "no idea ?", "banana split ?", "0 ?")

_INLINE_FLAGS = re.compile(r"\(\?[aiLmsux]+\)")

APP_NAMES: dict[str, str] = {
    "get_task": "evidence",
    "read_baseline": "evidence",
    "read_candidate": "evidence",
    "submit_preference": "submit",
}

# ------------------------------------------------------------------ prompting
#: Role framing, prefixed to ``WORKSPACE_AGENT_TOOL_CONTRACT`` by the runner.
#: This is the anti-sycophancy and anti-position-bias contract; it is the part of
#: this module most likely to change a measured result, so it is explicit rather
#: than terse.
JUDGE_INSTRUCTIONS = """\
YOUR ROLE

You are an impartial evaluator comparing two agent trajectories for the SAME
task. You are not the author of either one and you gain nothing from either
winning. Your job is to report what the evidence shows, including "no real
difference" when that is the truth.

WHAT YOU ARE NOT ALLOWED TO REWARD

* Do NOT prefer a trajectory because it is labelled the candidate, because it is
  new, or because it appears second. The labels are bookkeeping. The same
  comparison is run again with the two sides swapped, and a constant preference
  for one slot is detected and discarded.
* Do NOT prefer a trajectory for being longer, for using more words, for sounding
  more confident, or for describing its own work more enthusiastically. A
  confident wrong answer is worse than a hedged correct one.
* Do NOT reward stated intent. "I will search for the release date" is worth
  nothing unless the search actually ran and its result was used.
* Do NOT invent an expected answer. If none is provided, you do not know it.

WHAT YOU SHOULD REWARD

* Reaching the required answer, when the expected answer is available to you.
* Actually executing the tools the task needed, and using what came back.
* Verifying a result instead of asserting it, and recovering after a failed step
  instead of abandoning the task.
* Committing a definite answer in the required form. A trajectory that ends
  without answering has not answered.

CALIBRATION

Most pairs differ slightly. Reserve magnitudes near 1.0 for a clear difference in
outcome, use small magnitudes for a marginal difference in process, and use
exactly 0.0 when you cannot tell them apart. Inflating small differences destroys
the ranking signal just as badly as missing large ones.
"""

_PROMPT_TEMPLATE = """\
Write and execute Python code that calls the tools listed below, then report the
exact values they returned.

TASK: compare two trajectories for the same problem and report a signed
preference.

  baseline slot  -- the current reference harness's trajectory
  candidate slot -- a proposed replacement harness's trajectory

These are slot names only. Which trajectory sits in which slot tells you nothing
about quality.

TOOLS

  get_task()        the task text, and the expected answer if one is available
  read_baseline()   the trajectory in the baseline slot
  read_candidate()  the trajectory in the candidate slot
  submit_preference(score=<float>, rationale=<str>)   finalize, exactly ONCE

Read the task and BOTH trajectories before you submit. A verdict formed from one
side is not a comparison.

SCORE -- signed, in [{min_score}, {max_score}], oriented baseline -> candidate

  > 0    the candidate slot is better; +1.0 means decisively better
  = 0    genuinely indistinguishable
  < 0    the candidate slot is worse; -1.0 means decisively worse

{gt_guidance}

HOW TO WEIGH THE EVIDENCE

Judge the whole trajectory, not just the final string.
* A correct answer reached by luck, with no verification, is weaker evidence than
  the same answer reached by checking.
* An answer that was never committed is not an answer, however good the reasoning
  leading up to it looked.
* Ignore length. Ignore how confident either side sounds. Ignore which slot a
  trajectory is in.

RATIONALE

State the concrete difference you observed -- a tool that ran or did not run, a
value that was or was not verified, an answer that was or was not committed. Do
not restate the score in words.

You MUST call submit_preference or your comparison is discarded entirely; it is
not counted as a tie.
"""

_GT_PRESENT = """\
GROUND TRUTH IS AVAILABLE from get_task(). It is supplied as a REGULAR
EXPRESSION that a correct answer must match, not as literal answer text: a
trajectory that echoes the pattern itself has not answered. Weigh outcome first:
a trajectory whose committed answer satisfies the pattern is better than one
whose answer does not. Use process quality to break ties between two trajectories
with the same outcome.
"""

_GT_ABSENT = """\
NO GROUND TRUTH IS AVAILABLE for this task. You do not know the correct answer
and you must not invent one, guess one, or assume the more plausible-sounding
answer is right. Compare on process only: which trajectory reasoned more soundly,
actually executed the tools it needed, used what those tools returned, verified
its own result, and committed a definite answer. If both processes are equally
sound, the correct score is 0.0 even if the two answers differ.
"""


# ------------------------------------------------------------------ contracts
@dataclass(frozen=True, slots=True)
class PreferenceVerdict:
    """One signed pairwise comparison.

    ``score`` is oriented ``baseline -> candidate``. ``available`` is the only
    gate a consumer needs: an unavailable verdict must be excluded from averages,
    never folded in as a tie.
    """

    task_id: str
    score: float = 0.0
    winner: str = WINNER_UNAVAILABLE
    rationale: str = ""
    gt_available: bool = False
    available: bool = False
    error: str = ""
    status: str = STATUS_UNAVAILABLE
    orientation: str = ORIENTATION_FORWARD
    #: ``(forward + reversed) / 2`` -- the slot preference that survived the swap.
    #: Only meaningful for a symmetric comparison; 0.0 otherwise.
    position_bias: float = 0.0
    #: How many agent invocations backed this verdict (1 forward, 2 symmetric).
    comparisons: int = 1
    #: True when the judge read both slots before submitting.
    inspected_both: bool = False
    #: Our own tool-call ledger, in order. Narration is not evidence.
    tools_called: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreferenceSummary:
    """Aggregate of many verdicts for one candidate.

    ``decided`` is False when nothing usable came back; a caller must not read
    ``mean_score`` as "no improvement" in that case.
    """

    mean_score: float = 0.0
    available: int = 0
    unavailable: int = 0
    candidate_wins: int = 0
    baseline_wins: int = 0
    ties: int = 0
    mean_position_bias: float = 0.0

    @property
    def decided(self) -> bool:
        """True when at least one usable verdict backs ``mean_score``."""
        return self.available > 0


# ------------------------------------------------------------------ ground truth
def _strip_inline_flags(pattern: str) -> str:
    return _INLINE_FLAGS.sub("", pattern)


def is_placeholder_regex(pattern: str) -> bool:
    """True when ``pattern`` is a vacuous stub rather than real ground truth.

    Two independent checks, because the dataset stub appears with and without
    inline flags and a future split may use a different vacuous form:
    a normalised-core match, and a probe test where a pattern that matches four
    mutually unrelated strings is carrying no answer.
    """
    raw = (pattern or "").strip()
    if not raw or raw in PLACEHOLDER_REGEXES:
        return True
    core = _strip_inline_flags(raw).strip()
    if core in _VACUOUS_CORES:
        return True
    try:
        compiled = re.compile(raw)
    except re.error:
        # An uncompilable pattern is not usable ground truth either.
        return True
    return all(compiled.search(probe) for probe in _PROBES)


def _expected_answer(task: EvolutionTask) -> tuple[str, str]:
    """Return ``(value, kind)`` usable ground truth, or ``("", "")``.

    ``kind`` is reported to the judge so a regex is never presented as literal
    answer text.
    """
    for key, kind in (
        ("expected_regex", "regex"),
        ("expected_pattern", "regex"),
        ("regex", "regex"),
        ("expected_substring", "substring"),
        ("expected_answer", "literal"),
    ):
        raw = task.expected_contract.get(key)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value:
            continue
        if kind == "regex" and is_placeholder_regex(value):
            continue
        if kind != "regex" and value in PLACEHOLDER_REGEXES:
            continue
        return value, kind
    return "", ""


# ------------------------------------------------------------------ workspace
def _render_trace(trace: ExecutionTrace) -> str:
    return json.dumps(
        {
            "final_output": trace.final_output,
            "status": trace.status,
            "event_count": len(trace.events),
            "events": [
                {
                    "kind": event.kind,
                    "actor_id": event.actor_id,
                    "payload": dict(event.payload),
                }
                for event in trace.events
            ],
        },
        default=str,
    )


def _build_callables(
    task: EvolutionTask,
    baseline_slot: ExecutionTrace,
    candidate_slot: ExecutionTrace,
    expected: str,
    expected_kind: str,
    baseline_summary: str,
    candidate_summary: str,
    plan: dict,
) -> dict[str, Callable[..., str]]:
    """Build the judge's tools.

    Every tool has a docstring AND a real typed signature: LangChain's ``@tool``
    raises without a docstring and builds an EMPTY args schema without a
    signature, silently telling the model every tool takes no arguments.
    """

    def get_task() -> str:
        """Return the task text, and the expected answer if one is available."""
        payload: dict[str, object] = {
            "task_id": task.task_id,
            "input": task.input_text,
            "ground_truth_available": bool(expected),
        }
        if expected:
            payload["expected_answer"] = expected
            payload["expected_answer_kind"] = expected_kind
        return json.dumps(payload, default=str)

    def read_baseline() -> str:
        """Return the trajectory in the baseline slot."""
        plan["read_baseline"] = True
        payload = json.loads(_render_trace(baseline_slot))
        if baseline_summary:
            payload["harness_summary"] = baseline_summary
        return json.dumps(payload, default=str)

    def read_candidate() -> str:
        """Return the trajectory in the candidate slot."""
        plan["read_candidate"] = True
        payload = json.loads(_render_trace(candidate_slot))
        if candidate_summary:
            payload["harness_summary"] = candidate_summary
        return json.dumps(payload, default=str)

    def submit_preference(score: float, rationale: str = "") -> str:
        """Finalize the signed preference in [-1, 1]. Call exactly once."""
        if "score" in plan:
            return json.dumps(
                {
                    "status": "rejected",
                    "reason": "a preference was already submitted; the first "
                    "verdict stands",
                }
            )
        plan["score"] = score
        plan["rationale"] = rationale
        return json.dumps({"status": "ok"})

    return {
        "get_task": get_task,
        "read_baseline": read_baseline,
        "read_candidate": read_candidate,
        "submit_preference": submit_preference,
    }


def _winner_for(score: float) -> str:
    if score > 0:
        return WINNER_CANDIDATE
    if score < 0:
        return WINNER_BASELINE
    return WINNER_TIE


# ------------------------------------------------------------------ judge
@dataclass(slots=True)
class PreferenceJudge:
    """Interface B signed pairwise judge, shared by the RHO and genetic stages.

    ``agent_factory`` is the test seam: ``(callables, prompt) -> answer``. When
    ``None`` a real ``CugaAgent`` is constructed by the shared runner.
    """

    agent_factory: Callable[[dict, str], str] | None = None
    app_names: Mapping[str, str] = field(default_factory=lambda: dict(APP_NAMES))
    skills_dir: Path | None = None

    # ------------------------------------------------------------ one direction
    def compare(
        self,
        task: EvolutionTask,
        baseline: ExecutionTrace,
        candidate: ExecutionTrace,
        *,
        baseline_summary: str = "",
        candidate_summary: str = "",
        orientation: str = ORIENTATION_FORWARD,
    ) -> PreferenceVerdict:
        """Compare two trajectories once, returning failure as an unavailable verdict.

        ``baseline_summary`` / ``candidate_summary`` are optional plain-text
        descriptions of the two harnesses (for example an optimizer candidate's
        change summary). They are passed straight through to the judge; this
        module deliberately does not import any candidate type.
        """
        expected, expected_kind = _expected_answer(task)
        gt_available = bool(expected)
        plan: dict = {}
        callables = _build_callables(
            task,
            baseline,
            candidate,
            expected,
            expected_kind,
            baseline_summary,
            candidate_summary,
            plan,
        )
        prompt = _PROMPT_TEMPLATE.format(
            min_score=MIN_SCORE,
            max_score=MAX_SCORE,
            gt_guidance=_GT_PRESENT if gt_available else _GT_ABSENT,
        )

        run = run_workspace_agent(
            callables,
            prompt,
            app_names=self.app_names,
            skills_dir=self.skills_dir,
            special_instructions=JUDGE_INSTRUCTIONS,
            agent_factory=self.agent_factory,
        )

        inspected_both = bool(plan.get("read_baseline")) and bool(
            plan.get("read_candidate")
        )

        def unavailable(status: str, error: str) -> PreferenceVerdict:
            return PreferenceVerdict(
                task_id=task.task_id,
                gt_available=gt_available,
                status=status,
                error=error,
                orientation=orientation,
                inspected_both=inspected_both,
                tools_called=run.tools_called,
            )

        if not run.ok:
            return unavailable(STATUS_UNAVAILABLE, run.error)
        if run.no_tool_call:
            return unavailable(
                STATUS_NO_TOOL_CALL,
                "judge narrated a preference without executing any tool; discarded",
            )
        if "score" not in plan:
            return unavailable(
                STATUS_NO_SUBMIT,
                "judge never called submit_preference; comparison discarded, "
                "not counted as a tie",
            )

        raw = plan["score"]
        if isinstance(raw, bool):
            return unavailable(
                STATUS_INVALID_SCORE, f"score must be a signed number, got {raw!r}"
            )
        try:
            score = float(raw)
        except (TypeError, ValueError):
            return unavailable(
                STATUS_INVALID_SCORE, f"score is not a number: {raw!r}"
            )
        if score != score or score in (float("inf"), float("-inf")):
            return unavailable(STATUS_INVALID_SCORE, f"score is not finite: {raw!r}")
        if not MIN_SCORE <= score <= MAX_SCORE:
            return unavailable(
                STATUS_INVALID_SCORE,
                f"score {score} outside [{MIN_SCORE}, {MAX_SCORE}]",
            )

        return PreferenceVerdict(
            task_id=task.task_id,
            score=score,
            winner=_winner_for(score),
            rationale=str(plan.get("rationale") or ""),
            gt_available=gt_available,
            available=True,
            status=STATUS_OK,
            orientation=orientation,
            inspected_both=inspected_both,
            tools_called=run.tools_called,
        )

    # ------------------------------------------------------------- both directions
    def compare_symmetric(
        self,
        task: EvolutionTask,
        baseline: ExecutionTrace,
        candidate: ExecutionTrace,
        *,
        baseline_summary: str = "",
        candidate_summary: str = "",
    ) -> PreferenceVerdict:
        """Compare in both slot orders and cancel the judge's slot bias.

        Position bias is the dominant systematic error of an LLM preference
        judge: shown the same pair twice with the sides swapped, a biased judge
        gives the same sign twice. The antisymmetric part
        ``(forward - reversed) / 2`` keeps only the content-driven preference; the
        symmetric part ``(forward + reversed) / 2`` is the bias itself, reported
        as ``position_bias`` so it can be audited rather than absorbed.

        Both directions must succeed. Averaging one direction with a missing one
        would reintroduce exactly the bias this method exists to remove.
        """
        forward = self.compare(
            task,
            baseline,
            candidate,
            baseline_summary=baseline_summary,
            candidate_summary=candidate_summary,
            orientation=ORIENTATION_FORWARD,
        )
        if not forward.available:
            return PreferenceVerdict(
                task_id=task.task_id,
                gt_available=forward.gt_available,
                status=forward.status,
                error=f"forward pass: {forward.error}",
                orientation=ORIENTATION_SYMMETRIC,
                comparisons=1,
                inspected_both=forward.inspected_both,
                tools_called=forward.tools_called,
            )

        # Slots swapped: the candidate trajectory now occupies the baseline slot,
        # so this score is oriented candidate -> baseline.
        reversed_verdict = self.compare(
            task,
            candidate,
            baseline,
            baseline_summary=candidate_summary,
            candidate_summary=baseline_summary,
            orientation=ORIENTATION_REVERSED,
        )
        if not reversed_verdict.available:
            return PreferenceVerdict(
                task_id=task.task_id,
                gt_available=forward.gt_available,
                status=reversed_verdict.status,
                error=f"reversed pass: {reversed_verdict.error}",
                orientation=ORIENTATION_SYMMETRIC,
                comparisons=2,
                inspected_both=forward.inspected_both
                and reversed_verdict.inspected_both,
                tools_called=forward.tools_called + reversed_verdict.tools_called,
            )

        score = round((forward.score - reversed_verdict.score) / 2.0, _ROUND)
        bias = round((forward.score + reversed_verdict.score) / 2.0, _ROUND)
        return PreferenceVerdict(
            task_id=task.task_id,
            score=score,
            winner=_winner_for(score),
            rationale=(
                f"forward (baseline->candidate, {forward.score:+}): "
                f"{forward.rationale}\n"
                f"reversed (candidate->baseline, {reversed_verdict.score:+}): "
                f"{reversed_verdict.rationale}"
            ),
            gt_available=forward.gt_available,
            available=True,
            status=STATUS_OK,
            orientation=ORIENTATION_SYMMETRIC,
            position_bias=bias,
            comparisons=2,
            inspected_both=forward.inspected_both and reversed_verdict.inspected_both,
            tools_called=forward.tools_called + reversed_verdict.tools_called,
        )


# ------------------------------------------------------------------ aggregation
def aggregate_preferences(
    verdicts: Iterable[PreferenceVerdict] | Sequence[PreferenceVerdict],
) -> PreferenceSummary:
    """Average only the usable verdicts.

    Unavailable verdicts are EXCLUDED, not counted as ties: folding a failed
    judge call in as 0.0 drags every candidate toward its baseline and makes a
    broken judge look like a loop that produces no improvement.
    """
    items = list(verdicts)
    usable = [v for v in items if v.available]
    unavailable = len(items) - len(usable)
    if not usable:
        return PreferenceSummary(unavailable=unavailable)
    return PreferenceSummary(
        mean_score=round(sum(v.score for v in usable) / len(usable), _ROUND),
        available=len(usable),
        unavailable=unavailable,
        candidate_wins=sum(1 for v in usable if v.winner == WINNER_CANDIDATE),
        baseline_wins=sum(1 for v in usable if v.winner == WINNER_BASELINE),
        ties=sum(1 for v in usable if v.winner == WINNER_TIE),
        mean_position_bias=round(
            sum(v.position_bias for v in usable) / len(usable), _ROUND
        ),
    )
