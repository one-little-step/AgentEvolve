"""Four-category validation planning, acceptance rules, and rollout scoring.

Two responsibilities live here, both agent-neutral:

1. **Validation planning and acceptance.** :class:`ValidationPlan` carries four
   always-present categories (origin, worked, regression, generalization) and
   :func:`decide_acceptance` turns a :class:`ValidationResult` into an
   :class:`AcceptanceDecision`. Generalization is deferred by default: the plan
   records ``generalization_unverified`` rather than silently dropping the
   category or treating missing probes as passes. An ``unavailable`` case is
   never counted as a pass.

2. **Rollout scoring.** A :class:`Scorer` turns one ``(task, trace)`` into a
   :class:`RolloutScore`. Two implementations ship here:

   * :class:`ContractScorer` -- the pre-existing behaviour, measuring
     ``trace.final_output`` against ``task.expected_contract``.
   * :class:`BenchmarkScorer` -- benchmark-driven, so the headline number comes
     from the benchmark's own named grader instead of a contract embedded in the
     task.

   Every :class:`RolloutScore` carries its ``grader_name``. Without it a
   reported pass rate is ambiguous: two graders on the same benchmark disagree
   (Gaia's ``expected_regex`` and ``recorded_llm_verdict`` do), so a number that
   cannot name its grader cannot be compared to anything.

   ``scorable`` is the load-bearing field. A rollout that produced no answer --
   a crashed harness, a timeout, an empty output -- is **not** a wrong answer,
   and :func:`tally_scores` excludes it from the denominator. Scoring a broken
   harness as 0.0 would manufacture a self-improvement delta out of an outage,
   which is the single worst failure mode this pipeline can have.

Why this module may import ``agent_evolve.benchmarks.base``
----------------------------------------------------------
``benchmarks.base`` is agent-neutral (its only imports are stdlib) and defines
the ``Benchmark`` Protocol the core consumes. It is not an agent
implementation, so the ``core/`` boundary rule -- never import ``cuga``, never
import ``agent_evolve.adapters`` -- is intact. Benchmark *implementations*
(``benchmarks.gaia``) are never imported here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from agent_evolve.benchmarks.base import (
    Benchmark,
    GradingUnavailableError,
    UnknownGraderError,
)
from agent_evolve.core.analyzer import contract_score
from agent_evolve.core.blame import CausalAnalysis
from agent_evolve.core.contracts import (
    EvolutionTask,
    ExecutionTrace,
    ValidationCase,
    ValidationResult,
)
from agent_evolve.core.non_answer import classify_non_answer

#: Trace statuses that mean "this rollout produced an answer". Deliberately a
#: whitelist rather than a blacklist of failure words: an unrecognised status is
#: an unknown outcome, and treating an unknown outcome as an answer is exactly
#: how a broken harness gets scored as a wrong answer.
ANSWERED_TRACE_STATUSES: frozenset[str] = frozenset({"success", "ok", "completed"})

#: The grader name :class:`ContractScorer` reports. Named, not blank, so a run
#: scored against task contracts is never mistaken for a benchmark-graded run.
CONTRACT_GRADER_NAME = "expected_contract"



@dataclass(frozen=True)
class AcceptanceDecision:
    """Final accept/reject decision for one validated edit."""

    decision: str
    reason: str


@dataclass(frozen=True)
class CaseSummary:
    """Counts of passed, failed, and unavailable validation cases."""

    passed: int
    failed: int
    unavailable: int


@dataclass(frozen=True)
class ValidationPlan:
    """Four-category validation plan.

    ``origin_cases`` carry known evidence; ``worked_cases``, ``regression_cases``,
    and ``generalization_cases`` are planned-but-unexecuted cases whose evidence is
    not yet collected (outcome ``unavailable``). ``generalization_status`` records
    whether deferred generalization has been verified.
    """

    origin_cases: tuple[ValidationCase, ...]
    worked_cases: tuple[ValidationCase, ...]
    regression_cases: tuple[ValidationCase, ...]
    generalization_cases: tuple[ValidationCase, ...]
    generalization_status: str


def decide_acceptance(result: ValidationResult) -> AcceptanceDecision:
    """Apply the acceptance rule to a validated result.

    A protected-floor violation rejects regardless of gains; otherwise acceptance
    requires positive primary gain and positive weighted net gain.
    """
    if result.protected_floor_outcome == "violated":
        return AcceptanceDecision("reject", "protected_floor_violated")
    if result.primary_gain <= 0.0:
        return AcceptanceDecision("reject", "primary_gain_not_positive")
    if result.weighted_net_gain <= 0.0:
        return AcceptanceDecision("reject", "weighted_net_gain_not_positive")
    return AcceptanceDecision("accept", "validated_gain")


def summarize_cases(cases: tuple[ValidationCase, ...]) -> CaseSummary:
    """Count passed, failed, and unavailable cases.

    An ``unavailable`` case is never counted as passed.
    """
    passed = sum(1 for case in cases if case.outcome == "passed")
    failed = sum(1 for case in cases if case.outcome == "failed")
    unavailable = sum(1 for case in cases if case.outcome == "unavailable")
    return CaseSummary(passed=passed, failed=failed, unavailable=unavailable)


def _as_cases(cases: ValidationCase | Sequence[ValidationCase]) -> tuple[ValidationCase, ...]:
    if isinstance(cases, ValidationCase):
        return (cases,)
    return tuple(cases)


def _planned_cases(kind: str, artifacts: Sequence[str]) -> tuple[ValidationCase, ...]:
    return tuple(
        ValidationCase(case_id=f"{kind}:{artifact}", outcome="unavailable")
        for artifact in artifacts
    )


def build_validation_plan(
    origin_cases: ValidationCase | Sequence[ValidationCase],
    *,
    written_artifacts: Sequence[str] = (),
    probe_mode: str = "deferred",
    budget_fraction: float = 0.15,
) -> ValidationPlan:
    """Build a four-category plan with explicit generalization status.

    Deferred mode (default) creates zero executed generalization cases while keeping
    the category present and recording ``generalization_unverified``. Other modes
    record ``unavailable`` (no probe capacity) or ``verified`` (probes planned under
    ``budget_fraction``).
    """
    origin = _as_cases(origin_cases)
    worked = _planned_cases("worked", written_artifacts)
    regression = _planned_cases("regression", written_artifacts)

    if probe_mode == "deferred":
        generalization: tuple[ValidationCase, ...] = ()
        generalization_status = "generalization_unverified"
    elif probe_mode == "verified":
        probe_limit = max(1, round(len(written_artifacts) * budget_fraction))
        generalization = tuple(
            ValidationCase(case_id=f"generalization:{artifact}", outcome="unavailable")
            for artifact in written_artifacts[:probe_limit]
        )
        generalization_status = "verified" if generalization else "unavailable"
    else:
        generalization = ()
        generalization_status = "unavailable"

    return ValidationPlan(
        origin_cases=origin,
        worked_cases=worked,
        regression_cases=regression,
        generalization_cases=generalization,
        generalization_status=generalization_status,
    )


# --------------------------------------------------------------------------- #
# Rollout scoring
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RolloutScore:
    """One measurement of one rollout, by one explicitly named grader.

    ``scorable=False`` means *no measurement exists*: the rollout produced no
    answer, or the grader had no material for it. Such a result carries
    ``score=0.0`` only because the field must hold a number; it is never a
    denominator entry and never a failing verdict. ``reason`` says which.
    """

    task_id: str
    grader_name: str
    score: float
    scorable: bool
    passed: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("RolloutScore.task_id is required")
        if not self.grader_name:
            raise ValueError(
                "RolloutScore.grader_name is required: a pass rate that cannot "
                "name the grader that produced it is not a measurement"
            )
        score = float(self.score)
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"RolloutScore.score must be within [0, 1]; got {score!r}")
        object.__setattr__(self, "score", score)
        if self.scorable:
            if not self.reason:
                object.__setattr__(self, "reason", "graded")
        else:
            if not self.reason:
                raise ValueError(
                    "an unscorable result requires a reason: 'not measured' and "
                    "'measured as failing' are different facts"
                )
            if self.passed:
                raise ValueError("an unscorable result cannot be a pass")
            if score != 0.0:
                raise ValueError("an unscorable result must not carry a score")


@dataclass(frozen=True, slots=True)
class ObservedRollout:
    """One rollout plus what was measured and diagnosed about it.

    ``trace is None`` means the rollout never produced a trace at all (the
    executor raised, or the task timed out). ``analysis is None`` means no
    analyzer examined it -- either because it produced no answer, so there is no
    trajectory outcome to diagnose and a model call on it would be paid for
    nothing, or because the analyzer itself failed (see ``error``).

    A rollout with ``scorable=False`` is never a wrong answer and must never
    reach a score denominator.
    """

    task: EvolutionTask
    trace: ExecutionTrace | None
    score: RolloutScore | None
    analysis: CausalAnalysis | None = None
    error: str = ""

    @property
    def scorable(self) -> bool:
        return self.score is not None and self.score.scorable


@dataclass(frozen=True, slots=True)
class ScoreTally:
    """Aggregate of many :class:`RolloutScore` values, with its denominator.

    Mirrors :class:`agent_evolve.benchmarks.base.GraderStats`: ``pass_rate`` is
    ``None`` when nothing was scored, and :attr:`summary` never states a rate
    without the denominator it came from.
    """

    grader_name: str
    passed: int
    evaluated: int
    attempted: int
    unscorable: int
    unscorable_task_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.grader_name:
            raise ValueError("ScoreTally.grader_name is required")
        object.__setattr__(
            self, "unscorable_task_ids", tuple(self.unscorable_task_ids)
        )

    @property
    def pass_rate(self) -> float | None:
        """Passes over rollouts actually scored, or ``None`` when none were."""
        if self.evaluated == 0:
            return None
        return self.passed / self.evaluated

    @property
    def is_partial(self) -> bool:
        """True whenever a bare pass rate would misrepresent the run."""
        return self.evaluated < self.attempted

    @property
    def summary(self) -> str:
        rate = self.pass_rate
        rate_text = "n/a" if rate is None else f"{rate * 100:.2f}%"
        return (
            f"grader={self.grader_name} {self.passed}/{self.evaluated} scored "
            f"pass_rate={rate_text} attempted={self.attempted} "
            f"unscorable={self.unscorable}"
        )


def tally_scores(
    scores: Sequence[RolloutScore], *, grader_name: str
) -> ScoreTally:
    """Aggregate scores, excluding every unscorable rollout from the denominator.

    This is the one place the denominator is computed, so there is exactly one
    definition of it: ``evaluated`` counts scorable results only.
    """
    scorable = [s for s in scores if s.scorable]
    unscorable = [s for s in scores if not s.scorable]
    return ScoreTally(
        grader_name=grader_name,
        passed=sum(1 for s in scorable if s.passed),
        evaluated=len(scorable),
        attempted=len(scores),
        unscorable=len(unscorable),
        unscorable_task_ids=tuple(s.task_id for s in unscorable),
    )


class Scorer(Protocol):
    """Measures one rollout. Named graders only; never a silent default.

    ``grader_name`` is a read-only property rather than a mutable attribute so a
    frozen implementation satisfies it, and so the grader cannot be swapped
    mid-run: a pass rate whose grader changed halfway has no single denominator.
    """

    @property
    def grader_name(self) -> str: ...

    def score_rollout(
        self, task: EvolutionTask, trace: ExecutionTrace
    ) -> RolloutScore: ...


@dataclass(frozen=True, slots=True)
class RolloutOutcome:
    """One executed rollout, before it is scored or diagnosed.

    ``trace is None`` with a non-empty ``error`` is the "no answer" case a
    batch executor reports as data rather than raising: a crashed agent, a
    timeout, a worker that never started. It is not a wrong answer.
    """

    task: EvolutionTask
    trace: ExecutionTrace | None
    error: str = ""

    def __post_init__(self) -> None:
        if self.trace is None and not self.error:
            raise ValueError(
                "a rollout with no trace requires an error: 'no rollout' and "
                "'a rollout that answered nothing' are different facts"
            )
        if self.trace is not None and self.error:
            raise ValueError("a rollout with a trace must not carry an error")


class RolloutBatch(Protocol):
    """Executes one candidate version against many tasks.

    The seam that lets a real run replace the adapter's one-rollout-at-a-time
    path with a benchmark runner backed by process-isolated CUGA workers,
    without the orchestrator knowing which it got. Failures come back as
    ``RolloutOutcome(trace=None, error=...)``, never as exceptions, so one
    broken task cannot discard the rest of the batch.
    """

    def run_rollouts(
        self, version: str, tasks: Sequence[EvolutionTask], *, prefix: str
    ) -> tuple[RolloutOutcome, ...]: ...


def _answer_or_reason(
    trace: ExecutionTrace, *, question: str | None = None
) -> tuple[str | None, str]:
    """The rollout's answer, or the reason there is not one.

    Three things are "no answer" and none of them is a wrong answer:

    * a status outside :data:`ANSWERED_TRACE_STATUSES`;
    * an empty ``final_output``;
    * output that is not an answer -- an apology, a stated inability, pure
      narration, or an echo of the task (see
      :mod:`agent_evolve.core.non_answer`).

    The third case is the one a grader would otherwise silently record as a
    failure-to-match, inflating the denominator with rollouts that committed to
    no claim.
    """
    status = str(trace.status or "").strip().lower()
    if status not in ANSWERED_TRACE_STATUSES:
        return None, f"rollout produced no answer: trace status {trace.status!r}"
    verdict = classify_non_answer(trace.final_output, question=question)
    if verdict.is_non_answer:
        return None, f"rollout produced no answer: {verdict.reason}"
    return trace.final_output, ""


@dataclass(frozen=True, slots=True)
class ContractScorer:
    """Scores a rollout against ``task.expected_contract``.

    The pre-existing behaviour, factored into a named scorer so a caller can
    swap it for :class:`BenchmarkScorer` without the call site changing shape.
    Kept because much of the offline test suite drives evolution through task
    contracts and has no benchmark.
    """

    _grader_name: str = CONTRACT_GRADER_NAME

    @property
    def grader_name(self) -> str:
        return self._grader_name

    def score_rollout(
        self, task: EvolutionTask, trace: ExecutionTrace
    ) -> RolloutScore:
        answer, reason = _answer_or_reason(trace, question=task.input_text or None)
        if answer is None:
            return RolloutScore(
                task_id=task.task_id,
                grader_name=self.grader_name,
                score=0.0,
                scorable=False,
                reason=reason,
            )
        score = contract_score(task, trace)
        return RolloutScore(
            task_id=task.task_id,
            grader_name=self.grader_name,
            score=score,
            scorable=True,
            passed=score >= 1.0,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkScorer:
    """Scores a rollout with one named grader of a :class:`Benchmark`.

    The grader is validated at construction: a typo'd grader name must fail
    before the first billed rollout, not after forty of them.

    ``GradingUnavailableError`` is honoured as "no measurement" rather than
    converted into a failing score. On Gaia, ``recorded_llm_verdict`` raises it
    for every live answer -- so a run that silently turned it into 0.0 would
    report a 0% pass rate that measured nothing at all.
    """

    benchmark: Benchmark
    grader: str

    def __post_init__(self) -> None:
        if not self.grader:
            raise ValueError(
                "BenchmarkScorer.grader is required: the grader is never "
                "silently chosen"
            )
        available = tuple(self.benchmark.graders())
        if self.grader not in available:
            raise UnknownGraderError(
                f"unknown grader {self.grader!r} for benchmark "
                f"{getattr(self.benchmark, 'name', type(self.benchmark).__name__)!r}; "
                f"available: {available}"
            )

    @property
    def grader_name(self) -> str:
        return self.grader

    def score_rollout(
        self, task: EvolutionTask, trace: ExecutionTrace
    ) -> RolloutScore:
        answer, reason = _answer_or_reason(trace, question=task.input_text or None)
        if answer is None:
            return RolloutScore(
                task_id=task.task_id,
                grader_name=self.grader,
                score=0.0,
                scorable=False,
                reason=reason,
            )
        try:
            outcome = self.benchmark.score(task.task_id, answer, grader=self.grader)
        except GradingUnavailableError as exc:
            return RolloutScore(
                task_id=task.task_id,
                grader_name=self.grader,
                score=0.0,
                scorable=False,
                reason=f"grader has no measurement for this task/answer: {exc}",
            )
        return RolloutScore(
            task_id=task.task_id,
            grader_name=outcome.grader_name,
            score=outcome.score,
            scorable=True,
            passed=outcome.passed,
        )
