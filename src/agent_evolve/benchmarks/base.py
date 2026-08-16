"""Agent-neutral benchmark abstraction.

This module defines a benchmark-agnostic contract that the evolution core can
consume without knowing anything about a specific benchmark's on-disk JSON
schema or its notion of success.

Two structural invariants drive the design:

1. **Ground truth is a secret.** ``BenchmarkTask`` is the only object that is
   safe to hand to an analyzer or an editor. It cannot carry grading material:
   the constructor rejects grading-shaped metadata keys outright, so leakage is
   a construction-time error rather than a reviewer's responsibility.
   ``BenchmarkGrading`` is the scorer-only counterpart and redacts its own
   ``repr()``.

2. **A grader is never implicit.** ``score()`` requires an explicit ``grader``
   name and every ``TaskOutcome`` records which grader produced it. When a
   grader has no material for a task the call raises
   ``GradingUnavailableError`` instead of silently returning a failing score --
   "we could not grade this" and "the agent got it wrong" are different facts
   and conflating them corrupts the noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

__all__ = [
    "GRADING_KEY_DENYLIST",
    "BenchmarkError",
    "Benchmark",
    "BenchmarkGrading",
    "BenchmarkTask",
    "GraderStats",
    "GradingUnavailableError",
    "GraderDelta",
    "LeakageError",
    "RunComparison",
    "RunObservations",
    "RunStatistics",
    "TaskOutcome",
    "UnknownGraderError",
    "UnknownTaskError",
    "compare_runs",
    "compute_run_statistics",
    "outcomes_disagree",
]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class BenchmarkError(Exception):
    """Base class for benchmark abstraction errors."""


class LeakageError(BenchmarkError):
    """Raised when grading material is placed on a task-facing object."""


class UnknownGraderError(BenchmarkError):
    """Raised when an unknown grader name is requested."""


class UnknownTaskError(BenchmarkError):
    """Raised when an unknown task id is requested."""


class GradingUnavailableError(BenchmarkError):
    """Raised when a grader cannot produce a verdict for this task/answer.

    This is deliberately *not* a failing score. A missing pattern, a corrupt
    regex, an evaluation batch that errored out, or a recorded judgment that
    does not apply to the supplied answer all mean "no measurement", which must
    be excluded from the denominator rather than counted as a failure.
    """


# ---------------------------------------------------------------------------
# leakage guard
# ---------------------------------------------------------------------------

#: Metadata / detail keys that may never appear on a task-facing object.
#: Substring matching is used, so ``direct_regex`` is caught by ``regex`` and
#: ``recorded_verdict`` is caught by ``verdict``.
GRADING_KEY_DENYLIST: tuple[str, ...] = (
    "regex",
    "pattern",
    "verdict",
    "answer_span",
    "reason",
    "expected",
    "gold",
    "ground_truth",
    "groundtruth",
    "label",
    "solution",
    "reference_answer",
    "grade",
    "score",
    "correct",
    "pass",
)


def _reject_grading_keys(mapping: Mapping[str, object], *, where: str) -> None:
    for key in mapping:
        lowered = str(key).lower()
        for banned in GRADING_KEY_DENYLIST:
            if banned in lowered:
                raise LeakageError(
                    f"{where} may not carry grading material: key {key!r} "
                    f"matches denylisted token {banned!r}. Grading material "
                    f"belongs on BenchmarkGrading (scorer-only)."
                )


# ---------------------------------------------------------------------------
# task-facing (safe to expose)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkTask:
    """A task as the agent under evolution may see it.

    Safe to expose to rollout, analyzer and editor roles. Carries no grading
    material by construction.
    """

    task_id: str
    question: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("BenchmarkTask.task_id must be non-empty")
        _reject_grading_keys(self.metadata, where="BenchmarkTask.metadata")
        object.__setattr__(self, "metadata", dict(self.metadata))


# ---------------------------------------------------------------------------
# grading material (secret, scorer-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkGrading:
    """Grading material for one task. SCORER-ONLY -- never expose this.

    ``payload`` is intentionally opaque to the core: each benchmark stores
    whatever its graders need. ``repr()`` is redacted so an accidental log
    statement or exception traceback cannot leak the answer key.
    """

    task_id: str
    grader_names: tuple[str, ...]
    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "grader_names", tuple(self.grader_names))

    def __repr__(self) -> str:  # pragma: no cover - exercised via tests
        return (
            f"BenchmarkGrading(task_id={self.task_id!r}, "
            f"grader_names={self.grader_names!r}, payload=<redacted "
            f"{len(self.payload)} field(s)>)"
        )

    __str__ = __repr__

    def material(self, key: str, default: object = None) -> object:
        """Explicit, auditable read of one grading field."""
        return self.payload.get(key, default)


# ---------------------------------------------------------------------------
# outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskOutcome:
    """The result of scoring one answer with one explicitly named grader."""

    task_id: str
    score: float
    passed: bool
    grader_name: str
    detail: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.grader_name:
            raise ValueError(
                "TaskOutcome.grader_name is required: a grader must never be "
                "silently chosen or omitted"
            )
        score = float(self.score)
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"TaskOutcome.score must be within [0, 1]; got {score!r}")
        object.__setattr__(self, "score", score)
        _reject_grading_keys(self.detail, where="TaskOutcome.detail")
        object.__setattr__(self, "detail", dict(self.detail))


def outcomes_disagree(outcomes: Mapping[str, TaskOutcome]) -> bool:
    """True when two graders reached different pass/fail conclusions."""
    verdicts = {outcome.passed for outcome in outcomes.values()}
    return len(verdicts) > 1


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------


@runtime_checkable
class Benchmark(Protocol):
    """Contract every benchmark adapter satisfies."""

    name: str

    def load_tasks(self) -> tuple[BenchmarkTask, ...]:
        """Task-facing records only. Never grading material."""
        ...

    def grading_for(self, task_id: str) -> BenchmarkGrading | None:
        """Scorer-only grading material, or None when the task is unknown."""
        ...

    def score(self, task_id: str, answer: str, *, grader: str) -> TaskOutcome:
        """Score ``answer`` with the explicitly named ``grader``.

        Raises ``UnknownTaskError``, ``UnknownGraderError`` or
        ``GradingUnavailableError`` rather than defaulting to a failing score.
        """
        ...

    def score_all(self, task_id: str, answer: str) -> Mapping[str, TaskOutcome]:
        """Every grader's verdict for one answer, so disagreement is visible."""
        ...

    def graders(self) -> tuple[str, ...]:
        """Names of available graders, in stable order."""
        ...


# ---------------------------------------------------------------------------
# run-level observations and statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunObservations:
    """What a completed run recorded, in benchmark-neutral shape.

    ``answers`` maps task id -> the answer the agent produced during that run.
    An answer is *not* grading material: it is the agent's own output.
    ``failed_eval_batches`` counts grading batches that errored out, which is
    the reason a grader's denominator can be partial.
    """

    run_name: str
    task_ids: tuple[str, ...]
    answers: Mapping[str, str]
    timed_out_task_ids: tuple[str, ...] = ()
    errored_task_ids: tuple[str, ...] = ()
    failed_eval_batches: int = 0
    key_coverage: Mapping[str, object] = field(default_factory=dict)
    config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_ids", tuple(self.task_ids))
        object.__setattr__(self, "answers", dict(self.answers))
        object.__setattr__(self, "timed_out_task_ids", tuple(self.timed_out_task_ids))
        object.__setattr__(self, "errored_task_ids", tuple(self.errored_task_ids))
        object.__setattr__(self, "key_coverage", dict(self.key_coverage))
        object.__setattr__(self, "config", dict(self.config))


@dataclass(frozen=True)
class GraderStats:
    """One grader's pass count over an explicit denominator."""

    grader_name: str
    passed: int
    evaluated: int
    total_tasks: int
    unavailable: int
    unavailable_task_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "unavailable_task_ids", tuple(self.unavailable_task_ids))

    @property
    def pass_rate(self) -> float | None:
        """Pass rate over ``evaluated``, or None when nothing was graded.

        Never falls back to ``total_tasks``: dividing by a denominator the
        grader did not actually cover manufactures a number.
        """
        if self.evaluated <= 0:
            return None
        return self.passed / self.evaluated

    @property
    def is_partial(self) -> bool:
        """True when this grader covered fewer tasks than the run contains."""
        return self.evaluated < self.total_tasks

    @property
    def denominator_label(self) -> str:
        suffix = " PARTIAL" if self.is_partial else ""
        return f"{self.passed}/{self.evaluated} of {self.total_tasks} tasks{suffix}"


@dataclass(frozen=True)
class RunStatistics:
    """Run-level statistics establishing the noise floor for one run."""

    benchmark_name: str
    run_name: str
    task_ids: tuple[str, ...]
    grader_stats: Mapping[str, GraderStats]
    timed_out: int
    errored: int
    failed_eval_batches: int
    key_coverage: Mapping[str, object] = field(default_factory=dict)
    config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_ids", tuple(self.task_ids))
        object.__setattr__(self, "grader_stats", dict(self.grader_stats))
        object.__setattr__(self, "key_coverage", dict(self.key_coverage))
        object.__setattr__(self, "config", dict(self.config))

    @property
    def task_count(self) -> int:
        return len(self.task_ids)


@dataclass(frozen=True)
class GraderDelta:
    """Between-run delta for one grader, with comparability made explicit."""

    grader_name: str
    passed_a: int
    passed_b: int
    evaluated_a: int
    evaluated_b: int
    comparable: bool
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def passed_delta(self) -> int:
        return self.passed_b - self.passed_a

    @property
    def pass_rate_delta(self) -> float | None:
        """None when the two denominators are not comparable."""
        if not self.comparable or self.evaluated_a <= 0 or self.evaluated_b <= 0:
            return None
        return (self.passed_b / self.evaluated_b) - (self.passed_a / self.evaluated_a)


@dataclass(frozen=True)
class RunComparison:
    """Comparison of two runs, refusing to compare mismatched denominators."""

    run_a: str
    run_b: str
    same_task_set: bool
    shared_task_count: int
    deltas: Mapping[str, GraderDelta]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "deltas", dict(self.deltas))
        object.__setattr__(self, "notes", tuple(self.notes))


def compute_run_statistics(
    benchmark: "Benchmark",
    observations: RunObservations,
) -> RunStatistics:
    """Compute per-grader statistics for a completed run.

    Benchmark-agnostic: it replays each recorded answer through every grader the
    benchmark declares. A grader that raises ``GradingUnavailableError`` for a
    task increments ``unavailable`` and is excluded from that grader's
    denominator -- an ungraded task is not a failed task.
    """
    grader_stats: dict[str, GraderStats] = {}
    total = len(observations.task_ids)
    for grader in benchmark.graders():
        passed = 0
        evaluated = 0
        unavailable: list[str] = []
        for task_id in observations.task_ids:
            answer = observations.answers.get(task_id, "")
            try:
                outcome = benchmark.score(task_id, answer, grader=grader)
            except GradingUnavailableError:
                unavailable.append(task_id)
                continue
            evaluated += 1
            if outcome.passed:
                passed += 1
        grader_stats[grader] = GraderStats(
            grader_name=grader,
            passed=passed,
            evaluated=evaluated,
            total_tasks=total,
            unavailable=len(unavailable),
            unavailable_task_ids=tuple(unavailable),
        )

    return RunStatistics(
        benchmark_name=benchmark.name,
        run_name=observations.run_name,
        task_ids=observations.task_ids,
        grader_stats=grader_stats,
        timed_out=len(observations.timed_out_task_ids),
        errored=len(observations.errored_task_ids),
        failed_eval_batches=observations.failed_eval_batches,
        key_coverage=observations.key_coverage,
        config=observations.config,
    )


def compare_runs(a: RunStatistics, b: RunStatistics) -> RunComparison:
    """Compare two runs, per grader, with explicit denominator guards.

    A grader's delta is marked non-comparable when the two runs graded
    different numbers of tasks -- e.g. a run whose evaluation batch errored out
    covers 22 tasks and must never be compared head-to-head against a run that
    covered all 42.
    """
    ids_a, ids_b = set(a.task_ids), set(b.task_ids)
    same_task_set = ids_a == ids_b
    notes: list[str] = []
    if not same_task_set:
        notes.append(
            f"task set differs: {len(ids_a - ids_b)} only in {a.run_name}, "
            f"{len(ids_b - ids_a)} only in {b.run_name}; "
            f"{len(ids_a & ids_b)} shared"
        )

    deltas: dict[str, GraderDelta] = {}
    for grader in sorted(set(a.grader_stats) | set(b.grader_stats)):
        sa = a.grader_stats.get(grader)
        sb = b.grader_stats.get(grader)
        grader_notes: list[str] = []
        if sa is None or sb is None:
            missing = a.run_name if sa is None else b.run_name
            deltas[grader] = GraderDelta(
                grader_name=grader,
                passed_a=sa.passed if sa else 0,
                passed_b=sb.passed if sb else 0,
                evaluated_a=sa.evaluated if sa else 0,
                evaluated_b=sb.evaluated if sb else 0,
                comparable=False,
                notes=(f"grader absent from run {missing}",),
            )
            continue

        comparable = True
        if sa.evaluated != sb.evaluated:
            comparable = False
            grader_notes.append(
                f"denominator mismatch: {sa.evaluated} vs {sb.evaluated} "
                f"evaluated -- pass rates are not comparable"
            )
        if sa.evaluated == 0 or sb.evaluated == 0:
            comparable = False
            grader_notes.append("at least one run graded zero tasks")
        if not same_task_set:
            comparable = False
            grader_notes.append("task sets differ")

        deltas[grader] = GraderDelta(
            grader_name=grader,
            passed_a=sa.passed,
            passed_b=sb.passed,
            evaluated_a=sa.evaluated,
            evaluated_b=sb.evaluated,
            comparable=comparable,
            notes=tuple(grader_notes),
        )

    return RunComparison(
        run_a=a.run_name,
        run_b=b.run_name,
        same_task_set=same_task_set,
        shared_task_count=len(ids_a & ids_b),
        deltas=deltas,
        notes=tuple(notes),
    )
