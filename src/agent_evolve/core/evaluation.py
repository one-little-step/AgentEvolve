"""Four-category validation planning and acceptance decision rules.

The evaluation boundary constructs a :class:`ValidationPlan` with four always
present categories (origin, worked, regression, generalization) and applies the
acceptance rule that produces an :class:`AcceptanceDecision` from a
:class:`~agent_evolve.core.contracts.ValidationResult`.

Generalization is deferred by default: the plan records ``generalization_unverified``
rather than silently dropping the category or treating missing probes as passes.
An ``unavailable`` case is never counted as a pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agent_evolve.core.contracts import ValidationCase, ValidationResult


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
