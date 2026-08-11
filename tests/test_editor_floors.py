"""Protected floors reject positive-gain candidates when a named cell regresses."""
from __future__ import annotations

from agent_evolve.core.editor import ProtectedFloor, ValidationKind, ValidationResult, floors_violated


def test_floors_violated_matches_task_and_mechanism_cluster() -> None:
    floor = ProtectedFloor("task-1", "cluster-1", 0.8)
    results = (
        ValidationResult(ValidationKind.REGRESSION, "task-1", 0.5, "trace-1", False, "cluster-1"),
        ValidationResult(ValidationKind.REGRESSION, "task-1", 1.0, "trace-2", True, "cluster-2"),
    )

    assert floors_violated(results, (floor,)) == (floor,)


def test_floors_violated_does_not_cross_apply_clusters() -> None:
    floor = ProtectedFloor("task-1", "cluster-1", 0.8)
    results = (
        ValidationResult(ValidationKind.REGRESSION, "task-1", 1.0, "trace-2", True, "cluster-2"),
    )

    assert floors_violated(results, (floor,)) == ()
