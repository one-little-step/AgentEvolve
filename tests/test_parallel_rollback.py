"""A failed barrier callback must compensate every earlier staged mutation."""
from __future__ import annotations

import pytest

from agent_evolve.core.contracts import ArtifactEdit
from agent_evolve.core.parallel import BatchCoordinator, PoolSnapshot, WorkerResult


def _result(attempt_id: str) -> WorkerResult:
    return WorkerResult(
        attempt_id=attempt_id,
        workspace=None,  # type: ignore[arg-type] - not consumed by the coordinator
        edits=(ArtifactEdit(f"artifact-{attempt_id}", "replace", {}),),
        trace=None,  # type: ignore[arg-type] - not consumed by the coordinator
        attempt=None,  # type: ignore[arg-type] - not consumed by the coordinator
    )


def test_commit_barrier_rolls_back_earlier_callbacks_when_the_third_fails() -> None:
    coordinator = BatchCoordinator(PoolSnapshot(0, (), ()))
    for attempt_id in ("attempt-1", "attempt-2", "attempt-3"):
        coordinator.submit(_result(attempt_id))

    committed: list[str] = []
    rolled_back: list[str] = []

    def commit(result: WorkerResult) -> None:
        if result.attempt_id == "attempt-3":
            raise RuntimeError("injected commit failure")
        committed.append(result.attempt_id)

    def rollback(result: WorkerResult) -> None:
        rolled_back.append(result.attempt_id)
        committed.remove(result.attempt_id)

    with pytest.raises(RuntimeError, match="injected commit failure"):
        coordinator.commit_barrier(commit, on_attempt_rolled_back=rollback)

    assert committed == []
    assert rolled_back == ["attempt-2", "attempt-1"]
    assert not coordinator.is_committed
