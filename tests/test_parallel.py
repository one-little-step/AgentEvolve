"""Tests for the snapshot/lease manager and parallel batch coordinator."""
from __future__ import annotations

import threading
import pytest

from pathlib import Path

from agent_evolve.core.contracts import (
    ArtifactEdit,
    CandidateWorkspace,
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)
from agent_evolve.core.memory import AttemptStatus, EditAttempt
from agent_evolve.core.parallel import (
    BatchCoordinator,
    LeaseConflict,
    PoolSnapshot,
    SnapshotLeaseManager,
    WorkerResult,
    snapshot_pool,
)
from agent_evolve.core.pool import PersistentPool
from agent_evolve.core.contracts import EvolutionCandidate


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _ws(attempt: str, parent: str = "base-v0") -> CandidateWorkspace:
    return CandidateWorkspace(
        attempt_id=attempt,
        version=f"{parent}+{attempt}",
        path=Path("."),
        parent_version=parent,
    )


def _edit(aid: str) -> ArtifactEdit:
    return ArtifactEdit(artifact_id=aid, operation="replace", payload={"content": "x"})


def _trace(attempt_id: str) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=f"t-{attempt_id}",
        candidate_id=attempt_id,
        task_id="task-1",
        events=(TraceEvent("e1", "state", None, None, {}),),
        final_output="ok",
        status="success",
    )


def _attempt(aid: str) -> EditAttempt:
    return EditAttempt(
        attempt_id=aid,
        candidate_id=aid,
        issue_id="issue-1",
        artifact_ids=("skills/a",),
        operation="replace",
        sanitized_reasoning="r",
        sanitized_diff={},
        status=AttemptStatus.ACCEPTED,
    )


def _result(aid: str, artifacts: tuple[str, ...] = ("skills/a",)) -> WorkerResult:
    return WorkerResult(
        attempt_id=aid,
        workspace=_ws(aid),
        edits=tuple(_edit(a) for a in artifacts),
        trace=_trace(aid),
        attempt=_attempt(aid),
    )


def _pool_with_base() -> PersistentPool:
    p = PersistentPool()
    p.add_base(EvolutionCandidate(
        candidate_id="base", version="base-v0", artifact_hashes={}
    ))
    return p


# ---------------------------------------------------------------------- #
# PoolSnapshot
# ---------------------------------------------------------------------- #
def test_snapshot_pool_captures_ids_and_frontier():
    p = _pool_with_base()
    snap = snapshot_pool(p, iteration=1)
    assert snap.iteration == 1
    assert "base" in snap.candidate_ids
    assert "base" in snap.pareto_frontier


def test_snapshot_pool_rejects_negative_iteration():
    p = _pool_with_base()
    with pytest.raises(ValueError):
        snapshot_pool(p, iteration=-1)


# ---------------------------------------------------------------------- #
# LeaseManager
# ---------------------------------------------------------------------- #
class _FakeAdapter:
    """Minimal adapter for lease manager tests."""
    adapter_name = "fake"

    def __init__(self) -> None:
        self.materialized: list[tuple[str, str]] = []

    def materialize_candidate(self, parent_version: str, attempt_id: str) -> CandidateWorkspace:
        self.materialized.append((parent_version, attempt_id))
        return _ws(attempt_id, parent_version)


def test_lease_acquire_and_release():
    m = SnapshotLeaseManager(
        snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()),
        adapter=_FakeAdapter(),
    )
    lease = m.acquire_lease("skills/a", "worker-1")
    assert lease.artifact_id == "skills/a"
    assert lease.holder == "worker-1"
    assert lease.released is False
    assert "skills/a" in m.active_leases

    m.release_lease("skills/a", "worker-1")
    assert "skills/a" not in m.active_leases


def test_lease_conflict_when_two_workers_ask_same_artifact():
    m = SnapshotLeaseManager(
        snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()),
        adapter=_FakeAdapter(),
    )
    m.acquire_lease("skills/a", "worker-1")
    with pytest.raises(LeaseConflict) as exc:
        m.acquire_lease("skills/a", "worker-2")
    assert exc.value.artifact_id == "skills/a"
    assert exc.value.current_holder == "worker-1"


def test_lease_reacquire_after_release():
    m = SnapshotLeaseManager(
        snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()),
        adapter=_FakeAdapter(),
    )
    m.acquire_lease("skills/a", "worker-1")
    m.release_lease("skills/a", "worker-1")
    # Now worker-2 can acquire.
    lease = m.acquire_lease("skills/a", "worker-2")
    assert lease.holder == "worker-2"


def test_lease_reacquire_same_holder_is_idempotent():
    m = SnapshotLeaseManager(
        snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()),
        adapter=_FakeAdapter(),
    )
    l1 = m.acquire_lease("skills/a", "worker-1")
    l2 = m.acquire_lease("skills/a", "worker-1")
    assert l1 is l2


def test_lease_release_rejects_wrong_holder():
    m = SnapshotLeaseManager(
        snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()),
        adapter=_FakeAdapter(),
    )
    m.acquire_lease("skills/a", "worker-1")
    with pytest.raises(ValueError):
        m.release_lease("skills/a", "worker-2")


def test_lease_release_unknown_raises():
    m = SnapshotLeaseManager(
        snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()),
        adapter=_FakeAdapter(),
    )
    with pytest.raises(KeyError):
        m.release_lease("skills/missing", "worker-1")


def test_lease_acquire_rejects_empty_args():
    m = SnapshotLeaseManager(
        snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()),
        adapter=_FakeAdapter(),
    )
    with pytest.raises(ValueError):
        m.acquire_lease("", "w1")
    with pytest.raises(ValueError):
        m.acquire_lease("a", "")


def test_materialize_workspace_uses_adapter():
    adapter = _FakeAdapter()
    m = SnapshotLeaseManager(
        snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()),
        adapter=adapter,
    )
    ws = m.materialize_workspace("base-v0", "att-1")
    assert ws.attempt_id == "att-1"
    assert adapter.materialized == [("base-v0", "att-1")]
    assert m.workspace_for("att-1") is ws


def test_materialize_workspace_rejects_duplicate_attempt():
    adapter = _FakeAdapter()
    m = SnapshotLeaseManager(
        snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()),
        adapter=adapter,
    )
    m.materialize_workspace("base-v0", "att-1")
    with pytest.raises(ValueError):
        m.materialize_workspace("base-v0", "att-1")


def test_concurrent_acquire_serializes():
    """Two threads acquiring different artifacts should both succeed."""
    m = SnapshotLeaseManager(
        snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()),
        adapter=_FakeAdapter(),
    )
    errors: list[Exception] = []

    def worker(name: str, artifact: str) -> None:
        try:
            m.acquire_lease(artifact, name)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=worker, args=("w1", "skills/a"))
    t2 = threading.Thread(target=worker, args=("w2", "skills/b"))
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert errors == []
    assert set(m.active_leases) == {"skills/a", "skills/b"}


# ---------------------------------------------------------------------- #
# BatchCoordinator
# ---------------------------------------------------------------------- #
def test_coordinator_submit_and_commit():
    bc = BatchCoordinator(snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()))
    r1 = _result("att-1")
    r2 = _result("att-2", artifacts=("skills/b",))
    bc.submit(r1)
    bc.submit(r2)
    assert bc.submitted_count == 2

    committed: list[WorkerResult] = []
    out = bc.commit_barrier(on_attempt_committed=committed.append)
    assert len(out) == 2
    # Sorted by attempt_id.
    assert out[0].attempt_id == "att-1"
    assert out[1].attempt_id == "att-2"
    assert bc.is_committed is True
    assert [r.attempt_id for r in committed] == ["att-1", "att-2"]


def test_coordinator_rejects_duplicate_attempt_id():
    bc = BatchCoordinator(snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()))
    bc.submit(_result("att-1"))
    with pytest.raises(ValueError):
        bc.submit(_result("att-1"))


def test_coordinator_rejects_submit_after_commit():
    bc = BatchCoordinator(snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()))
    bc.commit_barrier(on_attempt_committed=lambda r: None)
    with pytest.raises(RuntimeError):
        bc.submit(_result("att-1"))


def test_coordinator_rejects_double_commit():
    bc = BatchCoordinator(snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()))
    bc.commit_barrier(on_attempt_committed=lambda r: None)
    with pytest.raises(RuntimeError):
        bc.commit_barrier(on_attempt_committed=lambda r: None)


def test_coordinator_rejects_artifact_clash():
    bc = BatchCoordinator(snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()))
    bc.submit(_result("att-1", artifacts=("skills/a",)))
    bc.submit(_result("att-2", artifacts=("skills/a",)))  # same artifact!
    with pytest.raises(ValueError):
        bc.commit_barrier(on_attempt_committed=lambda r: None)


def test_coordinator_empty_commit_is_ok():
    bc = BatchCoordinator(snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()))
    out = bc.commit_barrier(on_attempt_committed=lambda r: None)
    assert out == ()
    assert bc.is_committed is True


def test_coordinator_commit_sorts_deterministically():
    bc = BatchCoordinator(snapshot=PoolSnapshot(iteration=0, candidate_ids=(), pareto_frontier=()))
    # Submit in reverse order; commit must still sort by attempt_id.
    bc.submit(_result("att-z", artifacts=("skills/z",)))
    bc.submit(_result("att-a", artifacts=("skills/a",)))
    bc.submit(_result("att-m", artifacts=("skills/m",)))
    out = bc.commit_barrier(on_attempt_committed=lambda r: None)
    ids = [r.attempt_id for r in out]
    assert ids == ["att-a", "att-m", "att-z"]
