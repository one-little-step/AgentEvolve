"""Snapshot/lease manager and parallel batch coordinator.

Per docs/architecture/target-rho-parallel-gepa.md:

    Parallel mode creates an immutable pool/history snapshot, selects
    compatible issues, grants exclusive artifact write leases, and gives each
    worker an isolated candidate workspace. Workers do not write shared
    pool/history state. A coordinator commits sorted attempt results at a
    barrier.

Design
------
* :class:`SnapshotLeaseManager` takes an immutable snapshot of the pool and
  edit memory, hands out isolated workspaces via the adapter, and grants
  exclusive write leases per artifact_id. Two workers asking for the same
  artifact get a :class:`LeaseConflict`.
* :class:`BatchCoordinator` accepts worker results, sorts them by
  deterministic key, and commits them all at a barrier via
  :meth:`commit_barrier`. Before the barrier, no shared state is mutated.
* The coordinator refuses to commit attempts whose artifacts would clash
  with another committed attempt in the same batch (defensive double-check;
  the lease manager should already prevent this).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from agent_evolve.core.contracts import (
    ArtifactEdit,
    CandidateWorkspace,
    EvolutionAdapter,
    EvolutionTask,
    ExecutionTrace,
)
from agent_evolve.core.memory import EditAttempt
from agent_evolve.core.pool import PersistentPool


# ---------------------------------------------------------------------- #
# Snapshot
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PoolSnapshot:
    """Immutable view of the pool at barrier time.

    The snapshot only carries identity + score tensor references; it is NOT
    a deep copy of the pool. Workers must NOT mutate the pool through the
    snapshot; they read it for selection decisions only.
    """

    iteration: int
    candidate_ids: tuple[str, ...]
    pareto_frontier: tuple[str, ...]


def snapshot_pool(pool: PersistentPool, iteration: int) -> PoolSnapshot:
    if iteration < 0:
        raise ValueError("iteration must be >= 0")
    return PoolSnapshot(
        iteration=iteration,
        candidate_ids=pool.candidate_ids(),
        pareto_frontier=pool.pareto_frontier(),
    )


# ---------------------------------------------------------------------- #
# Lease manager
# ---------------------------------------------------------------------- #
class LeaseConflict(Exception):
    """Raised when two workers request write leases on the same artifact."""

    def __init__(self, artifact_id: str, current_holder: str) -> None:
        super().__init__(
            f"artifact {artifact_id!r} already leased to {current_holder!r}"
        )
        self.artifact_id = artifact_id
        self.current_holder = current_holder


@dataclass(slots=True)
class Lease:
    artifact_id: str
    holder: str
    acquired_at: float
    released: bool = False


@dataclass(slots=True)
class SnapshotLeaseManager:
    """Grants exclusive per-artifact write leases for one parallel batch.

    The manager is single-threaded by construction (callers must serialize
    acquire/release), but it uses an internal lock so concurrent worker
    threads can call acquire/release safely. The underlying assumption is
    that the orchestrator selects compatible issues upfront so lease
    conflicts indicate a real bug, not a normal race.
    """

    snapshot: PoolSnapshot
    adapter: EvolutionAdapter
    _leases: dict[str, Lease] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # attempt_id -> workspace, so workers can retrieve their workspace.
    _workspaces: dict[str, CandidateWorkspace] = field(default_factory=dict)

    def acquire_lease(self, artifact_id: str, holder: str) -> Lease:
        if not artifact_id:
            raise ValueError("artifact_id is required")
        if not holder:
            raise ValueError("holder is required")
        with self._lock:
            existing = self._leases.get(artifact_id)
            if existing is not None and not existing.released:
                if existing.holder == holder:
                    return existing
                raise LeaseConflict(artifact_id, existing.holder)
            lease = Lease(
                artifact_id=artifact_id,
                holder=holder,
                acquired_at=time.time(),
            )
            self._leases[artifact_id] = lease
            return lease

    def release_lease(self, artifact_id: str, holder: str) -> None:
        with self._lock:
            existing = self._leases.get(artifact_id)
            if existing is None:
                raise KeyError(f"no lease for {artifact_id!r}")
            if existing.holder != holder:
                raise ValueError(
                    f"holder {holder!r} cannot release lease held by {existing.holder!r}"
                )
            existing.released = True

    def materialize_workspace(
        self,
        parent_version: str,
        attempt_id: str,
    ) -> CandidateWorkspace:
        if not parent_version:
            raise ValueError("parent_version is required")
        if not attempt_id:
            raise ValueError("attempt_id is required")
        with self._lock:
            if attempt_id in self._workspaces:
                raise ValueError(f"attempt_id already in use: {attempt_id!r}")
            ws = self.adapter.materialize_candidate(parent_version, attempt_id)
            self._workspaces[attempt_id] = ws
            return ws

    def workspace_for(self, attempt_id: str) -> CandidateWorkspace:
        with self._lock:
            if attempt_id not in self._workspaces:
                raise KeyError(attempt_id)
            return self._workspaces[attempt_id]

    @property
    def active_leases(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(
                aid for aid, l in self._leases.items() if not l.released
            ))


# ---------------------------------------------------------------------- #
# Batch coordinator
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class WorkerResult:
    """One worker's completed attempt, ready for commit."""

    attempt_id: str
    workspace: CandidateWorkspace
    edits: tuple[ArtifactEdit, ...]
    trace: ExecutionTrace
    attempt: EditAttempt  # the structured attempt record (post-validation)


@dataclass(slots=True)
class BatchCoordinator:
    """Collects worker results and commits them at a barrier.

    The coordinator does NOT execute workers; the orchestrator runs them
    (sequentially or in parallel) and submits results here. The coordinator
    only ensures:
    * No two committed results touch the same artifact (defensive check).
    * Results are sorted by attempt_id before commit (deterministic order).
    * Commit is atomic: either all results are applied to the pool/edit
      memory, or none are.
    """

    snapshot: PoolSnapshot
    _results: dict[str, WorkerResult] = field(default_factory=dict)
    _committed: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def submit(self, result: WorkerResult) -> None:
        with self._lock:
            if self._committed:
                raise RuntimeError("batch already committed; cannot submit")
            if result.attempt_id in self._results:
                raise ValueError(f"duplicate attempt_id: {result.attempt_id!r}")
            self._results[result.attempt_id] = result

    @property
    def submitted_count(self) -> int:
        return len(self._results)

    @property
    def is_committed(self) -> bool:
        return self._committed

    def commit_barrier(
        self,
        on_attempt_committed: "Callable[[WorkerResult], None]",
        on_attempt_rolled_back: "Callable[[WorkerResult], None] | None" = None,
    ) -> tuple[WorkerResult, ...]:
        """Atomically commit all submitted results.

        The coordinator does NOT mutate the pool or edit memory directly,
        because constructing an :class:`EvolutionCandidate` requires
        adapter-specific knowledge (artifact_hashes, parent_ids, etc.) that
        the orchestrator owns. Instead, the orchestrator passes a callback
        that receives each result in deterministic order; the callback
        performs the actual pool/edit-memory mutation.

        The coordinator still enforces:
        * No two submitted results touch the same artifact (clash check).
        * Results are passed to the callback in deterministic (attempt_id)
          order.
        * Commit happens at most once.
        * A callback failure invokes compensation callbacks in reverse order
          before the error is re-raised. The real production barrier will
          move this responsibility into ``core.storage``'s ACID transaction;
          this compensation hook keeps the prototype path fail-closed.
        """
        with self._lock:
            if self._committed:
                raise RuntimeError("batch already committed")
            if not self._results:
                self._committed = True
                return ()

            # Defensive artifact-clash check.
            seen_artifacts: dict[str, str] = {}
            for rid, r in self._results.items():
                for e in r.edits:
                    if e.artifact_id in seen_artifacts:
                        raise ValueError(
                            f"artifact {e.artifact_id!r} touched by both "
                            f"{seen_artifacts[e.artifact_id]!r} and {rid!r}"
                        )
                    seen_artifacts[e.artifact_id] = rid

            # Sort by attempt_id for deterministic commit order.
            sorted_results = tuple(
                self._results[aid] for aid in sorted(self._results.keys())
            )

            committed: list[WorkerResult] = []
            try:
                for r in sorted_results:
                    on_attempt_committed(r)
                    committed.append(r)
            except Exception:
                if on_attempt_rolled_back is not None:
                    for r in reversed(committed):
                        on_attempt_rolled_back(r)
                raise

            self._committed = True
            return sorted_results
