"""Persistent candidate pool with common score tensor and Pareto selection.

Per docs/architecture/target-rho-parallel-gepa.md:

    The initial pool contains the base harness and every RHO proposal. The
    base receives repeated ``G`` rollout evidence. Each post-RHO candidate
    starts with one rollout per selected task to maintain RHO-scale cost.
    Candidates receive adaptive repeat rollouts when they become Pareto
    relevant, have uncertain attribution, need merge evaluation, or require
    worked-set validation.

    Candidate evidence must retain provenance:

        task ID
        trace IDs
        rollout count
        analyzer/judge model ID
        mechanism cluster ID
        score coverage
        blame confidence and stability
        artifact versions

    No candidate may be compared as Pareto-equivalent merely because it has a
    number; score provenance and coverage must be compatible.

Design
------
* The pool stores :class:`PoolEntry` records, one per candidate version.
* Each entry carries a *score tensor* — a sparse map keyed by (task_id,
  mechanism_cluster_id) -> :class:`ScoreCell` with provenance.
* Pareto dominance is evaluated only between entries whose score coverage is
  *compatible*: same set of (task, mechanism) keys, OR overlapping keys where
  both entries have >= ``min_comparable_rollouts`` rollouts.
* The pool is append-only by default. Removing an entry requires an explicit
  ``prune`` call (used for size-bounded ablations, NOT for elite-only
  retention — the architecture explicitly forbids elite-only retention).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from agent_evolve.core.contracts import EvolutionCandidate


# ---------------------------------------------------------------------- #
# Score tensor
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ScoreProvenance:
    """Provenance for one measured score value."""

    task_id: str
    mechanism_cluster_id: str
    trace_id: str
    rollout_seq: int  # which rollout this was (0-indexed within the cell)
    analyzer_model_id: str
    judge_model_id: str
    blame_confidence: float
    blame_stability: float
    artifact_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id is required")
        if not self.mechanism_cluster_id:
            raise ValueError("mechanism_cluster_id is required")
        if not self.trace_id:
            raise ValueError("trace_id is required")
        if self.rollout_seq < 0:
            raise ValueError("rollout_seq must be >= 0")
        if not (0.0 <= self.blame_confidence <= 1.0):
            raise ValueError("blame_confidence must be in [0, 1]")
        if not (0.0 <= self.blame_stability <= 1.0):
            raise ValueError("blame_stability must be in [0, 1]")
        # Freeze artifact_versions.
        object.__setattr__(self, "artifact_versions", dict(self.artifact_versions))


@dataclass(slots=True)
class ScoreCell:
    """One cell of the score tensor: all rollouts for (cand, task, mech)."""

    scores: list[float] = field(default_factory=list)
    provenance: list[ScoreProvenance] = field(default_factory=list)

    def add(self, score: float, prov: ScoreProvenance) -> None:
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {score}")
        if prov.rollout_seq != len(self.scores):
            raise ValueError(
                f"rollout_seq must be {len(self.scores)} (next slot), got {prov.rollout_seq}"
            )
        self.scores.append(score)
        self.provenance.append(prov)

    @property
    def rollout_count(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    @property
    def max(self) -> float:
        if not self.scores:
            return 0.0
        return max(self.scores)


# ---------------------------------------------------------------------- #
# Pool entry
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class PoolEntry:
    """One candidate version tracked by the persistent pool."""

    candidate: EvolutionCandidate
    is_base: bool
    # (task_id, mechanism_cluster_id) -> ScoreCell
    score_tensor: dict[tuple[str, str], ScoreCell] = field(default_factory=dict)
    # Origin attempt IDs (RHO proposals or edits).
    origin_attempt_ids: tuple[str, ...] = ()

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def version(self) -> str:
        return self.candidate.version

    def cell(self, task_id: str, mechanism_cluster_id: str) -> ScoreCell:
        key = (task_id, mechanism_cluster_id)
        if key not in self.score_tensor:
            self.score_tensor[key] = ScoreCell()
        return self.score_tensor[key]

    def cell_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(self.score_tensor.keys())

    def mean_score_per_task(self) -> Mapping[str, float]:
        """Mean across mechanisms, per task. Used for quick Pareto checks.

        The grouping key is the complete ``task_id``. Prefixes, slices, and
        truncations are forbidden as aggregation keys per
        docs/architecture/data-contracts.md, because they silently merge
        distinct tasks into a single Pareto objective.
        """
        by_task: dict[str, list[float]] = {}
        for (task_id, _cluster_id), cell in self.score_tensor.items():
            # A cell with no rollouts carries no evidence; it is not a zero.
            if cell.rollout_count == 0:
                continue
            by_task.setdefault(task_id, []).append(cell.mean)
        return {t: sum(v) / len(v) for t, v in by_task.items() if v}


# ---------------------------------------------------------------------- #
# Pool
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class PersistentPool:
    """Append-only persistent candidate pool.

    Initial pool contains base + every RHO proposal. Entries are never
    silently dropped; ``prune`` is the only removal path and is intended for
    ablation studies, not for elite-only retention.
    """

    min_comparable_rollouts: int = 2
    _entries: dict[str, PoolEntry] = field(default_factory=dict)
    _insertion_order: list[str] = field(default_factory=list)
    _base_id: str = ""

    def __post_init__(self) -> None:
        if self.min_comparable_rollouts < 1:
            raise ValueError("min_comparable_rollouts must be >= 1")

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def add_base(self, candidate: EvolutionCandidate) -> PoolEntry:
        if self._base_id:
            raise ValueError("base already exists in pool")
        entry = PoolEntry(
            candidate=candidate,
            is_base=True,
            origin_attempt_ids=(),
        )
        self._entries[candidate.candidate_id] = entry
        self._insertion_order.append(candidate.candidate_id)
        self._base_id = candidate.candidate_id
        return entry

    def add_candidate(self, candidate: EvolutionCandidate, origin_attempt_ids: Iterable[str] = ()) -> PoolEntry:
        if candidate.candidate_id in self._entries:
            raise ValueError(f"duplicate candidate_id: {candidate.candidate_id!r}")
        if candidate.candidate_id == self._base_id:
            raise ValueError("cannot add base twice via add_candidate")
        entry = PoolEntry(
            candidate=candidate,
            is_base=False,
            origin_attempt_ids=tuple(origin_attempt_ids),
        )
        self._entries[candidate.candidate_id] = entry
        self._insertion_order.append(candidate.candidate_id)
        return entry

    def record_score(self, candidate_id: str, score: float, prov: ScoreProvenance) -> None:
        if candidate_id not in self._entries:
            raise KeyError(f"unknown candidate: {candidate_id!r}")
        cell = self._entries[candidate_id].cell(prov.task_id, prov.mechanism_cluster_id)
        cell.add(score, prov)

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    @property
    def base_id(self) -> str:
        if not self._base_id:
            raise ValueError("pool has no base yet")
        return self._base_id

    @property
    def base(self) -> PoolEntry:
        return self._entries[self.base_id]

    def get(self, candidate_id: str) -> PoolEntry:
        if candidate_id not in self._entries:
            raise KeyError(candidate_id)
        return self._entries[candidate_id]

    def __contains__(self, candidate_id: object) -> bool:
        return candidate_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def all_entries(self) -> tuple[PoolEntry, ...]:
        return tuple(self._entries[eid] for eid in self._insertion_order)

    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(self._insertion_order)

    # ------------------------------------------------------------------ #
    # Pareto
    # ------------------------------------------------------------------ #
    def _compatible_keys(self, a: PoolEntry, b: PoolEntry) -> tuple[tuple[str, str], ...]:
        """Return the (task, mechanism) keys both entries share with enough rollouts."""
        out: list[tuple[str, str]] = []
        keys_a = {k: v for k, v in a.score_tensor.items() if v.rollout_count >= self.min_comparable_rollouts}
        keys_b = {k: v for k, v in b.score_tensor.items() if v.rollout_count >= self.min_comparable_rollouts}
        for k in keys_a.keys() & keys_b.keys():
            out.append(k)
        # Sort for determinism.
        return tuple(sorted(out))

    def dominates(self, a_id: str, b_id: str) -> bool:
        """True iff a Pareto-dominates b on their compatible key overlap.

        Per the architecture: candidates are only comparable on cells where
        both have enough rollout evidence (``min_comparable_rollouts``). If
        the overlap is empty, neither dominates.
        """
        a = self.get(a_id)
        b = self.get(b_id)
        keys = self._compatible_keys(a, b)
        if not keys:
            return False
        a_strictly_better = False
        for k in keys:
            ca = a.score_tensor[k]
            cb = b.score_tensor[k]
            if ca.mean < cb.mean:
                return False
            if ca.mean > cb.mean:
                a_strictly_better = True
        return a_strictly_better

    def pareto_frontier(self) -> tuple[str, ...]:
        """All non-dominated candidate IDs in insertion order."""
        ids = self.candidate_ids()
        out: list[str] = []
        for cand in ids:
            dominated = False
            for other in ids:
                if other == cand:
                    continue
                if self.dominates(other, cand):
                    dominated = True
                    break
            if not dominated:
                out.append(cand)
        return tuple(out)

    # ------------------------------------------------------------------ #
    # Prune (ablation only; never used for elite-only retention)
    # ------------------------------------------------------------------ #
    def prune(self, candidate_id: str) -> PoolEntry:
        """Remove a candidate from the pool.

        This is intended ONLY for ablation studies (e.g., bounding pool size
        for memory-constrained runs). The default behavior is to retain all
        candidates; pruning is never automatic.
        """
        if candidate_id == self._base_id:
            raise ValueError("cannot prune the base harness")
        if candidate_id not in self._entries:
            raise KeyError(candidate_id)
        entry = self._entries.pop(candidate_id)
        self._insertion_order.remove(candidate_id)
        return entry
