"""Entropy tracker with score floors and hierarchical DPP issue selection.

Per docs/architecture/target-rho-parallel-gepa.md:

    For comparable candidates:

        H(t,m) = Var({Q(h_i, t, m)})
                 * max(max_i Q(h_i, t, m), epsilon_floor)

    The score floor retains frontier-exploration signal where candidates differ
    but no strong solution exists yet. Entropy cannot drive selection until at
    least three comparable candidates and two rollouts per candidate support
    the cell.

    Use a max-heap for incremental entropy priority. Use hierarchical DPP:
    task selection first, then mechanism selection within tasks. Selection
    modes are ``dpp``, ``severity_rank``, and seeded ``random`` for ablations.

Notation
--------
* ``t``  = task ID
* ``m``  = mechanism cluster ID (per task)
* ``h_i`` = candidate ID
* ``Q(h_i, t, m)`` = measured outcome score for candidate h_i on task t,
                     attributed to mechanism cluster m. In [0, 1].

A "cell" is a (task, mechanism) pair. Each cell holds a dict of
candidate -> list of scores (one per rollout). Entropy is computed per cell;
the heap orders cells by entropy.
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------- #
# Cell-level entropy
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CellKey:
    task_id: str
    mechanism_cluster_id: str

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id is required")
        if not self.mechanism_cluster_id:
            raise ValueError("mechanism_cluster_id is required")


@dataclass(slots=True)
class _Cell:
    key: CellKey
    # candidate_id -> list of scores (one per rollout)
    scores: dict[str, list[float]] = field(default_factory=dict)
    # Candidate IDs that have been "promoted" into comparability for this cell.
    comparable: set[str] = field(default_factory=set)
    last_refreshed_iter: int = 0

    def add_score(self, candidate_id: str, score: float) -> None:
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {score}")
        self.scores.setdefault(candidate_id, []).append(score)

    def all_scores_flat(self) -> tuple[float, ...]:
        out: list[float] = []
        for v in self.scores.values():
            out.extend(v)
        return tuple(out)

    def candidate_count(self) -> int:
        return len(self.scores)

    def min_rollouts(self) -> int:
        if not self.scores:
            return 0
        return min(len(v) for v in self.scores.values())


def _variance(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / n


# ---------------------------------------------------------------------- #
# Entropy tracker
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class EntropyTracker:
    """Per-cell entropy with score floor and min-evidence gate.

    The architecture doc requires:
    * at least 3 comparable candidates per cell, AND
    * at least 2 rollouts per candidate,
    before entropy drives selection. Below that floor, entropy weight is 0.
    """

    epsilon_floor: float = 0.05
    min_comparable_candidates: int = 3
    min_rollouts_per_candidate: int = 2
    _cells: dict[CellKey, _Cell] = field(default_factory=dict)
    _heap: list[tuple[float, int, CellKey]] = field(default_factory=list)
    _heap_seq: int = 0  # tiebreaker for stable heap ordering
    _heap_dirty: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.epsilon_floor <= 1.0):
            raise ValueError("epsilon_floor must be in [0, 1]")
        if self.min_comparable_candidates < 1:
            raise ValueError("min_comparable_candidates must be >= 1")
        if self.min_rollouts_per_candidate < 1:
            raise ValueError("min_rollouts_per_candidate must be >= 1")

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def record_score(
        self,
        task_id: str,
        mechanism_cluster_id: str,
        candidate_id: str,
        score: float,
    ) -> None:
        key = CellKey(task_id=task_id, mechanism_cluster_id=mechanism_cluster_id)
        cell = self._cells.get(key)
        if cell is None:
            cell = _Cell(key=key)
            self._cells[key] = cell
        cell.add_score(candidate_id, score)
        self._heap_dirty = True

    def mark_comparable(self, task_id: str, mechanism_cluster_id: str, candidate_id: str) -> None:
        key = CellKey(task_id=task_id, mechanism_cluster_id=mechanism_cluster_id)
        cell = self._cells.get(key)
        if cell is None:
            cell = _Cell(key=key)
            self._cells[key] = cell
        cell.comparable.add(candidate_id)
        self._heap_dirty = True

    def refresh_at_barrier(self, iteration: int) -> None:
        """Mark a refresh barrier: all cells' last_refreshed_iter is updated."""
        if iteration < 0:
            raise ValueError("iteration must be >= 0")
        for cell in self._cells.values():
            cell.last_refreshed_iter = iteration
        self._heap_dirty = True

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def cell_entropy(self, task_id: str, mechanism_cluster_id: str) -> float:
        key = CellKey(task_id=task_id, mechanism_cluster_id=mechanism_cluster_id)
        cell = self._cells.get(key)
        if cell is None:
            return 0.0
        return self._entropy(cell)

    def _entropy(self, cell: _Cell) -> float:
        # Filter to comparable candidates only.
        comp = {c for c in cell.comparable if c in cell.scores}
        if len(comp) < self.min_comparable_candidates:
            return 0.0
        # And require min rollouts per candidate.
        if any(len(cell.scores[c]) < self.min_rollouts_per_candidate for c in comp):
            return 0.0
        # Flatten scores across comparable candidates.
        all_scores: list[float] = []
        for c in comp:
            all_scores.extend(cell.scores[c])
        if not all_scores:
            return 0.0
        var = _variance(all_scores)
        max_score = max(all_scores)
        return var * max(max_score, self.epsilon_floor)

    def entropy_weighted_with_freshness(
        self, task_id: str, mechanism_cluster_id: str, current_iter: int
    ) -> float:
        """Entropy reduced when evidence is stale.

        Staleness factor: 1 / (1 + iterations_since_refresh).
        """
        key = CellKey(task_id=task_id, mechanism_cluster_id=mechanism_cluster_id)
        cell = self._cells.get(key)
        if cell is None:
            return 0.0
        e = self._entropy(cell)
        if e <= 0.0:
            return 0.0
        age = max(0, current_iter - cell.last_refreshed_iter)
        return e / (1.0 + age)

    def all_cells(self) -> tuple[CellKey, ...]:
        return tuple(self._cells.keys())

    # ------------------------------------------------------------------ #
    # Heap
    # ------------------------------------------------------------------ #
    def _rebuild_heap(self) -> None:
        if not self._heap_dirty:
            return
        self._heap = []
        self._heap_seq = 0
        for key, cell in self._cells.items():
            e = self._entropy(cell)
            if e > 0.0:
                # Python's heapq is a min-heap; negate for max-heap.
                # seq as tiebreaker keeps insertion order deterministic.
                heapq.heappush(self._heap, (-e, self._heap_seq, key))
                self._heap_seq += 1
        self._heap_dirty = False

    def top_entropy_cells(self, k: int) -> tuple[tuple[CellKey, float], ...]:
        """Return up to k cells with the highest entropy, descending."""
        if k < 0:
            raise ValueError("k must be >= 0")
        self._rebuild_heap()
        out: list[tuple[CellKey, float]] = []
        tmp: list[tuple[float, int, CellKey]] = []
        while self._heap and len(out) < k:
            neg_e, seq, key = heapq.heappop(self._heap)
            out.append((key, -neg_e))
            tmp.append((neg_e, seq, key))
        # Restore.
        for item in tmp:
            heapq.heappush(self._heap, item)
        return tuple(out)


# ---------------------------------------------------------------------- #
# Hierarchical DPP
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Issue:
    """A selectable issue: (task, mechanism) pair with metadata."""

    task_id: str
    mechanism_cluster_id: str
    severity: float
    entropy: float
    freshness_weight: float


def greedy_map_dpp(
    kernel: np.ndarray,
    k: int,
    min_gain: float = 1e-12,
) -> tuple[int, ...]:
    """Select a DPP MAP approximation via incremental Schur complements.

    Selecting an item subtracts its projection from each remaining item's
    marginal gain. Therefore highly similar candidates lose gain after one of
    their near-duplicates is selected. This is the required bounded O(N^2)
    alternative to exact eigendecomposition.
    """
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be a square matrix")
    if k <= 0 or kernel.shape[0] == 0:
        return ()
    n = kernel.shape[0]
    if k >= n:
        return tuple(range(n))

    gains = np.diag(kernel).astype(float, copy=True)
    coefficients: list[list[float]] = [[] for _ in range(n)]
    remaining = set(range(n))
    selected: list[int] = []

    for _ in range(k):
        # Index is the stable-ID order once callers have sorted their items.
        chosen = min(remaining, key=lambda index: (-gains[index], index))
        if gains[chosen] <= min_gain:
            break
        selected.append(chosen)
        remaining.remove(chosen)
        diagonal = math.sqrt(gains[chosen])
        for index in remaining:
            projection = sum(
                left * right
                for left, right in zip(coefficients[index], coefficients[chosen])
            )
            coefficient = (kernel[index, chosen] - projection) / diagonal
            coefficients[index].append(coefficient)
            # Clamp only floating-point drift; never reward similarity.
            gains[index] = max(0.0, gains[index] - coefficient * coefficient)

    return tuple(selected)


def _dpp_select(
    items: Sequence[tuple[str, float, float]],
    k: int,
    similarity: "callable[[str, str], float]",
    rng: random.Random,
) -> tuple[str, ...]:
    """Build a quality-weighted kernel and run deterministic greedy-MAP DPP."""
    del rng  # DPP is deterministic; random is an explicitly separate ablation.
    if k <= 0 or not items:
        return ()

    # Bound the dense matrix and make its positional tie-break reflect stable IDs.
    capped = sorted(items, key=lambda item: (-item[1], item[0]))[:100]
    ids = tuple(item[0] for item in capped)
    raw_qualities = [max(item[1], 0.1) for item in capped]
    maximum = max(raw_qualities)
    qualities = [quality / maximum for quality in raw_qualities]
    kernel = np.empty((len(capped), len(capped)), dtype=float)
    for row, item_id in enumerate(ids):
        for column, other_id in enumerate(ids):
            if row == column:
                kernel[row, column] = qualities[row] ** 2 + 1e-6
            else:
                sim = min(1.0, max(0.0, similarity(item_id, other_id)))
                kernel[row, column] = qualities[row] * sim * qualities[column]
    return tuple(ids[index] for index in greedy_map_dpp(kernel, min(k, len(ids))))


@dataclass(slots=True)
class HierarchicalDPPSelector:
    """Two-level DPP: tasks first, then mechanisms within tasks.

    Modes
    -----
    * ``dpp``: greedy k-DPP at both levels (default).
    * ``severity_rank``: pick by severity descending, no diversity bonus.
    * ``random``: seeded random shuffle, no diversity bonus.
    """

    mode: str = "dpp"
    seed: int = 0
    lambda_div: float = 0.5
    # Optional similarity functions. Default = no diversity bonus.
    task_similarity: "callable[[str, str], float] | None" = None
    mechanism_similarity: "callable[[str, str], float] | None" = None
    _rng: random.Random = field(init=False, default_factory=lambda: random.Random(0))

    def __post_init__(self) -> None:
        if self.mode not in ("dpp", "severity_rank", "random"):
            raise ValueError(f"unknown mode: {self.mode!r}")
        if self.lambda_div < 0.0:
            raise ValueError("lambda_div must be >= 0")
        object.__setattr__(self, "_rng", random.Random(self.seed))

    def select(
        self, issues: Sequence[Issue], k_tasks: int, k_mechanisms_per_task: int
    ) -> tuple[Issue, ...]:
        if k_tasks < 0 or k_mechanisms_per_task < 0:
            raise ValueError("k_tasks and k_mechanisms_per_task must be >= 0")
        if not issues:
            return ()

        # Group issues by task.
        by_task: dict[str, list[Issue]] = {}
        for iss in issues:
            by_task.setdefault(iss.task_id, []).append(iss)

        # --- Task selection ---
        task_quality: list[tuple[str, float, float]] = []
        for task_id, task_issues in by_task.items():
            avg_entropy = (
                sum(i.entropy * i.freshness_weight for i in task_issues) / len(task_issues)
            )
            max_severity = max(i.severity for i in task_issues)
            # Quality = blend of severity and entropy.
            quality = 0.5 * max_severity + 0.5 * avg_entropy
            task_quality.append((task_id, quality, 1.0))

        if self.mode == "random":
            self._rng.shuffle(task_quality)
            chosen_tasks = tuple(t[0] for t in task_quality[:k_tasks])
        elif self.mode == "severity_rank":
            task_quality.sort(key=lambda t: (-t[1], t[0]))
            chosen_tasks = tuple(t[0] for t in task_quality[:k_tasks])
        else:  # dpp
            sim = self.task_similarity or (lambda a, b: 0.0)
            chosen_tasks = _dpp_select(task_quality, k_tasks, sim, self._rng)

        # --- Mechanism selection within chosen tasks ---
        out: list[Issue] = []
        for task_id in chosen_tasks:
            task_issues = by_task[task_id]
            mech_items: list[tuple[str, float, float]] = []
            for iss in task_issues:
                quality = 0.5 * iss.severity + 0.5 * (iss.entropy * iss.freshness_weight)
                mech_items.append((iss.mechanism_cluster_id, quality, 1.0))

            if self.mode == "random":
                self._rng.shuffle(mech_items)
                chosen = tuple(m[0] for m in mech_items[:k_mechanisms_per_task])
            elif self.mode == "severity_rank":
                mech_items.sort(key=lambda m: (-m[1], m[0]))
                chosen = tuple(m[0] for m in mech_items[:k_mechanisms_per_task])
            else:  # dpp
                sim = self.mechanism_similarity or (lambda a, b: 0.0)
                chosen = _dpp_select(mech_items, k_mechanisms_per_task, sim, self._rng)

            chosen_set = set(chosen)
            for iss in task_issues:
                if iss.mechanism_cluster_id in chosen_set:
                    out.append(iss)
        return tuple(out)
