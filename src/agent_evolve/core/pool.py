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
from typing import TYPE_CHECKING, AbstractSet, Iterable, Mapping, Sequence

from agent_evolve.core.contracts import EvolutionCandidate

if TYPE_CHECKING:
    from agent_evolve.core.config import ResolvedConfig


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
    severity: float = 1.0
    confidence: float = 1.0
    artifact_versions: Mapping[str, str] = field(default_factory=dict)

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
        if not (0.0 <= self.severity <= 1.0):
            raise ValueError("severity must be in [0, 1]")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")
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

    @property
    def severity(self) -> float:
        """Cell-level severity: mean severity across rollouts.

        ``severity`` is a property of the (task, mechanism) pair, so it is
        expected to be constant within a cell; the mean is a defensive summary.
        """
        if not self.provenance:
            return 0.0
        return sum(p.severity for p in self.provenance) / len(self.provenance)

    @property
    def confidence(self) -> float:
        """Cell-level confidence: mean confidence across rollouts."""
        if not self.provenance:
            return 0.0
        return sum(p.confidence for p in self.provenance) / len(self.provenance)

    def weighted_score(self) -> float:
        """Weighted cell value = mean score * severity * confidence.

        Per docs/architecture/selection-algorithms.md, the Pareto and parent
        objectives use ``score * severity * confidence`` rather than the raw
        mean. A cell with no rollouts yields 0.0 and is never treated as
        evidence.
        """
        return self.mean * self.severity * self.confidence


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

    def mean_weighted_score_per_task(self) -> Mapping[str, float]:
        """Mean weighted score across mechanisms, per task.

        Mirrors :meth:`mean_score_per_task` but uses the weighted cell value
        (``score * severity * confidence``) as the per-cell objective. The
        grouping key is the complete ``task_id``.
        """
        by_task: dict[str, list[float]] = {}
        for (task_id, _cluster_id), cell in self.score_tensor.items():
            if cell.rollout_count == 0:
                continue
            by_task.setdefault(task_id, []).append(cell.weighted_score())
        return {t: sum(v) / len(v) for t, v in by_task.items() if v}


@dataclass(frozen=True, slots=True)
class ChampionReport:
    """Structured champion-selection manifest.

    Per ``docs/architecture/selection-algorithms.md:324-338``, the manifest must
    expose every aggregate component, the coverage figure, the tie-breaker, and
    the disqualification list. ``entry`` is the winning :class:`PoolEntry`;
    ``candidate_id`` is exposed as a convenience alias.
    """

    entry: PoolEntry
    outcome: float
    coverage: float
    stability: float
    regression_risk: float
    aggregate: float
    tie_breaker: str = "ascending_candidate_id"
    disqualifications: tuple[str, ...] = ()

    @property
    def candidate_id(self) -> str:
        return self.entry.candidate_id


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
    epsilon: float = 1e-9
    _entries: dict[str, PoolEntry] = field(default_factory=dict)
    _insertion_order: list[str] = field(default_factory=list)
    _base_id: str = ""

    def __post_init__(self) -> None:
        if self.min_comparable_rollouts < 1:
            raise ValueError("min_comparable_rollouts must be >= 1")
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be >= 0")

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
    def comparable_cells(self, a_id: str, b_id: str) -> tuple[tuple[str, str], ...]:
        """Return the (task, mechanism) keys both entries share with enough rollouts.

        Comparability requires the same complete ``task_id`` and
        ``mechanism_cluster_id`` (exact full string), and at least
        ``min_comparable_rollouts`` rollouts on both sides. Cells present for
        only one candidate, or below the rollout floor, are excluded — never
        zero-filled.
        """
        a = self.get(a_id)
        b = self.get(b_id)
        keys_a = {k for k, v in a.score_tensor.items() if v.rollout_count >= self.min_comparable_rollouts}
        keys_b = {k for k, v in b.score_tensor.items() if v.rollout_count >= self.min_comparable_rollouts}
        # Sort for determinism.
        return tuple(sorted(keys_a & keys_b))

    def is_comparable(self, a_id: str, b_id: str) -> bool:
        """True iff the two candidates share at least one comparable cell."""
        return bool(self.comparable_cells(a_id, b_id))

    def comparison_exclusions(self, a_id: str, b_id: str) -> Mapping[tuple[str, str], str]:
        """Map each non-comparable (task, mechanism) cell to its exclusion reason.

        Reasons cover cells missing for one candidate and cells that fall below
        ``min_comparable_rollouts`` on either side.
        """
        a = self.get(a_id)
        b = self.get(b_id)
        reasons: dict[tuple[str, str], str] = {}
        all_keys = set(a.score_tensor) | set(b.score_tensor)
        for k in sorted(all_keys):
            ca = a.score_tensor.get(k)
            cb = b.score_tensor.get(k)
            if ca is None or cb is None:
                missing = a_id if ca is None else b_id
                reasons[k] = f"missing for {missing}"
            elif (
                ca.rollout_count < self.min_comparable_rollouts
                or cb.rollout_count < self.min_comparable_rollouts
            ):
                reasons[k] = (
                    f"insufficient rollouts (min {self.min_comparable_rollouts})"
                )
        return reasons

    def dominates(self, a_id: str, b_id: str) -> bool:
        """True iff a Pareto-dominates b on their comparable key overlap.

        Per docs/architecture/selection-algorithms.md, dominance is evaluated on
        the weighted cell value ``score * severity * confidence`` using
        comparable cells only, with an ``epsilon`` tolerance. If the overlap is
        empty, neither dominates.
        """
        a = self.get(a_id)
        b = self.get(b_id)
        keys = self.comparable_cells(a_id, b_id)
        if not keys:
            return False
        a_strictly_better = False
        for k in keys:
            wa = a.score_tensor[k].weighted_score()
            wb = b.score_tensor[k].weighted_score()
            if wa < wb - self.epsilon:
                return False
            if wa > wb + self.epsilon:
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
    # Parent sampling
    # ------------------------------------------------------------------ #
    def parent_frequencies(self) -> Mapping[str, float]:
        """Per-candidate parent frequency for seeded parent sampling.

        Per docs/architecture/selection-algorithms.md::

            frequency(c) = sum over winning (t, m) of severity * confidence

        A candidate wins ``(t, m)`` when it holds the strict maximum comparable
        weighted score for that cell; ties award all tied winners. Returns a
        mapping over every candidate in insertion order (zero for non-winners).
        """
        # Group the weighted scores of every comparable cell across candidates.
        winners: dict[tuple[str, str], list[str]] = {}
        all_cells = {
            k
            for e in self.all_entries()
            for k, v in e.score_tensor.items()
            if v.rollout_count >= self.min_comparable_rollouts
        }
        for k in sorted(all_cells):
            scored: dict[str, float] = {}
            for e in self.all_entries():
                cell = e.score_tensor.get(k)
                if cell is not None and cell.rollout_count >= self.min_comparable_rollouts:
                    scored[e.candidate_id] = cell.weighted_score()
            if not scored:
                continue
            best = max(scored.values())
            winners[k] = [
                cid for cid, w in scored.items() if abs(w - best) <= self.epsilon
            ]

        freq = {cid: 0.0 for cid in self.candidate_ids()}
        for k, winning_ids in winners.items():
            for cid in winning_ids:
                cell = self.get(cid).score_tensor[k]
                freq[cid] += cell.severity * cell.confidence
        return freq

    # ------------------------------------------------------------------ #
    # Champion selection
    # ------------------------------------------------------------------ #
    def _observed_cells(self) -> set[tuple[str, str]]:
        """Union of every evaluated (task, mechanism) cell across the pool."""
        return {
            k
            for e in self.all_entries()
            for k, v in e.score_tensor.items()
            if v.rollout_count >= 1
        }

    def _champion_outcome(self, entry: PoolEntry) -> float:
        """Outcome = mean of the candidate's per-task mean weighted scores."""
        per_task = entry.mean_weighted_score_per_task()
        if not per_task:
            return 0.0
        return sum(per_task.values()) / len(per_task)

    def _champion_coverage(self, entry: PoolEntry, total_cells: set[tuple[str, str]]) -> float:
        """ProcessCoverage = fraction of evaluated cells vs total observed."""
        if not total_cells:
            return 0.0
        evaluated = {k for k, v in entry.score_tensor.items() if v.rollout_count >= 1}
        return len(evaluated & total_cells) / len(total_cells)

    def select_champion(
        self,
        protected_floor_violations: AbstractSet[str] = frozenset(),
        config: ResolvedConfig | None = None,
        *,
        min_coverage_fraction: float | None = None,
    ) -> ChampionReport:
        """Select the champion and return a structured :class:`ChampionReport`.

        Per docs/architecture/selection-algorithms.md::

            aggregate(c) = alpha*Outcome(c) + beta*ProcessCoverage(c)
                         + gamma*Stability(c) - delta*RegressionRisk(c)

        Defaults: ``alpha=0.55``, ``beta=0.20``, ``gamma=0.15``, ``delta=0.10``
        (or ``config.champion_*`` when supplied). Stability defaults to 1.0
        (single-source) and RegressionRisk to 0.0. Candidates in
        ``protected_floor_violations`` are disqualified before ranking.
        Candidates whose ProcessCoverage is below ``min_coverage_fraction``
        (from the explicit argument or ``config.champion_min_coverage_fraction``)
        are likewise disqualified before ranking. Ties break deterministically
        by ascending ``candidate_id``; the tie-breaker and the full
        disqualification list are recorded on the report.
        """
        alpha = config.champion_alpha if config is not None else 0.55
        beta = config.champion_beta if config is not None else 0.20
        gamma = config.champion_gamma if config is not None else 0.15
        delta = config.champion_delta if config is not None else 0.10
        if min_coverage_fraction is None:
            min_coverage_fraction = (
                config.champion_min_coverage_fraction if config is not None else 0.0
            )

        total_cells = self._observed_cells()
        disqualified = set(protected_floor_violations)
        scored: list[tuple[float, str, PoolEntry, float, float, float, float]] = []
        for entry in self.all_entries():
            if entry.candidate_id in protected_floor_violations:
                continue
            outcome = self._champion_outcome(entry)
            coverage = self._champion_coverage(entry, total_cells)
            if coverage < min_coverage_fraction:
                disqualified.add(entry.candidate_id)
                continue
            stability = 1.0
            regression_risk = 0.0
            aggregate = (
                alpha * outcome
                + beta * coverage
                + gamma * stability
                - delta * regression_risk
            )
            scored.append(
                (aggregate, entry.candidate_id, entry, outcome, coverage, stability, regression_risk)
            )

        if not scored:
            raise ValueError("no eligible candidates for champion selection")

        # Highest aggregate, then ascending candidate_id for determinism.
        scored.sort(key=lambda item: (-item[0], item[1]))
        _, _, entry, outcome, coverage, stability, regression_risk = scored[0]
        return ChampionReport(
            entry=entry,
            outcome=outcome,
            coverage=coverage,
            stability=stability,
            regression_risk=regression_risk,
            aggregate=scored[0][0],
            tie_breaker="ascending_candidate_id",
            disqualifications=tuple(sorted(disqualified)),
        )

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
