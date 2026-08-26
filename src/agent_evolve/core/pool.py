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
    #: ``None`` means *no diagnosis exists* for this measurement -- same
    #: explicit-absence contract as ``blame_confidence`` below. Every passing
    #: rollout is legitimately undiagnosed; ``""`` would read as an unnamed
    #: analyzer on a diagnosed score.
    analyzer_model_id: str | None
    judge_model_id: str
    #: ``None`` means *no diagnosis exists* for this measurement -- e.g. the
    #: probe passed, so the diagnose gate legitimately produced nothing. This
    #: is explicit absence (the same distinction ``EntropyAvailabilityReport``
    #: draws): it must never be replaced by ``0.0``, which would read as a
    #: measured zero and make an undiagnosed score look like a confidently
    #: diagnosed one.
    blame_confidence: float | None
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
        if self.blame_confidence is not None and not (
            0.0 <= self.blame_confidence <= 1.0
        ):
            raise ValueError("blame_confidence must be in [0, 1] or None")
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

        .. warning::

           **Both weights are inert in every production path today, so this
           returns ``self.mean`` unchanged.** Verified, not assumed: no caller
           anywhere in ``src/`` passes ``severity=`` or ``confidence=`` to
           :class:`ScoreProvenance`. All four construction sites --
           ``core/orchestrator.py:342``, ``:1498``, ``:1872`` and
           ``pipeline.py:1469`` -- omit them, the class is frozen, and there is
           no ``dataclasses.replace``/``**kwargs`` path, so both hold their
           ``1.0`` defaults for the lifetime of every cell.

           There are **two unrelated fields named ``severity``** and conflating
           them is the trap here:

           * **(A)** ``CausalAnalysis.severity`` / ``CausalFinding.severity``
             -- the diagnoser's per-candidate LLM judgment. Genuinely written
             (``core/orchestrator.py:462``, ``:611``, ``:1405``) and consumed by
             issue synthesis, issue selection, and DPP targeting.
           * **(B)** ``ScoreProvenance.severity`` -> :attr:`ScoreCell.severity`
             -> this method -- the ``(task, mechanism)`` difficulty weight the
             spec asks for. **Never written.**

           (A) never flows into (B). A reading that assumes it does concludes
           this method creates a perverse selection gradient -- that a candidate
           the diagnoser is more alarmed about wins. It cannot: (B) is constant
           at ``1.0`` for every candidate, so it cancels in every comparison.

           A second trap sits alongside: :class:`ScoreProvenance` carries both
           ``blame_confidence`` (always passed) and ``confidence`` (never
           passed). Every production site sets the former, which reads as though
           the weight is wired. This method uses the latter.

           Consequences while this stands: Pareto dominance weights an easy task
           and a hard one identically, and :meth:`PersistentPool.parent_frequencies`
           degenerates to a count of cells won. Tracked as SV-1 (reclassified
           from "perverse gradient" to "inert multiplier", joining SV-5) in
           ``docs/SEVERE-OPEN-ISSUES.md``. Note that any test which passes
           ``severity=``/``confidence=`` by hand exercises a path production
           cannot take, so it demonstrates the arithmetic rather than the
           behaviour.
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
    #: Symmetric pairwise preference ``S_j`` against the incumbent, in [-1, 1].
    #:
    #: ``None`` means *no verdict was obtained*, which is deliberately distinct
    #: from ``0.0`` (a measured tie). Collapsing the two would make the ``S_j >
    #: 0`` gate reject an unjudged candidate for the wrong stated reason, and
    #: would make "the judge failed" indistinguishable from "the judge saw no
    #: difference" in an exported manifest.
    preference: float | None = None
    preference_available: int = 0
    preference_unavailable: int = 0
    #: SV-13 generational retirement. ``True`` means this entry has been
    #: superseded by one of its own offspring and is no longer eligible to
    #: *breed* -- excluded from parent sampling, the Pareto frontier and champion
    #: selection.
    #:
    #: **Its score cells are deliberately retained.** Retirement is a status, not
    #: a deletion: hard-removing the entry would destroy the comparable cells
    #: cross-candidate entropy requires (``core/entropy.py`` wants
    #: ``min_comparable_candidates`` per cell) and the negative evidence a later
    #: analysis needs, which is a strictly worse trade than the memory it saves.
    #: ``AGENTS.md``'s retention requirement is therefore preserved in full --
    #: only breeding eligibility changes.
    retired: bool = False
    #: Which candidate superseded this one. Provenance for a pool that shrank:
    #: an audit must be able to ask *why* an entry left the breeding population.
    superseded_by: str | None = None

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
    #: The winner's ``S_j``, or ``None`` when it had no verdict (the base
    #: normally has none: it is the comparison subject, not a candidate).
    preference: float | None = None
    #: Whether the RHO ``S_j > 0`` gate was enforced for this selection. An
    #: exported manifest must state this, or a paper run and an ablation run
    #: cannot be told apart after the fact.
    preference_gate_applied: bool = True
    #: SV-2: how many ``(task, mechanism)`` cells the winner shared with the
    #: entry it was most comparable against. ``0`` means the decision rested on no
    #: shared evidence at all -- the incumbent simply held -- and a reader of the
    #: manifest must be able to see that rather than infer a measured victory.
    comparable_cells: int = 0

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

    def record_preference(
        self,
        candidate_id: str,
        preference: float,
        *,
        available: int,
        unavailable: int = 0,
    ) -> None:
        """Attach the symmetric pairwise preference ``S_j`` to a candidate.

        ``available`` is the number of judge verdicts that actually returned a
        comparison. It is required rather than optional because ``preference``
        alone cannot express "unjudged": with ``available == 0`` the stored
        preference is forced to ``None`` no matter what value was passed, so an
        undecided candidate can never present itself as a measured tie.
        """
        if candidate_id not in self._entries:
            raise KeyError(f"unknown candidate: {candidate_id!r}")
        if not (-1.0 <= float(preference) <= 1.0):
            raise ValueError(
                f"preference must be in [-1, 1], got {preference}"
            )
        if available < 0 or unavailable < 0:
            raise ValueError("verdict counts must be >= 0")
        entry = self._entries[candidate_id]
        entry.preference = float(preference) if available > 0 else None
        entry.preference_available = int(available)
        entry.preference_unavailable = int(unavailable)

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
        """All non-dominated *live* candidate IDs in insertion order.

        SV-13: retired entries are excluded. A retired candidate occupying a
        frontier slot would defeat the cap the retirement mechanism exists to
        impose. Dominance itself is still evaluated against live entries only, so
        a retired ancestor cannot suppress a live descendant.
        """
        ids = self.live_candidate_ids()
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

        .. warning::

           ``severity`` and ``confidence`` are never written in production (see
           :meth:`ScoreCell.weighted_score`), so the increment below is
           effectively ``+= 1.0`` and this returns a **count of cells won**, not
           the importance-weighted strength the formula above describes.
           Tracked as SV-1.

        **SV-13:** only *live* candidates are considered, both as mass recipients
        and as cell competitors. Letting a retired entry still win a cell would
        deny that cell's mass to the live candidate that superseded it, which
        would defeat the point of retiring it.
        """
        # Group the weighted scores of every comparable cell across candidates.
        live = self.live_entries()
        winners: dict[tuple[str, str], list[str]] = {}
        all_cells = {
            k
            for e in live
            for k, v in e.score_tensor.items()
            if v.rollout_count >= self.min_comparable_rollouts
        }
        for k in sorted(all_cells):
            scored: dict[str, float] = {}
            for e in live:
                cell = e.score_tensor.get(k)
                if cell is not None and cell.rollout_count >= self.min_comparable_rollouts:
                    scored[e.candidate_id] = cell.weighted_score()
            if not scored:
                continue
            best = max(scored.values())
            winners[k] = [
                cid for cid, w in scored.items() if abs(w - best) <= self.epsilon
            ]

        freq = {cid: 0.0 for cid in self.live_candidate_ids()}
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
        """Outcome = mean of the candidate's per-task mean weighted scores.

        .. warning::

           **SV-2.** This averages over whatever tasks *this* entry happened to
           measure, so two entries measured on different task sets produce means
           that are not comparable. It is retained for the manifest -- the report
           must still publish an outcome figure -- but it no longer ranks anything.
           Ranking uses :meth:`_pairwise_outcome_preference`, which restricts to
           the cells both entries have evidence for.
        """
        per_task = entry.mean_weighted_score_per_task()
        if not per_task:
            return 0.0
        return sum(per_task.values()) / len(per_task)

    def _intersection_outcome(
        self, entry: PoolEntry, keys: Sequence[tuple[str, str]]
    ) -> float:
        """Mean weighted score of ``entry`` over exactly ``keys``.

        Grouped by complete ``task_id`` first, so a task carrying more mechanism
        clusters than another cannot dominate the mean by cell count alone --
        matching :meth:`PoolEntry.mean_weighted_score_per_task`.
        """
        by_task: dict[str, list[float]] = {}
        for task_id, cluster_id in keys:
            cell = entry.score_tensor.get((task_id, cluster_id))
            if cell is None or cell.rollout_count == 0:
                continue
            by_task.setdefault(task_id, []).append(cell.weighted_score())
        if not by_task:
            return 0.0
        per_task = [sum(v) / len(v) for v in by_task.values()]
        return sum(per_task) / len(per_task)

    def _pairwise_outcome_preference(self, a_id: str, b_id: str) -> int:
        """``1`` if ``a`` beats ``b`` on shared evidence, ``-1`` if worse, else ``0``.

        **SV-2.** The comparison is restricted to :meth:`comparable_cells` -- the
        cells both entries have evidence for, at or above
        ``min_comparable_rollouts``. Without that restriction a candidate raises
        its own mean by *not attempting* a hard task::

            base   easy(0.9) + hard(0.1)  -> 0.500
            candA  easy(0.9)              -> 0.900   <- wins by skipping

        On the shared cell those two are identical, so the honest verdict is a tie
        and the incumbent holds.

        ``0`` is returned when the overlap is empty: no shared evidence is not
        evidence of being better, and the alternative -- letting an unmeasured
        candidate displace a measured one -- is the defect this exists to remove.
        The tolerance is the pool's ``epsilon``, as in :meth:`dominates`.
        """
        keys = self.comparable_cells(a_id, b_id)
        if not keys:
            return 0
        a_score = self._intersection_outcome(self.get(a_id), keys)
        b_score = self._intersection_outcome(self.get(b_id), keys)
        if a_score > b_score + self.epsilon:
            return 1
        if a_score < b_score - self.epsilon:
            return -1
        return 0

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

        **Ranking is pairwise, not by the aggregate (SV-2).** Eligible entries are
        compared king-of-the-hill in insertion order, and an incumbent is displaced
        only by a challenger that scores better on the ``(task, mechanism)`` cells
        *both* measured (:meth:`comparable_cells`, so at or above
        ``min_comparable_rollouts``). A tie, a loss, or an empty overlap all leave
        the incumbent standing.

        The aggregate this method used to sort on is still computed and still
        reported::

            aggregate(c) = alpha*Outcome(c) + beta*ProcessCoverage(c)
                         + gamma*Stability(c) - delta*RegressionRisk(c)

        Defaults ``alpha=0.55``, ``beta=0.20``, ``gamma=0.15``, ``delta=0.10`` (or
        ``config.champion_*``). It is a **diagnostic only** -- no weight can change
        which candidate wins. That is deliberate, and it is what the four defects
        below required:

        * **SV-2.** ``Outcome`` averages over whatever cells each entry happened to
          measure, so a candidate raised its own mean by *not attempting* a hard
          task. Ranking pairwise on shared cells removes the exploit; the scalar
          ``outcome`` is retained for the manifest because an intersection-based
          figure is defined only relative to a second entry.
        * **SV-3.** ``ProcessCoverage`` measures *how much you measured*, not how
          well you did, and it carried 27% of the live weight -- so a strictly worse
          candidate could win on breadth. Coverage now acts only as the
          ``min_coverage_fraction`` *eligibility floor*, which is enforced.
        * **SV-5.** ``Stability`` is the constant ``1.0`` and ``RegressionRisk`` the
          constant ``0.0``. Neither was implemented, so both cancel in every
          comparison; ``gamma`` and ``delta`` weight nothing.

        Disqualification, in order, before any comparison: entries in
        ``protected_floor_violations``; retired entries (SV-13); candidates failing
        the pairwise gate below; and candidates whose coverage is under
        ``min_coverage_fraction`` (explicit argument, else
        ``config.champion_min_coverage_fraction``). Genuine ties between
        *comparable* entries break by ascending ``candidate_id``; the tie-breaker,
        the shared-cell count (``comparable_cells``) and the full disqualification
        list are all recorded on the report.

        **The RHO pairwise gate (SV-4).** Per RHO Algorithm 1 a candidate is
        accepted only when its symmetric pairwise preference ``S_j > 0``. That gate
        runs here, before anything is ranked, and is *eligibility* rather than
        score. Three consequences worth being explicit about, because each is a
        decision and not an accident:

        * **Strict** ``> 0``. A measured tie is not evidence of improvement.
        * **No verdict disqualifies.** ``preference is None`` means the judge
          never returned a comparison, so there is no evidence the candidate
          improved anything. The conservative reading is required: the permissive
          alternative would let a candidate whose judging silently failed inherit
          a promotion it never earned.
        * **The base is exempt.** The incumbent is the comparison subject, not a
          proposal competing for promotion. Gating it would empty the eligible
          set whenever nothing improved and turn that ordinary outcome into a
          ``ValueError``.

        The gate governs *promotion only*. Pool membership is untouched --
        AGENTS.md requires base plus every proposal to be retained, and the
        negative evidence of a rejected candidate is exactly what a later
        analysis needs.

        Set ``config.experimental_candidate_promotion=True`` to disable the gate
        for an ablation arm; the report records which mode was used.
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
        # Paper behaviour unless an ablation explicitly opts out.
        gate_applied = not (
            config.experimental_candidate_promotion if config is not None else False
        )
        scored: list[tuple[float, str, PoolEntry, float, float, float, float]] = []
        for entry in self.all_entries():
            # SV-13: a retired candidate has been superseded by its own
            # offspring; exporting it as champion would carry a harness the run
            # already replaced into the next run via --harness.
            if entry.retired:
                disqualified.add(entry.candidate_id)
                continue
            if entry.candidate_id in protected_floor_violations:
                continue
            # RHO Algorithm 1 acceptance gate. The base is exempt: it is the
            # incumbent being compared against, never a promotion candidate.
            if gate_applied and not entry.is_base:
                if entry.preference is None or entry.preference <= 0.0:
                    disqualified.add(entry.candidate_id)
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

        # SV-2: rank by *pairwise* comparison on shared evidence, not by the
        # per-candidate aggregate. The aggregate's `outcome` term averages over
        # whatever each entry happened to measure, so a candidate that skipped a
        # hard task outranked one that attempted it. A single sort key cannot
        # express this: intersection-restricted outcome is defined only relative
        # to a second entry, and two entries may share no cell at all.
        #
        # King-of-the-hill in insertion order. An incumbent is displaced only by a
        # challenger that is *better on the cells both measured*; a tie, a loss, or
        # absent overlap all leave it standing. Linear, deterministic, and it
        # reduces to the old behaviour when every entry measured the same cells.
        by_id = {item[1]: item for item in scored}
        order = [item[1] for item in scored]  # all_entries() is insertion-ordered
        champion_id = order[0]
        for challenger_id in order[1:]:
            verdict = self._pairwise_outcome_preference(challenger_id, champion_id)
            if verdict > 0:
                champion_id = challenger_id
            elif verdict == 0:
                # No shared evidence, or a genuine tie. The incumbent holds, but a
                # tie between *comparable* entries still needs a stable rule, so
                # fall back to the documented ascending-id tie-breaker rather than
                # to whichever happened to be inserted first.
                if self.comparable_cells(challenger_id, champion_id) and (
                    challenger_id < champion_id
                ):
                    champion_id = challenger_id

        _, _, entry, outcome, coverage, stability, regression_risk = by_id[champion_id]
        aggregate = by_id[champion_id][0]
        # How much shared evidence the decision actually rested on. A champion
        # chosen on one comparable cell and one chosen on forty are not equally
        # trustworthy, and the manifest must let a reader tell them apart.
        comparable = max(
            (
                len(self.comparable_cells(champion_id, other))
                for other in order
                if other != champion_id
            ),
            default=0,
        )
        return ChampionReport(
            entry=entry,
            outcome=outcome,
            coverage=coverage,
            stability=stability,
            regression_risk=regression_risk,
            aggregate=aggregate,
            tie_breaker="ascending_candidate_id",
            disqualifications=tuple(sorted(disqualified)),
            preference=entry.preference,
            preference_gate_applied=gate_applied,
            comparable_cells=comparable,
        )

    # ------------------------------------------------------------------ #
    # SV-13: generational retirement (soft; evidence is retained)
    # ------------------------------------------------------------------ #
    def live_candidate_ids(self) -> tuple[str, ...]:
        """Candidate ids still eligible to breed, in insertion order.

        "Live" is the breeding population. Retired entries remain in the pool and
        remain comparable for evidence; they are simply no longer sampled as
        parents, ranked on the frontier, or exported as champion.
        """
        return tuple(
            cid for cid in self._insertion_order if not self._entries[cid].retired
        )

    def live_entries(self) -> tuple[PoolEntry, ...]:
        return tuple(self._entries[cid] for cid in self.live_candidate_ids())

    def retire(self, candidate_id: str, *, superseded_by: str) -> PoolEntry:
        """Retire ``candidate_id`` from the breeding population.

        Called when an offspring has been shown to supersede its parent. The
        *decision* to retire belongs to the caller -- in production that is the
        RHO symmetric pairwise preference judge, the same instrument that gates
        promotion (SV-4) and resolves the final winner. This method enforces only
        the structural invariants:

        * **The live pool is never emptied.** ``select_parent`` and
          ``select_champion`` both need at least one eligible entry;
          emptying it would turn an ordinary "nothing improved" run into a
          ``ValueError``.
        * **Nothing supersedes itself.**
        * **Idempotent.** A replayed commit or a retry must not crash a run.

        Evidence is retained: see :attr:`PoolEntry.retired`.
        """
        if candidate_id not in self._entries:
            raise KeyError(candidate_id)
        if candidate_id == superseded_by:
            raise ValueError(
                f"{candidate_id!r} cannot be superseded by itself"
            )
        entry = self._entries[candidate_id]
        if entry.retired:
            return entry  # idempotent
        live = self.live_candidate_ids()
        if len(live) <= 1 and candidate_id in live:
            raise ValueError(
                f"refusing to retire {candidate_id!r}: it is the only live "
                "candidate, and an empty breeding population makes parent and "
                "champion selection unsatisfiable"
            )
        entry.retired = True
        entry.superseded_by = superseded_by
        return entry

    def has_sole_survivor(self) -> bool:
        """True when exactly one candidate is still live.

        This is the terminal condition: a breeding population of one needs no
        pairwise judging, because there is nothing left to compare it against.
        """
        return len(self.live_candidate_ids()) == 1

    def sole_survivor(self) -> PoolEntry:
        """The single live candidate. Raises unless exactly one remains."""
        live = self.live_candidate_ids()
        if len(live) != 1:
            raise ValueError(
                f"not a sole survivor: {len(live)} candidates are still live "
                f"({sorted(live)}); resolve them by pairwise preference instead"
            )
        return self._entries[live[0]]

    # ------------------------------------------------------------------ #
    # Prune (ablation only; never used for elite-only retention)
    # ------------------------------------------------------------------ #
    def prune(self, candidate_id: str) -> PoolEntry:
        """Remove a candidate from the pool **entirely**, destroying its evidence.

        This is intended ONLY for ablation studies (e.g., bounding pool size for
        memory-constrained runs). Pruning is never automatic.

        **Not the retirement path.** Generational retirement (SV-13) uses
        :meth:`retire`, which removes a superseded parent from the *breeding*
        population while keeping every score cell. Pruning is strictly stronger
        and strictly lossier: it deletes the comparable cells cross-candidate
        entropy depends on and the negative evidence that makes a rejected line
        analysable after the fact. If the goal is to stop breeding from a
        candidate, :meth:`retire` is the correct call.
        """
        if candidate_id == self._base_id:
            raise ValueError("cannot prune the base harness")
        if candidate_id not in self._entries:
            raise KeyError(candidate_id)
        entry = self._entries.pop(candidate_id)
        self._insertion_order.remove(candidate_id)
        return entry
