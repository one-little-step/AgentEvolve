"""Trace-backed issues and hierarchical DPP selection.

This module is the target-correct implementation of the selection-algorithms
contract (``docs/architecture/selection-algorithms.md:67-280``). It supersedes
the legacy ``Issue`` / ``greedy_map_dpp`` / ``HierarchicalDPPSelector`` in
:mod:`agent_evolve.core.entropy`, which remain only until the orchestrator is
migrated.

Design notes
------------
* :class:`Issue` carries a *trace-backed writable-artifact attribution*: a
  finding with no inventory-declared writable artifact referenced by trace
  evidence is rejected in :func:`build_issue` before it can ever rank.
* Raw issue quality is a weighted sum of severity, confidence,
  min-max-normalized entropy, coverage need, and pareto relevance. Entropy is
  normalized *within the candidate set* (``selection-algorithms.md``), so a
  single issue's ``raw_quality`` uses ``normalized_entropy=0.0`` and the
  selector recomputes the final raw quality over the full set at selection time.
* DPP builds ``L = Q S Q + jitter*I`` and runs greedy MAP inference with
  incremental Schur-complement updates. Degenerate kernels fall back to a
  deterministic quality-ordered selection and *record the reason*; silent
  fallback is forbidden.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Callable, Sequence, TypeVar

_T = TypeVar("_T")

import numpy as np

# ---------------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------------- #
DEFAULT_QUALITY_WEIGHTS: tuple[float, ...] = (0.3, 0.2, 0.2, 0.2, 0.1)
DEFAULT_THETA = 0.7
DEFAULT_SCORE_FLOOR = 0.1
DEFAULT_MAX_ITEMS = 100
DEFAULT_MIN_GAIN = 1e-12
DEFAULT_JITTER = 1e-6

_KERNEL_CONDITION_LIMIT = 1e12


# ---------------------------------------------------------------------- #
# Issue model
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Issue:
    """A selectable, trace-backed issue.

    ``writable_artifact_ids`` is the write set: inventory-declared writable
    artifacts attributed to this issue by trace evidence. Issues with an empty
    write set are rejected before ranking.
    """

    issue_id: str
    task_id: str
    mechanism_cluster_id: str
    severity: float
    confidence: float
    entropy: float
    coverage_need: float
    pareto_relevance: float
    raw_quality: float
    embedding: tuple[float, ...]
    writable_artifact_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    lineage: str = ""
    entropy_tier: str = "recombination_target"

    def __post_init__(self) -> None:
        if not self.issue_id:
            raise ValueError("issue_id is required")
        if not self.task_id:
            raise ValueError("task_id is required")
        object.__setattr__(self, "embedding", tuple(self.embedding))
        object.__setattr__(self, "writable_artifact_ids", tuple(self.writable_artifact_ids))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        for name in ("severity", "confidence", "coverage_need", "pareto_relevance"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class IssueSelectionReport:
    """The outcome of one selection call, with full configuration recorded."""

    selected: tuple[Issue, ...]
    mode: str
    theta: float
    alpha: float
    score_floor: float
    prefilter_total: int
    prefilter_retained: int
    fallback_reason: str | None
    weights: tuple[float, ...]

    @property
    def items(self) -> tuple[Issue, ...]:
        """Alias for ``selected`` (the brief's test surface)."""
        return self.selected


# ---------------------------------------------------------------------- #
# Building issues
# ---------------------------------------------------------------------- #
def raw_issue_quality(
    issue: Issue,
    *,
    normalized_entropy: float = 0.0,
    weights: tuple[float, ...] = DEFAULT_QUALITY_WEIGHTS,
    frontier_weight: float = 0.30,
    entropy_tier: str | None = None,
) -> float:
    """Weighted sum of the five evidence signals.

    ``normalized_entropy`` is supplied by the caller because entropy is
    min-max normalized within the candidate set, not per issue.

    ``entropy_tier`` (from the issue or an explicit override) controls how the
    entropy component is treated per ``selection-algorithms.md:87-95``:

    * ``"frontier_exploration"`` scales the entropy component by
      ``frontier_weight`` (default 0.30), retaining but dampening frontier
      signal where no strong solution exists yet.
    * ``"skip"`` contributes zero entropy (evidence floor unmet).
    * anything else contributes the entropy component at full weight.
    """
    if len(weights) != 5:
        raise ValueError("weights must have exactly 5 components")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError("weights must sum to 1.0")
    w_severity, w_confidence, w_entropy, w_coverage, w_pareto = weights
    tier = entropy_tier if entropy_tier is not None else issue.entropy_tier
    if tier == "skip":
        entropy_term = 0.0
    else:
        entropy_term = w_entropy * normalized_entropy
        if tier == "frontier_exploration":
            entropy_term *= frontier_weight
    return (
        w_severity * issue.severity
        + w_confidence * issue.confidence
        + entropy_term
        + w_coverage * issue.coverage_need
        + w_pareto * issue.pareto_relevance
    )


def build_issue(
    finding,
    inventory: Sequence,
    *,
    entropy: float = 0.0,
    coverage_need: float = 0.0,
    pareto_relevance: float = 0.0,
    embedding: tuple[float, ...] = (),
    lineage: str = "",
    entropy_tier: str = "recombination_target",
) -> Issue | None:
    """Build a trace-backed issue from a finding, or reject it with ``None``.

    ``inventory`` is a sequence of ``ArtifactDescriptor`` (each exposing
    ``.artifact_id`` and ``.writable``). An issue is only produced when at least
    one attributed artifact is inventory-declared writable; otherwise ``None``
    is returned (``selection-algorithms.md:135``).
    """
    attributed: set[str] = set()
    blame_graph = getattr(finding, "blame_graph", None)
    if blame_graph is not None:
        for node in getattr(blame_graph, "nodes", ()):
            attributed.update(getattr(node, "artifacts", ()))
    for ref in getattr(finding, "evidence_refs", ()) or ():
        attributed.add(ref)

    writable_ids = {
        desc.artifact_id
        for desc in inventory
        if getattr(desc, "writable", False)
    }
    write_set = tuple(sorted(aid for aid in attributed if aid in writable_ids))
    if not write_set:
        return None

    severity = getattr(finding, "severity", None)
    confidence = getattr(finding, "confidence", None)
    mechanism_cluster_id = getattr(finding, "mechanism_cluster_id", None) or ""

    issue = Issue(
        issue_id=getattr(finding, "verdict_id", ""),
        task_id=getattr(finding, "task_id", ""),
        mechanism_cluster_id=mechanism_cluster_id,
        severity=float(severity) if severity is not None else 0.0,
        confidence=float(confidence) if confidence is not None else 0.0,
        entropy=float(entropy),
        coverage_need=float(coverage_need),
        pareto_relevance=float(pareto_relevance),
        raw_quality=0.0,
        embedding=tuple(embedding),
        writable_artifact_ids=write_set,
        evidence_refs=tuple(getattr(finding, "evidence_refs", ()) or ()),
        lineage=lineage,
        entropy_tier=entropy_tier,
    )
    raw = raw_issue_quality(issue, normalized_entropy=0.0)
    return replace(issue, raw_quality=raw)


# ---------------------------------------------------------------------- #
# Quality
# ---------------------------------------------------------------------- #
def quality(
    issue: Issue,
    maximum_raw_quality: float,
    theta: float,
    score_floor: float,
) -> float:
    """Floor, normalize, and exponentiate a raw issue quality."""
    floored = max(issue.raw_quality, score_floor)
    normalized = floored / maximum_raw_quality
    alpha = theta / (2 * max(1 - theta, 1e-6)) if theta < 1.0 else 1.0
    return normalized ** alpha


# ---------------------------------------------------------------------- #
# Kernel and greedy MAP
# ---------------------------------------------------------------------- #
def build_kernel(
    qualities: Sequence[float],
    similarity: Callable[[int, int], float],
    jitter: float = DEFAULT_JITTER,
) -> np.ndarray:
    """Build the DPP kernel ``L = Q S Q + jitter*I``."""
    n = len(qualities)
    kernel = np.empty((n, n), dtype=float)
    for i in range(n):
        qi = float(qualities[i])
        kernel[i, i] = qi * qi + jitter
        for j in range(i):
            sim = similarity(i, j)
            kernel[i, j] = qi * sim * float(qualities[j])
            kernel[j, i] = kernel[i, j]
    return kernel


def greedy_map(
    kernel: np.ndarray,
    ids: tuple[str, ...],
    k: int,
    min_gain: float,
) -> tuple[int, ...]:
    """Greedy MAP inference via incremental Schur-complement updates.

    Tie-breaks on the full ascending string ID (``ids[i]``), clamps gains at
    0.0 only to absorb floating-point drift.
    """
    k = min(k, len(ids))
    gains = np.diag(kernel).copy()
    factors: list[list[float]] = [[] for _ in ids]
    selected: list[int] = []
    while len(selected) < k:
        remaining = sorted(
            (i for i in range(len(ids)) if i not in selected),
            key=lambda i: (-gains[i], ids[i]),
        )
        j = remaining[0]
        if gains[j] <= min_gain:
            break
        selected.append(j)
        d_j = math.sqrt(gains[j])
        for i in remaining[1:]:
            projection = sum(left * right for left, right in zip(factors[i], factors[j]))
            e = (kernel[i, j] - projection) / d_j
            factors[i].append(e)
            gains[i] = max(0.0, gains[i] - e * e)
    return tuple(selected)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _similarity_matrix(
    embeddings: Sequence[tuple[float, ...]],
) -> tuple[np.ndarray | None, str | None]:
    """Cosine similarity over L2-normalized embeddings, clamped to [0, 1].

    Returns ``(None, reason)`` when embeddings are missing, empty, or
    dimension-incompatible.
    """
    dims = {len(e) for e in embeddings}
    if any(len(e) == 0 for e in embeddings):
        return None, "incompatible_embeddings"
    if len(dims) != 1:
        return None, "incompatible_embeddings"
    n = len(embeddings)
    normalized: list[list[float]] = []
    for e in embeddings:
        mag = math.sqrt(sum(x * x for x in e))
        if mag == 0.0:
            normalized.append([0.0] * len(e))
        else:
            normalized.append([x / mag for x in e])
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = min(1.0, max(0.0, _cosine(normalized[i], normalized[j])))
            sim[i, j] = c
            sim[j, i] = c
    return sim, None


def _mean_embedding(embeddings: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
    if not embeddings:
        return ()
    dim = len(embeddings[0])
    mean = [0.0] * dim
    for e in embeddings:
        for i, v in enumerate(e):
            mean[i] += v
    return tuple(m / len(embeddings) for m in mean)


# ---------------------------------------------------------------------- #
# Selector
# ---------------------------------------------------------------------- #
class HierarchicalDPPSelector:
    """Issue selection with DPP, severity-rank, coverage, and seeded-random modes.

    ``select(issues, k_tasks, k_mechanisms_per_task)`` performs two-stage
    hierarchical selection (tasks, then mechanisms within tasks). The flat
    convenience form ``select(issues, k=k)`` selects ``k`` issues across the
    whole set and is what the mandated DPP behavior tests exercise.
    """

    def __init__(
        self,
        mode: str = "dpp",
        theta: float = DEFAULT_THETA,
        score_floor: float = DEFAULT_SCORE_FLOOR,
        max_items: int = DEFAULT_MAX_ITEMS,
        min_gain: float = DEFAULT_MIN_GAIN,
        seed: int = 0,
        task_similarity: Callable[[str, str], float] | None = None,
        mechanism_similarity: Callable[[str, str], float] | None = None,
        *,
        weights: tuple[float, ...] = DEFAULT_QUALITY_WEIGHTS,
        jitter: float = DEFAULT_JITTER,
        max_per_mechanism: int | None = None,
        frontier_weight: float = 0.30,
    ) -> None:
        if mode not in ("dpp", "severity_rank", "coverage", "random"):
            raise ValueError(f"unknown mode: {mode!r}")
        if not (0.0 <= theta <= 1.0):
            raise ValueError("theta must be in [0, 1]")
        if not (0.0 < score_floor <= 1.0):
            raise ValueError("score_floor must be in (0, 1]")
        if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1:
            raise ValueError("max_items must be a positive integer")
        if min_gain < 0.0:
            raise ValueError("min_gain must be >= 0")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if len(weights) != 5 or not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("weights must have 5 components summing to 1.0")
        if jitter < 0.0:
            raise ValueError("jitter must be >= 0")
        if max_per_mechanism is not None and max_per_mechanism < 1:
            raise ValueError("max_per_mechanism must be a positive integer or None")
        if not (0.0 <= frontier_weight <= 1.0):
            raise ValueError("frontier_weight must be in [0, 1]")

        self.mode = mode
        self.theta = float(theta)
        self.score_floor = float(score_floor)
        self.max_items = max_items
        self.min_gain = float(min_gain)
        self.seed = seed
        self.task_similarity = task_similarity
        self.mechanism_similarity = mechanism_similarity
        self.weights = tuple(weights)
        self.jitter = float(jitter)
        self.max_per_mechanism = max_per_mechanism
        self.frontier_weight = float(frontier_weight)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def select(
        self,
        issues: Sequence[Issue],
        k_tasks: int | None = None,
        k_mechanisms_per_task: int | None = None,
        *,
        k: int | None = None,
    ) -> IssueSelectionReport:
        if k is not None:
            return self._select_flat(tuple(issues), k)
        if k_tasks is None or k_mechanisms_per_task is None:
            raise ValueError(
                "select() requires either k (flat) or both k_tasks and "
                "k_mechanisms_per_task (hierarchical)"
            )
        return self._select_hierarchical(tuple(issues), k_tasks, k_mechanisms_per_task)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _alpha(self) -> float:
        if self.theta < 1.0:
            return self.theta / (2 * max(1 - self.theta, 1e-6))
        return 1.0

    def _report(
        self,
        selected: tuple[Issue, ...],
        prefilter_total: int,
        prefilter_retained: int,
        fallback_reason: str | None,
    ) -> IssueSelectionReport:
        return IssueSelectionReport(
            selected=selected,
            mode=self.mode,
            theta=self.theta,
            alpha=self._alpha(),
            score_floor=self.score_floor,
            prefilter_total=prefilter_total,
            prefilter_retained=prefilter_retained,
            fallback_reason=fallback_reason,
            weights=self.weights,
        )

    def _final_raw_quality(self, issues: Sequence[Issue]) -> list[float]:
        if not issues:
            return []
        entropies = [i.entropy for i in issues]
        lo, hi = min(entropies), max(entropies)
        span = hi - lo
        normalized = [(e - lo) / span if span > 0.0 else 0.0 for e in entropies]
        return [
            raw_issue_quality(
                i,
                normalized_entropy=n,
                weights=self.weights,
                frontier_weight=self.frontier_weight,
                entropy_tier=i.entropy_tier,
            )
            for i, n in zip(issues, normalized)
        ]

    def _apply_hard_constraints(self, issues: Sequence[Issue]) -> list[Issue]:
        attributed = [i for i in issues if i.writable_artifact_ids]
        raw = self._final_raw_quality(attributed)
        order = sorted(
            range(len(attributed)),
            key=lambda idx: (-raw[idx], attributed[idx].issue_id),
        )
        claimed: set[str] = set()
        kept: list[int] = []
        for idx in order:
            ws = set(attributed[idx].writable_artifact_ids)
            if ws & claimed:
                continue
            claimed |= ws
            kept.append(idx)

        if self.max_per_mechanism is not None:
            per_mech: dict[str, int] = {}
            capped: list[int] = []
            for idx in sorted(
                kept, key=lambda j: (-raw[j], attributed[j].issue_id)
            ):
                mech = attributed[idx].mechanism_cluster_id
                if per_mech.get(mech, 0) >= self.max_per_mechanism:
                    continue
                per_mech[mech] = per_mech.get(mech, 0) + 1
                capped.append(idx)
            kept = capped

        return [attributed[idx] for idx in sorted(kept, key=lambda j: attributed[j].issue_id)]

    def _quality_ordered(self, issues: Sequence[Issue], k: int) -> tuple[Issue, ...]:
        raw = self._final_raw_quality(issues)
        order = sorted(
            range(len(issues)),
            key=lambda idx: (-raw[idx], issues[idx].issue_id),
        )
        return tuple(issues[idx] for idx in order[: max(0, k)])

    def _dpp_over(
        self,
        items: Sequence[tuple[_T, float, tuple[float, ...]]],
        k: int,
        *,
        key: Callable[[_T], str] | None = None,
        sim: np.ndarray | None = None,
    ) -> tuple[list[_T], str | None]:
        """Run greedy-MAP DPP over generic (item, quality, embedding) triples.

        ``sim`` is an optional caller-supplied similarity matrix over ``items``
        order; when absent, cosine similarity is derived from embeddings.
        """
        if len(items) < 2:
            return [], "fewer_than_two_candidates"
        if sim is None:
            sim, reason = _similarity_matrix([e for _, _, e in items])
            if sim is None:
                return [], reason
        try:
            qvals = [q for _, q, _ in items]
            kernel = build_kernel(qvals, lambda i, j: sim[i, j], self.jitter)
            eig = np.linalg.eigvalsh(kernel)
            min_eig = float(eig[0])
            if min_eig <= 0.0:
                return [], "degenerate_kernel"
            if float(eig[-1]) / min_eig > _KERNEL_CONDITION_LIMIT:
                return [], "degenerate_kernel"
            ids = tuple(
                key(item_id) if key is not None else str(item_id)
                for item_id, _, _ in items
            )
            chosen = greedy_map(kernel, ids, min(k, len(items)), self.min_gain)
        except Exception:
            return [], "exception"
        return [items[i][0] for i in chosen], None

    def _theta_qualities(self, raw: Sequence[float]) -> list[float]:
        floored = [max(r, self.score_floor) for r in raw]
        maximum = max(floored) if floored else 1.0
        alpha = self._alpha()
        return [(f / maximum) ** alpha for f in floored]

    # ------------------------------------------------------------------ #
    # Flat selection
    # ------------------------------------------------------------------ #
    def _select_flat(self, issues: tuple[Issue, ...], k: int) -> IssueSelectionReport:
        filtered = self._apply_hard_constraints(issues)
        prefilter_total = len(filtered)
        if not filtered or k <= 0:
            return self._report((), prefilter_total, 0, None)

        raw = self._final_raw_quality(filtered)

        if self.mode == "dpp":
            order = sorted(
                range(len(filtered)),
                key=lambda idx: (-(raw[idx] + filtered[idx].entropy), filtered[idx].issue_id),
            )
            capped_idx = order[: self.max_items]
            capped = [filtered[idx] for idx in capped_idx]
            capped_raw = [raw[idx] for idx in capped_idx]
            prefilter_retained = len(capped)

            qvals = self._theta_qualities(capped_raw)
            items = [(iss, q, iss.embedding) for iss, q in zip(capped, qvals)]
            chosen, fallback = self._dpp_over(items, k, key=lambda iss: iss.issue_id)
            if fallback is not None:
                selected = self._quality_ordered(capped, k)
                return self._report(selected, prefilter_total, prefilter_retained, fallback)
            return self._report(tuple(chosen), prefilter_total, prefilter_retained, None)

        if self.mode == "severity_rank":
            order = sorted(
                range(len(filtered)),
                key=lambda idx: (
                    -filtered[idx].severity,
                    -filtered[idx].confidence,
                    -filtered[idx].entropy,
                    filtered[idx].issue_id,
                ),
            )
            selected = tuple(filtered[idx] for idx in order[:k])
            return self._report(selected, prefilter_total, prefilter_total, None)

        if self.mode == "coverage":
            selected = self._coverage_select(filtered, k)
            return self._report(selected, prefilter_total, prefilter_total, None)

        # random
        rng = random.Random(self.seed)
        indices = list(range(len(filtered)))
        rng.shuffle(indices)
        selected = tuple(filtered[idx] for idx in indices[:k])
        return self._report(selected, prefilter_total, prefilter_total, None)

    def _coverage_select(
        self, issues: Sequence[Issue], k: int
    ) -> tuple[Issue, ...]:
        n = len(issues)
        if n == 0 or k <= 0:
            return ()
        if k >= n:
            return tuple(issues)
        sim, _ = _similarity_matrix([i.embedding for i in issues])
        if sim is None:
            return tuple(sorted(issues, key=lambda i: i.issue_id)[:k])
        raw = self._final_raw_quality(issues)
        seed = max(range(n), key=lambda idx: (raw[idx], _neg_id(issues[idx].issue_id)))
        selected = [seed]
        remaining = set(range(n)) - {seed}
        while len(selected) < k and remaining:
            best_idx = -1
            best_dist = -1.0
            for idx in sorted(remaining, key=lambda j: issues[j].issue_id):
                dist = min(1.0 - sim[idx, j] for j in selected)
                if dist > best_dist + 1e-15:
                    best_dist = dist
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        return tuple(issues[idx] for idx in selected)

    # ------------------------------------------------------------------ #
    # Hierarchical selection
    # ------------------------------------------------------------------ #
    def _select_hierarchical(
        self, issues: tuple[Issue, ...], k_tasks: int, k_mechanisms_per_task: int
    ) -> IssueSelectionReport:
        filtered = self._apply_hard_constraints(issues)
        prefilter_total = len(filtered)
        if not filtered or k_tasks <= 0 or k_mechanisms_per_task <= 0:
            return self._report((), prefilter_total, 0, None)

        raw = self._final_raw_quality(filtered)
        by_task: dict[str, list[int]] = {}
        for idx, issue in enumerate(filtered):
            by_task.setdefault(issue.task_id, []).append(idx)

        task_ids = sorted(by_task.keys())
        task_raw = {
            t: sum(raw[idx] for idx in by_task[t]) / len(by_task[t]) for t in task_ids
        }
        task_emb = {
            t: _mean_embedding([filtered[idx].embedding for idx in by_task[t]])
            for t in task_ids
        }
        chosen_tasks, fallback_reason = self._select_level(
            task_ids, task_raw, task_emb, k_tasks, self.task_similarity
        )

        out: list[Issue] = []
        for t in chosen_tasks:
            mech_groups: dict[str, list[int]] = {}
            for idx in by_task[t]:
                mech_groups.setdefault(filtered[idx].mechanism_cluster_id, []).append(idx)
            mech_ids = sorted(mech_groups.keys())
            mech_raw = {
                m: sum(raw[idx] for idx in mech_groups[m]) / len(mech_groups[m])
                for m in mech_ids
            }
            mech_emb = {
                m: _mean_embedding([filtered[idx].embedding for idx in mech_groups[m]])
                for m in mech_ids
            }
            chosen_mechs, mech_fallback = self._select_level(
                mech_ids, mech_raw, mech_emb, k_mechanisms_per_task, self.mechanism_similarity
            )
            if fallback_reason is None and mech_fallback is not None:
                fallback_reason = mech_fallback
            for m in chosen_mechs:
                for idx in sorted(mech_groups[m], key=lambda j: filtered[j].issue_id):
                    out.append(filtered[idx])

        return self._report(tuple(out), prefilter_total, len(filtered), fallback_reason)

    def _select_level(
        self,
        ids: list[str],
        raw: dict[str, float],
        embeddings: dict[str, tuple[float, ...]],
        k: int,
        similarity_override: Callable[[str, str], float] | None,
    ) -> tuple[tuple[str, ...], str | None]:
        if k >= len(ids):
            return tuple(ids), None
        if self.mode == "random":
            rng = random.Random(self.seed)
            shuffled = list(ids)
            rng.shuffle(shuffled)
            return tuple(shuffled[:k]), None
        if self.mode == "severity_rank":
            return tuple(sorted(ids, key=lambda i: (-raw[i], i))[:k]), None
        if self.mode == "coverage":
            return self._coverage_level(ids, embeddings, k), None

        # dpp
        reason: str | None = None
        if similarity_override is not None:
            sim = np.eye(len(ids))
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    s = min(1.0, max(0.0, similarity_override(ids[a], ids[b])))
                    sim[a, b] = sim[b, a] = s
        else:
            sim, reason = _similarity_matrix([embeddings[i] for i in ids])
        if sim is None:
            return tuple(sorted(ids, key=lambda i: (-raw[i], i))[:k]), reason

        qvals = self._theta_qualities([raw[i] for i in ids])
        items = [(i, q, embeddings[i]) for i, q in zip(ids, qvals)]
        chosen, fallback = self._dpp_over(items, k, sim=sim)
        if fallback is not None:
            return tuple(sorted(ids, key=lambda i: (-raw[i], i))[:k]), fallback
        return tuple(str(c) for c in chosen), None

    def _coverage_level(
        self, ids: list[str], embeddings: dict[str, tuple[float, ...]], k: int
    ) -> tuple[str, ...]:
        sim, _ = _similarity_matrix([embeddings[i] for i in ids])
        if sim is None:
            return tuple(ids[:k])
        selected: list[int] = [0]
        remaining = set(range(1, len(ids)))
        while len(selected) < k and remaining:
            best_idx = -1
            best_dist = -1.0
            for idx in sorted(remaining):
                dist = min(1.0 - sim[idx, j] for j in selected)
                if dist > best_dist + 1e-15:
                    best_dist = dist
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        return tuple(ids[idx] for idx in selected)


def _neg_id(issue_id: str) -> float:
    """A numeric key that orders larger strings first (for a max() tie-break)."""
    if not issue_id:
        return 0.0
    return -1.0 / (len(issue_id) + 1)
