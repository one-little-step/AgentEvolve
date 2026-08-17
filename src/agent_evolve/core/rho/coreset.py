"""RHO coreset selection: difficulty-weighted diversity over historical records.

RHO does not diagnose every historical trajectory. It selects a small coreset
that is simultaneously:

1. difficult, because difficult trajectories expose harness weakness, and
2. diverse, so the optimizer does not learn from one narrow failure mode.

The DPP primitives are shared with genetic issue selection
(``build_kernel``/``greedy_map`` in :mod:`agent_evolve.core.issues`); only the
*quality vector* differs. Here quality is judge-assigned difficulty. In genetic
issue selection quality is cross-candidate score variance. Those two must not be
unified: they answer different questions (spec §6, "two quality functions, one
DPP").

``embedding_text`` comes from the comprehended trajectory summary plus the
abstract fingerprint, never the raw trace. A raw trace is ~60% identifiers and
schema keys, which saturates cosine similarity and neutralizes the diversity
term (spec §4.2).

Boundary note: ``core/`` never imports the CUGA judge adapter. Difficulty and
fingerprints arrive as plain values in :class:`CoresetCandidate`;
:func:`candidates_from_verdicts` duck-types the adapter's ``DifficultyVerdict``
shape without importing it.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

from agent_evolve.core.clustering import (
    EmbeddingProviderUnavailable,
    MechanismEmbedder,
)
from agent_evolve.core.issues import build_kernel, greedy_map

DEFAULT_THETA = 0.7
DEFAULT_SCORE_FLOOR = 0.1
DEFAULT_MIN_GAIN = 1e-12
DEFAULT_JITTER = 1e-9
MAX_DIFFICULTY = 10.0

SELECTORS = ("dpp", "difficulty_rank", "random")

_NO_EMBEDDER_REASON = "no embedder supplied; diversity term disabled"


@dataclass(frozen=True, slots=True)
class CoresetCandidate:
    """One historical record eligible for coreset selection.

    ``observed`` mirrors the judge contract: a rejected verdict carries
    ``difficulty=0.0``, which is *indistinguishable by value* from a legitimate
    easy task. Selection therefore gates on ``observed``, never on the value.
    """

    task_id: str
    difficulty: float
    fingerprint: str
    embedding_text: str
    observed: bool = True


@dataclass(frozen=True, slots=True)
class CoresetReport:
    """The selected coreset plus how it was selected."""

    selected_ids: tuple[str, ...] = ()
    selection_method: str = "dpp"
    fallback_reason: str = ""
    excluded_ids: tuple[str, ...] = ()


class _VerdictLike(Protocol):
    """Structural shape of the adapter's ``DifficultyVerdict``."""

    task_id: str
    difficulty: float
    abstract_fingerprint: str
    observed: bool


def candidates_from_verdicts(
    verdicts: Iterable[_VerdictLike],
    summaries: Mapping[str, str] | None = None,
) -> tuple[CoresetCandidate, ...]:
    """Adapt judge verdicts (duck-typed) into coreset candidates.

    ``summaries`` maps ``task_id`` to the comprehended trajectory summary. When
    present it is appended to the fingerprint to form ``embedding_text``, since
    spec §4.2 makes the summary -- never the raw trace -- the embedded text.
    Unobserved verdicts are carried through so the report can name them as
    excluded rather than silently dropping them.
    """
    summaries = summaries or {}
    built: list[CoresetCandidate] = []
    for verdict in verdicts:
        fingerprint = getattr(verdict, "abstract_fingerprint", "") or ""
        summary = summaries.get(verdict.task_id, "")
        embedding_text = "\n".join(part for part in (fingerprint, summary) if part)
        built.append(
            CoresetCandidate(
                task_id=verdict.task_id,
                difficulty=float(getattr(verdict, "difficulty", 0.0) or 0.0),
                fingerprint=fingerprint,
                embedding_text=embedding_text,
                observed=bool(getattr(verdict, "observed", False)),
            )
        )
    return tuple(built)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _quality(difficulty: float, theta: float, score_floor: float) -> float:
    """Floor, normalize to [0,1], then exponentiate by the theta exponent.

    Mirrors ``issues.quality`` so RHO and genetic selection sharpen quality the
    same way; only the input signal differs.
    """
    normalized = max(difficulty / MAX_DIFFICULTY, score_floor)
    alpha = theta / (2 * max(1 - theta, 1e-6)) if theta < 1.0 else 1.0
    return normalized**alpha


def _by_difficulty(
    candidates: Sequence[CoresetCandidate],
) -> list[CoresetCandidate]:
    """Descending difficulty, ascending task id -- fully deterministic."""
    return sorted(candidates, key=lambda c: (-c.difficulty, c.task_id))


def collapse_by_task(
    candidates: Sequence[CoresetCandidate],
) -> tuple[CoresetCandidate, ...]:
    """Collapse several candidates for one ``task_id`` into one, keep-hardest.

    A history corpus routinely holds several traces for the same task, and
    ``candidates_from_verdicts`` emits one candidate per verdict. Left
    uncollapsed, one task can occupy several coreset slots: the round then
    spends the full ``k * (G + N * R)`` rollout budget while gathering evidence
    over fewer than ``k`` tasks, defeating the diversity objective silently.

    Policy is keep-hardest: the highest ``difficulty`` wins, because difficulty
    is the DPP quality term and the hardest observation of a task is the one
    worth diagnosing. Ties break on the longer ``embedding_text`` (more
    mechanism signal for the diversity term) and then on ``fingerprint``, so the
    result is deterministic regardless of input order.
    """
    best: dict[str, CoresetCandidate] = {}
    for candidate in candidates:
        incumbent = best.get(candidate.task_id)
        if incumbent is None:
            best[candidate.task_id] = candidate
            continue
        challenger_key = (
            candidate.difficulty,
            len(candidate.embedding_text),
            candidate.fingerprint,
        )
        incumbent_key = (
            incumbent.difficulty,
            len(incumbent.embedding_text),
            incumbent.fingerprint,
        )
        if challenger_key > incumbent_key:
            best[candidate.task_id] = candidate
    # Preserve first-appearance order so callers see a stable sequence.
    seen: set[str] = set()
    collapsed: list[CoresetCandidate] = []
    for candidate in candidates:
        if candidate.task_id in seen:
            continue
        seen.add(candidate.task_id)
        collapsed.append(best[candidate.task_id])
    return tuple(collapsed)


def select_coreset(
    candidates: Sequence[CoresetCandidate],
    k: int,
    *,
    selector: str = "dpp",
    theta: float = DEFAULT_THETA,
    score_floor: float = DEFAULT_SCORE_FLOOR,
    min_gain: float = DEFAULT_MIN_GAIN,
    seed: int = 0,
    embedder: MechanismEmbedder | None = None,
) -> CoresetReport:
    """Select up to ``k`` task ids from ``candidates``.

    Deterministic for identical inputs: the ``random`` ablation seeds an
    explicit :class:`random.Random`, and every ordering tie-breaks on task id.
    """
    if selector not in SELECTORS:
        raise ValueError(f"unknown selector: {selector!r}; expected one of {SELECTORS}")
    if k < 1:
        raise ValueError("k must be >= 1")
    if not candidates:
        return CoresetReport(selection_method=selector)

    # The observed gate. A verdict the judge could not produce is absent
    # evidence, not evidence of an easy task, so it must not compete.
    eligible = tuple(c for c in candidates if c.observed)
    excluded = tuple(c.task_id for c in candidates if not c.observed)
    if not eligible:
        return CoresetReport(
            selection_method=selector,
            fallback_reason="no observed difficulty verdicts",
            excluded_ids=excluded,
        )

    # One slot per task. Several traces of one task must not occupy several
    # coreset slots, or the round burns k*(G + N*R) rollouts over < k tasks.
    eligible = collapse_by_task(eligible)

    if selector == "difficulty_rank":
        ordered = _by_difficulty(eligible)
        return CoresetReport(
            selected_ids=tuple(c.task_id for c in ordered[:k]),
            selection_method="difficulty_rank",
            excluded_ids=excluded,
        )

    if selector == "random":
        rng = random.Random(seed)
        pool = sorted(eligible, key=lambda c: c.task_id)
        picked = rng.sample(pool, min(k, len(pool)))
        return CoresetReport(
            selected_ids=tuple(c.task_id for c in picked),
            selection_method="random",
            excluded_ids=excluded,
        )

    # --- dpp ---------------------------------------------------------- #
    qualities = [_quality(c.difficulty, theta, score_floor) for c in eligible]
    ids = tuple(c.task_id for c in eligible)

    fallback_reason = ""
    vectors: list[tuple[float, ...]] | None = None
    if embedder is None:
        fallback_reason = _NO_EMBEDDER_REASON
    else:
        try:
            vectors = [embedder.embed(c.embedding_text) for c in eligible]
        except EmbeddingProviderUnavailable as exc:
            # Degrade to quality-only ordering, but record why. Silent
            # substitution would make a degraded run indistinguishable from a
            # healthy one.
            fallback_reason = f"embedding provider unavailable: {exc}"
            vectors = None

    if vectors is None:
        ordered = _by_difficulty(eligible)
        return CoresetReport(
            selected_ids=tuple(c.task_id for c in ordered[:k]),
            selection_method="dpp_quality_only",
            fallback_reason=fallback_reason,
            excluded_ids=excluded,
        )

    def similarity(i: int, j: int) -> float:
        return _cosine(vectors[i], vectors[j])

    kernel = build_kernel(qualities, similarity, DEFAULT_JITTER)
    chosen = greedy_map(kernel, ids, k, min_gain)
    return CoresetReport(
        selected_ids=tuple(ids[i] for i in chosen),
        selection_method="dpp",
        fallback_reason=fallback_reason,
        excluded_ids=excluded,
    )
