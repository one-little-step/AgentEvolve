"""Final candidate resolution (SV-13d).

Agent-neutral: the preference judge arrives as an injected ``compare`` callable,
the same seam ``core/rho/rounds.py`` and ``core/retirement.py`` use. This module
must never import an adapter, ``cuga``, or ``litellm``.

**Two paths, in order.**

1. **Sole survivor.** Generational retirement (SV-13) removes a parent from the
   breeding population once its offspring supersedes it. If that leaves exactly one
   live candidate, it is the winner outright -- there is nothing to compare it
   against, so no judge call is spent.
2. **Symmetric ladder.** Otherwise survivors are resolved by pairwise preference
   over the coreset. A linear king-of-the-hill: an incumbent meets each challenger
   once, ``N-1`` comparisons for ``N`` survivors. Since each comparison is the
   symmetric two-call form, that is ``2(N-1)`` model calls -- deliberately not the
   ``N choose 2`` a full round-robin would cost.

**Why preference and not the score ranking.** When this module was written
``select_champion`` ranked by a weighted aggregate whose ``outcome`` term averaged
over whatever cells each candidate happened to measure, so::

    base     outcome=0.5000 coverage=1.0000 aggregate=0.6250
    cand-A   outcome=0.9000 coverage=0.5000 aggregate=0.7450   <- would be exported

``cand-A`` never ran the hard task. It won because breadth of measurement was
scored as quality (SV-3) and unequal task sets were averaged as if comparable
(SV-2).

Both are now fixed in the pool: ranking is a pairwise comparison over the cells
two candidates share, so that outcome is no longer expressible there either. This
module remains the *preferred* resolver for a different reason -- it reads
trajectories rather than scores, so it can prefer a candidate whose numbers tie
but whose reasoning is sound. The score path is the fallback, not the wrong
answer.

**Determinism.** The ladder walks live candidates in insertion order, base first. An
LLM judge is not perfectly transitive, so the winner can depend on comparison order;
fixing the order makes a run reproducible from the pool alone.

**Fallback is explicit.** With no judge, an unavailable verdict, or a raising judge,
resolution falls back to ``select_champion`` and *says so* in
``FinalResolution.method``. The string is ``aggregate_fallback`` for backward
compatibility with existing manifests, though what it now runs is the pairwise
score comparison. Silently returning the first candidate in a list would make a
judge outage indistinguishable from a considered verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from agent_evolve.core.config import ResolvedConfig
from agent_evolve.core.pool import PersistentPool

__all__ = ["FinalResolution", "resolve_final_candidate"]


class _HasTaskId(Protocol):
    """Anything carrying a task id. Read-only, so a frozen task satisfies it."""

    @property
    def task_id(self) -> str: ...


CompareFn = Callable[[Any, Any, Any], Any]


@dataclass(frozen=True, slots=True)
class FinalResolution:
    """Who won, how, and what it cost. Pure data; the pool is not mutated."""

    candidate_id: str
    #: ``sole_survivor`` | ``pairwise_ladder`` | ``aggregate_fallback``
    method: str
    reason: str
    comparisons: int = 0
    judge_calls: int = 0


def _aggregate_fallback(
    pool: PersistentPool,
    reason: str,
    config: ResolvedConfig | None = None,
) -> FinalResolution:
    """Fall back to :meth:`PersistentPool.select_champion`, naming it as a fallback.

    When ``select_champion`` itself has no eligible candidate -- every proposal
    was gated or lacks comparable evidence -- the answer is the **base**, never
    the first live entry. ``live[0]`` would hand the run to whichever candidate
    happened to be retained, including one the SV-4 gate just disqualified, which
    is precisely the promotion the gate exists to refuse. The base is always
    runnable and is what the next run would execute against anyway.

    **The name is historical.** ``method="aggregate_fallback"`` is retained because
    exported manifests and their readers already key on that string. Since SV-2 the
    fallback no longer ranks by the weighted aggregate: ``select_champion`` compares
    candidates pairwise on the cells both measured. So this path is weaker than the
    judge ladder only in that it reads *scores* rather than trajectories -- it is no
    longer the structurally-wrong ranking the name implies.

    **``config`` is forwarded, and that is not cosmetic.** Omitting it reverted
    every ``champion_*`` setting to its dataclass default, including
    ``champion_min_coverage_fraction``, which then became ``0.0``. The weights are
    only reported, so losing them merely corrupts the manifest; losing the coverage
    floor *readmits a candidate the operator disqualified*. Worse, this path runs
    precisely when the preference judge is unavailable -- outage, raised exception,
    unavailable verdict -- so the floor was discarded at the one moment it was the
    only guard still standing.
    """
    try:
        champion_id = pool.select_champion(config=config).entry.candidate_id
    except ValueError:
        champion_id = pool.base.candidate_id
    return FinalResolution(
        candidate_id=champion_id,
        method="aggregate_fallback",
        reason=reason,
        comparisons=0,
        judge_calls=0,
    )


def _is_promotable(pool: PersistentPool, candidate_id: str) -> bool:
    """Whether ``candidate_id`` may win at all (the SV-4 acceptance gate).

    The base is exempt: it is the incumbent being compared against, never a
    proposal competing for promotion, and gating it would leave nothing to fall
    back to. A non-base candidate needs a *measured positive* preference; ``None``
    (never judged) and ``<= 0`` (judged and not preferred) are both ineligible,
    exactly as ``PersistentPool.select_champion`` treats them.
    """
    entry = pool.get(candidate_id)
    if entry.is_base:
        return True
    preference = entry.preference
    return preference is not None and preference > 0.0


def resolve_final_candidate(
    pool: PersistentPool,
    *,
    tasks: Sequence[_HasTaskId],
    traces: Mapping[str, Mapping[str, Any]],
    compare: CompareFn | None,
    config: ResolvedConfig | None = None,
) -> FinalResolution:
    """Resolve the live pool to a single winner.

    ``traces`` maps ``candidate_id -> {task_id: trace}``. A candidate with no
    usable trace pair against the incumbent is skipped rather than treated as a
    loss: an unrollable candidate has produced no evidence either way, and
    scoring it zero would promote whoever happened to be measured.

    ``config`` is used only by the aggregate fallback, but it matters there:
    without it the operator's ``champion_min_coverage_fraction`` silently becomes
    ``0.0`` on the exact path a judge outage forces. Optional, so a caller with no
    resolved profile still works.
    """
    live = pool.live_candidate_ids()
    if not live:
        raise ValueError("cannot resolve a final candidate: no candidates in pool")

    # SV-4 must survive SV-13d. A candidate the RHO judge already dispreferred
    # (``preference <= 0``) or never judged (``preference is None``) is not
    # promotable, and re-judging it here would let a second opinion overturn the
    # acceptance gate -- the exported-harness defect SV-4 exists to prevent.
    #
    # Filtering happens *before* the sole-survivor short-circuit, or retiring a
    # parent could leave an ineligible candidate alone in the pool and hand it the
    # win by default.
    eligible = tuple(cid for cid in live if _is_promotable(pool, cid))
    if not eligible:
        return _aggregate_fallback(
            pool,
            "no live candidate passes the pairwise acceptance gate; ranked by "
            "aggregate instead",
            config,
        )
    live = eligible

    if len(live) == 1:
        return FinalResolution(
            candidate_id=live[0],
            method="sole_survivor",
            reason=(
                "one live candidate remains after generational retirement; it is "
                "the winner outright and no comparison is needed"
            ),
        )

    if compare is None:
        return _aggregate_fallback(
            pool, "no preference judge supplied; ranked by aggregate instead", config
        )
    if not tasks:
        return _aggregate_fallback(
            pool, "no tasks to compare on; ranked by aggregate instead", config
        )

    incumbent = live[0]
    comparisons = 0
    judge_calls = 0
    decided = False

    for challenger in live[1:]:
        held = traces.get(incumbent)
        bid = traces.get(challenger)
        if not held or not bid:
            # One side was never rolled out; no evidence either way.
            continue
        scores: list[float] = []
        for task in tasks:
            baseline = held.get(task.task_id)
            candidate = bid.get(task.task_id)
            if baseline is None or candidate is None:
                continue
            try:
                verdict = compare(task, baseline, candidate)
            except Exception as exc:  # noqa: BLE001 - never lose the run to a judge fault
                return _aggregate_fallback(
                    pool,
                    f"judge error while resolving {challenger!r}: {exc}; "
                    "ranked by aggregate instead",
                    config,
                )
            judge_calls += 1
            if not bool(getattr(verdict, "available", False)):
                return _aggregate_fallback(
                    pool,
                    f"verdict unavailable while resolving {challenger!r}; "
                    "ranked by aggregate instead",
                    config,
                )
            scores.append(float(getattr(verdict, "score", 0.0)))
        if not scores:
            continue
        comparisons += 1
        decided = True
        # Strict '> 0': the incumbent holds a tie. Consistent with the SV-4
        # promotion gate and with retirement -- a tie is not evidence of
        # improvement, so it must not change the standing answer.
        if sum(scores) / len(scores) > 0.0:
            incumbent = challenger

    if not decided:
        return _aggregate_fallback(
            pool,
            "no comparable trace pair among survivors; ranked by aggregate instead",
            config,
        )

    return FinalResolution(
        candidate_id=incumbent,
        method="pairwise_ladder",
        reason=(
            f"survived {comparisons} symmetric pairwise comparison(s) over "
            f"{len(tasks)} task(s)"
        ),
        comparisons=comparisons,
        judge_calls=judge_calls,
    )
