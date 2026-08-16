"""Bounded parallel fan-out for trajectory analysis.

Analyzing distinct (candidate, task) rollout groups is independent work: each
analysis reads one immutable :class:`RolloutGroupReport` and returns findings
without touching shared state. That makes it safe to parallelize, and analysis
is LLM-latency-bound, so the speedup is close to linear in worker count.

Why this module exists separately from :mod:`agent_evolve.core.parallel`
------------------------------------------------------------------------
``parallel.py`` coordinates parallel *edits*: it exists to make concurrent
artifact **writes** safe (snapshots, exclusive write leases, commit barriers).
Analysis performs no writes, so none of that machinery applies. Sharing the
lease manager here would impose write-serialization on read-only work.

Design constraints this module encodes
--------------------------------------
* **One analyzer per worker thread.** The analyzer is expected to become a CUGA
  agent, which carries conversation state across ``analyze`` calls. Sharing one
  such agent across threads would interleave two trajectories into a single
  conversation. Callers therefore pass an ``analyzer_factory``, not an analyzer,
  and each worker thread builds exactly one instance and reuses it (agent
  construction is expensive, so per-item construction is wasteful).
* **Input order out, not completion order.** Downstream clustering and entropy
  accounting must not vary with thread scheduling.
* **Per-item failure isolation.** One malformed trajectory or one model error
  must not discard the whole batch's findings. Failures are returned as data
  (``ok=False`` plus ``error``), not raised.
* **No budget accounting here.** Workers do not touch the shared ledger; the
  caller charges budget when it consumes the returned outcomes, on the
  coordinator thread. Concurrent ledger mutation would be a race.

This module is agent-neutral and imports no agent implementation.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from agent_evolve.core.analysis import RolloutGroupReport
from agent_evolve.core.blame import CausalFinding


class _Analyzer(Protocol):
    def analyze(self, report: RolloutGroupReport) -> tuple[CausalFinding, ...]: ...


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    """The result of analyzing one rollout group.

    A failure is data, not an exception: ``ok=False`` carries a non-empty
    ``error`` and no findings, so a caller can record the gap (and, if it
    chooses, retry or mark the cell as lacking evidence) without losing the
    findings from sibling items.
    """

    report: RolloutGroupReport
    findings: tuple[CausalFinding, ...]
    error: str
    ok: bool

    def __post_init__(self) -> None:
        if self.ok:
            if self.error:
                raise ValueError("a successful outcome must not carry an error")
        else:
            if not self.error:
                raise ValueError("a failed outcome requires a non-empty error")
            if self.findings:
                raise ValueError("a failed outcome must not carry findings")


@dataclass(slots=True)
class ParallelAnalysisRunner:
    """Runs ``analyzer.analyze`` over many reports with bounded concurrency.

    ``max_workers=1`` runs inline on the calling thread (no executor, no worker
    threads) so sequential debugging shows a straightforward stack.
    """

    analyzer_factory: Callable[[], _Analyzer]
    max_workers: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.max_workers, bool) or not isinstance(self.max_workers, int):
            raise ValueError("max_workers must be a positive integer")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def run(
        self, reports: Sequence[RolloutGroupReport]
    ) -> tuple[AnalysisOutcome, ...]:
        """Analyze every report, returning outcomes in input order.

        Never raises for a per-item analyzer failure; see :class:`AnalysisOutcome`.
        """
        if not reports:
            return ()

        if self.max_workers == 1:
            analyzer_holder: list[_Analyzer] = []
            return tuple(
                self._analyze_one(report, analyzer_holder) for report in reports
            )

        # One analyzer per worker thread, built lazily on first use by that
        # thread and reused for every item it handles.
        local = threading.local()

        def work(report: RolloutGroupReport) -> AnalysisOutcome:
            holder = getattr(local, "holder", None)
            if holder is None:
                holder = []
                local.holder = holder
            return self._analyze_one(report, holder)

        workers = min(self.max_workers, len(reports))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="analyzer"
        ) as pool:
            # executor.map preserves input order regardless of completion order.
            return tuple(pool.map(work, reports))

    def _analyze_one(
        self,
        report: RolloutGroupReport,
        holder: list[_Analyzer],
    ) -> AnalysisOutcome:
        """Analyze one report, converting any failure into a recorded outcome.

        ``holder`` caches this thread's analyzer. Construction failure is
        isolated per item too: a factory that needs model credentials should
        report a missing-configuration error, not abort the batch.
        """
        try:
            if not holder:
                holder.append(self.analyzer_factory())
            analyzer = holder[0]
            findings = tuple(analyzer.analyze(report))
        except Exception as exc:  # noqa: BLE001 - failures are returned as data
            return AnalysisOutcome(
                report=report,
                findings=(),
                error=f"{type(exc).__name__}: {exc}",
                ok=False,
            )
        return AnalysisOutcome(report=report, findings=findings, error="", ok=True)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def flatten(
        outcomes: Sequence[AnalysisOutcome],
    ) -> tuple[CausalFinding, ...]:
        """One ordered stream of findings from successful outcomes."""
        return tuple(f for o in outcomes if o.ok for f in o.findings)

    @staticmethod
    def failures(
        outcomes: Sequence[AnalysisOutcome],
    ) -> tuple[AnalysisOutcome, ...]:
        """The failed outcomes, so a caller can log or retry them."""
        return tuple(o for o in outcomes if not o.ok)
