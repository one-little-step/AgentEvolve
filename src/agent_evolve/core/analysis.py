"""Bounded analyzer+judge exchange record and protocol.

This module defines the neutral boundary between the orchestrator and the
analyzer+judge component. :class:`RolloutGroupReport` is a frozen snapshot of a
candidate/task rollout group's trace references and sanitized evidence; it is
passed to an :class:`AnalyzerJudge`, which returns a tuple of trace-backed
:class:`~agent_evolve.core.blame.CausalFinding` verdicts.

The generic core never mutates artifacts here and never imports CUGA/Gaia. A
finding whose graph nodes lack trace evidence must not be manufactured; absence
of evidence is expressed via the ``insufficient_evidence`` status.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from agent_evolve.core.blame import CausalFinding


@dataclass(frozen=True, slots=True)
class RolloutGroupReport:
    """Immutable evidence snapshot for one (candidate, task) rollout group."""

    candidate_id: str
    task_id: str
    trace_refs: tuple[str, ...]
    rollout_ids: tuple[str, ...]
    sanitized_evidence: tuple[Mapping[str, object], ...]


class AnalyzerJudge(Protocol):
    """Turns a bounded rollout-group report into trace-backed findings."""

    def analyze(self, report: RolloutGroupReport) -> tuple[CausalFinding, ...]: ...
