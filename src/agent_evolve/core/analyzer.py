"""Analyzer+judge protocol and a deterministic fake implementation.

The analyzer+judge is the LLM-driven component that produces a
:class:`CausalAnalysis` for one (candidate, task, rollout) triple. The
generic core defines the Protocol; an adapter or host application supplies
a concrete implementation backed by a real LLM.

For testing and offline demos, :class:`FakeAnalyzerJudge` produces a
deterministic analysis derived from the trace's ``final_output`` and the
task's ``expected_contract``. It does NOT call any LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from agent_evolve.core.blame import (
    BlameEdge,
    BlameGraph,
    BlameNode,
    CausalAnalysis,
)
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace


class AnalyzerJudge(Protocol):
    """Maps a (task, trace) to a causal analysis verdict."""

    analyzer_model_id: str
    judge_model_id: str

    def analyze(
        self,
        task: EvolutionTask,
        trace: ExecutionTrace,
    ) -> CausalAnalysis: ...


@dataclass(slots=True)
class FakeAnalyzerJudge:
    """Deterministic offline analyzer+judge.

    Scoring rule
    ------------
    * If ``task.expected_contract["expected_substring"]`` is set and appears
      in ``trace.final_output``, score = 1.0 (success).
    * Else if ``task.expected_contract["expected_regex"]`` is set and matches
      ``trace.final_output``, score = 1.0.
    * Else score = 0.0 (failure).

    Blame assignment
    ----------------
    * On success: empty blame graph, severity 0.0.
    * On failure: blame is split across all actors that appear in the trace
      events. The first actor (by event order) gets 0.6 blame, the rest get
      equal shares of 0.4. Severity is 1.0.

    This is deliberately simple and deterministic; production analyzer+judge
    implementations will be backed by LLMs.
    """

    analyzer_model_id: str = "fake-analyzer"
    judge_model_id: str = "fake-judge"

    def analyze(self, task: EvolutionTask, trace: ExecutionTrace) -> CausalAnalysis:
        substring = task.expected_contract.get("expected_substring")
        regex = task.expected_contract.get("expected_regex")

        success = False
        if substring is not None and str(substring) in trace.final_output:
            success = True
        elif regex is not None and re.search(str(regex), trace.final_output):
            success = True

        if success:
            return CausalAnalysis(
                mechanism="none",
                severity=0.0,
                score=1.0,
                blame_graph=BlameGraph(nodes=()),
                analyzer_model_id=self.analyzer_model_id,
                judge_model_id=self.judge_model_id,
            )

        # Failure: blame actors from the trace.
        actors: list[str] = []
        for e in trace.events:
            if e.actor_id and e.actor_id not in actors:
                actors.append(e.actor_id)
        if not actors:
            actors = ["unknown"]

        nodes: list[BlameNode] = []
        if len(actors) == 1:
            nodes.append(BlameNode(actor_id=actors[0], blame=1.0, artifacts=()))
        else:
            nodes.append(BlameNode(actor_id=actors[0], blame=0.6, artifacts=()))
            rest_share = 0.4 / (len(actors) - 1)
            for a in actors[1:]:
                nodes.append(BlameNode(actor_id=a, blame=rest_share, artifacts=()))

        edges: list[BlameEdge] = []
        for i in range(len(actors) - 1):
            edges.append(
                BlameEdge(
                    from_actor=actors[i],
                    to_actor=actors[i + 1],
                    mechanism=f"chain-{i}",
                )
            )

        return CausalAnalysis(
            mechanism=f"trace-{trace.trace_id}-failed-to-match",
            severity=1.0,
            score=0.0,
            blame_graph=BlameGraph(nodes=tuple(nodes), edges=tuple(edges)),
            analyzer_model_id=self.analyzer_model_id,
            judge_model_id=self.judge_model_id,
        )
