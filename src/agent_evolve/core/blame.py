"""Causal blame graph data model.

The analyzer+judge emits dynamic failure mechanisms and causal blame graphs.
This module is pure data + small query helpers; the analyzer+judge caller
(see :mod:`agent_evolve.core.editor` and :mod:`agent_evolve.core.orchestrator`)
is responsible for producing instances.

Design rules (from docs/architecture/target-rho-parallel-gepa.md):
* Mechanisms are free-form strings; no fixed taxonomy.
* Each blame node carries a blame weight in [0, 1] and the artifacts it owns.
* Edges carry a mechanism description so causal chains are auditable.
* ``counterfactual_evidence`` is a list of human/LLM-readable claims, not a
  structured counterfactual result; structured counterfactuals require an
  adapter that supports replay (see EvolutionAdapter.supports_counterfactual_replay).
* ``severity`` and ``score`` are in [0, 1]; severity is the analyzer's judged
  impact, score is the measured task outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _check_unit_interval(name: str, value: float) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    v = float(value)
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {v}")
    return v


@dataclass(frozen=True, slots=True)
class BlameNode:
    """One actor in a causal blame graph (tool, subagent, retriever, etc.)."""

    actor_id: str
    blame: float
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.actor_id:
            raise ValueError("actor_id is required")
        object.__setattr__(self, "blame", _check_unit_interval("blame", self.blame))
        object.__setattr__(
            self, "artifacts", tuple(self.artifacts) if self.artifacts else ()
        )


@dataclass(frozen=True, slots=True)
class BlameEdge:
    """A directed causal link between two actors."""

    from_actor: str
    to_actor: str
    mechanism: str

    def __post_init__(self) -> None:
        if not self.from_actor or not self.to_actor:
            raise ValueError("from_actor and to_actor are required")
        if not self.mechanism:
            raise ValueError("mechanism is required")


@dataclass(frozen=True, slots=True)
class BlameGraph:
    """The graph itself: nodes + edges."""

    nodes: tuple[BlameNode, ...]
    edges: tuple[BlameEdge, ...] = ()

    def __post_init__(self) -> None:
        node_ids = {n.actor_id for n in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("duplicate actor_id in blame nodes")
        for e in self.edges:
            if e.from_actor not in node_ids or e.to_actor not in node_ids:
                raise ValueError(
                    f"edge {e.from_actor}->{e.to_actor} references unknown actor"
                )

    def artifacts_for(self, actor_id: str) -> tuple[str, ...]:
        for n in self.nodes:
            if n.actor_id == actor_id:
                return n.artifacts
        return ()

    def total_blame(self) -> float:
        """Sum of blame weights; not normalized."""
        return sum(n.blame for n in self.nodes)

    def top_blame_artifacts(self, k: int) -> tuple[str, ...]:
        """Return up to k artifact IDs sorted by their owner's blame weight.

        Ties broken by artifact_id for determinism.
        """
        if k < 0:
            raise ValueError("k must be >= 0")
        pairs: list[tuple[float, str]] = []
        for n in self.nodes:
            for aid in n.artifacts:
                pairs.append((n.blame, aid))
        pairs.sort(key=lambda p: (-p[0], p[1]))
        return tuple(aid for _, aid in pairs[:k])


@dataclass(frozen=True, slots=True)
class CausalAnalysis:
    """A single analyzer+judge verdict for one candidate/task rollout."""

    mechanism: str
    severity: float
    score: float
    blame_graph: BlameGraph
    counterfactual_evidence: tuple[str, ...] = ()
    analyzer_model_id: str = ""
    judge_model_id: str = ""

    def __post_init__(self) -> None:
        if not self.mechanism:
            raise ValueError("mechanism is required")
        object.__setattr__(self, "severity", _check_unit_interval("severity", self.severity))
        object.__setattr__(self, "score", _check_unit_interval("score", self.score))
        object.__setattr__(
            self,
            "counterfactual_evidence",
            tuple(self.counterfactual_evidence) if self.counterfactual_evidence else (),
        )

    @property
    def artifact_ids(self) -> tuple[str, ...]:
        """All artifacts referenced anywhere in the blame graph."""
        seen: list[str] = []
        for n in self.blame_graph.nodes:
            for aid in n.artifacts:
                if aid not in seen:
                    seen.append(aid)
        return tuple(seen)

    @property
    def actor_ids(self) -> tuple[str, ...]:
        return tuple(n.actor_id for n in self.blame_graph.nodes)


class CausalFinding(BaseModel):
    """A trace-backed causal finding with a validated status.

    Unlike :class:`CausalAnalysis` (a plain data record), this model enforces
    the data contract for analyzer+judge findings (docs/architecture/
    data-contracts.md:81-104). ``observed`` findings must carry non-empty
    mechanism attribution and trace-backed evidence references; other statuses
    may leave the trace-specific fields unset. A graph node without trace
    evidence is never synthesized.

    ``verdict_id``, ``candidate_id``, ``task_id``, ``trace_id``, and
    ``rationale`` are genuinely required (no defaults): an omitted field does
    not silently fall back to an empty string.

    For ``status == "observed"``, every artifact ID referenced by any
    :class:`BlameNode.artifacts` in ``blame_graph`` must be present in
    ``evidence_refs`` (the trace-backed reference set). A node attributing an
    artifact with no evidence reference is invalid and raises ``ValidationError``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    verdict_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    status: Literal["observed", "uncertain", "insufficient_evidence", "malformed"]
    mechanism_description: str | None = None
    mechanism_cluster_id: str | None = None
    severity: float | None = None
    confidence: float | None = None
    blame_graph: BlameGraph = Field(default_factory=lambda: BlameGraph(nodes=()))
    evidence_refs: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)
    counterfactual_notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> "CausalFinding":
        for name in ("severity", "confidence"):
            value = getattr(self, name)
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        if any(not ref.strip() for ref in self.evidence_refs):
            raise ValueError("evidence_refs contains blank IDs")
        if self.status == "observed":
            missing: list[str] = []
            if not self.mechanism_description:
                missing.append("mechanism_description")
            if not self.mechanism_cluster_id:
                missing.append("mechanism_cluster_id")
            if self.severity is None:
                missing.append("severity")
            if self.confidence is None:
                missing.append("confidence")
            if not self.evidence_refs:
                missing.append("evidence_refs")
            if missing:
                raise ValueError(
                    f"status=observed requires: {', '.join(missing)}"
                )
            referenced = {
                aid
                for node in self.blame_graph.nodes
                for aid in node.artifacts
            }
            unbacked = sorted(referenced - set(self.evidence_refs))
            if unbacked:
                raise ValueError(
                    "status=observed blame_graph references artifacts without "
                    f"trace evidence: {', '.join(unbacked)}"
                )
        return self


def empty_analysis() -> CausalAnalysis:
    """An analysis with no blame, used for successful rollouts."""
    return CausalAnalysis(
        mechanism="none",
        severity=0.0,
        score=1.0,
        blame_graph=BlameGraph(nodes=()),
    )


def merge_analyses(analyses: Iterable[CausalAnalysis]) -> CausalAnalysis:
    """Combine multiple analyses of the same rollout into one verdict.

    Used when the consensus/calibration ablation is enabled (default is one
    call per the architecture doc, so this is rarely invoked). Severity and
    score are averaged; blame weights are summed and clipped to [0, 1];
    mechanisms are concatenated into a single string.
    """
    analyses = list(analyses)
    if not analyses:
        raise ValueError("cannot merge an empty analysis list")
    if len(analyses) == 1:
        return analyses[0]

    severity = sum(a.severity for a in analyses) / len(analyses)
    score = sum(a.score for a in analyses) / len(analyses)
    mechanisms = " | ".join(sorted({a.mechanism for a in analyses}))

    # Sum blame per actor across all analyses, clip to 1.0.
    by_actor: dict[str, float] = {}
    artifacts_by_actor: dict[str, set[str]] = {}
    for a in analyses:
        for n in a.blame_graph.nodes:
            by_actor[n.actor_id] = by_actor.get(n.actor_id, 0.0) + n.blame
            artifacts_by_actor.setdefault(n.actor_id, set()).update(n.artifacts)

    nodes = tuple(
        BlameNode(
            actor_id=aid,
            blame=min(1.0, by_actor[aid]),
            artifacts=tuple(sorted(artifacts_by_actor[aid])),
        )
        for aid in sorted(by_actor)
    )
    # Edges: union, preserving mechanism strings.
    seen: set[tuple[str, str, str]] = set()
    edges: list[BlameEdge] = []
    for a in analyses:
        for e in a.blame_graph.edges:
            key = (e.from_actor, e.to_actor, e.mechanism)
            if key not in seen:
                seen.add(key)
                edges.append(e)

    counterfactuals: list[str] = []
    for a in analyses:
        for c in a.counterfactual_evidence:
            if c not in counterfactuals:
                counterfactuals.append(c)

    return CausalAnalysis(
        mechanism=mechanisms,
        severity=severity,
        score=score,
        blame_graph=BlameGraph(nodes=nodes, edges=tuple(edges)),
        counterfactual_evidence=tuple(counterfactuals),
    )
