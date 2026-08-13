"""Task-local incremental mechanism clustering.

Free-form analyzer+judge mechanisms are embedded with task, phase/tool,
artifact, and counterfactual context. Base-harness mechanisms provide anchors.
Task-local incremental clusters assign ``mechanism_cluster_id``, which is the
cross-candidate alignment key.

Clusters are stable inside an outer iteration. New observations may join an
existing cluster; cluster create/merge/split occurs at refresh barriers. Track
cluster freshness and reduce entropy weight when evidence is stale.

Implementation note
-------------------
This module deliberately does NOT depend on any embedding model. It exposes
a :class:`MechanismEmbedder` Protocol so an adapter or host application can
plug in a real embedder (e.g., sentence-transformers, an OpenAI embedding
endpoint, or the ``z-ai-web-dev-sdk`` VLM/LLM embeddings). The default
:class:`LexicalEmbedder` is a deterministic tf-idf-style bag-of-tokens embedder
suitable for tests and offline smoke runs. It is NOT a research-grade
embedder and must not be used to draw research conclusions.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

from agent_evolve.core.blame import CausalAnalysis, CausalFinding


def _tokenize(text: str) -> tuple[str, ...]:
    """Lowercase, split on non-alphanumeric, drop empties and 1-char tokens."""
    out: list[str] = []
    cur: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                tok = "".join(cur)
                if len(tok) > 1:
                    out.append(tok)
                cur = []
    if cur:
        tok = "".join(cur)
        if len(tok) > 1:
            out.append(tok)
    return tuple(out)


class MechanismEmbedder(Protocol):
    """Maps a mechanism string (with context) to a fixed-length vector."""

    dim: int

    def embed(self, text: str) -> tuple[float, ...]: ...


class LexicalEmbedder:
    """Deterministic lexical embedder for tests and offline demos.

    Builds a vocab from all tokens it has ever seen (in token-id order, sorted
    for determinism), then emits a sparse->dense L2-normalized vector.
    Vocab grows over the lifetime of the embedder; subsequent calls see the
    expanded vocab.
    """

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.dim = dim
        self._vocab: dict[str, int] = {}

    def _token_id(self, token: str) -> int:
        # Hash into [0, dim) deterministically so we don't have to grow the
        # vector length. Collisions are acceptable for a test embedder.
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big") % self.dim

    def embed(self, text: str) -> tuple[float, ...]:
        tokens = _tokenize(text)
        if not tokens:
            return tuple(0.0 for _ in range(self.dim))
        counts: list[float] = [0.0] * self.dim
        for tok in tokens:
            counts[self._token_id(tok)] += 1.0
        # L2 normalize.
        norm = math.sqrt(sum(c * c for c in counts))
        if norm == 0.0:
            return tuple(0.0 for _ in range(self.dim))
        return tuple(c / norm for c in counts)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    """Where one mechanism observation landed."""

    cluster_id: str
    similarity: float  # in [-1, 1]; cosine to the cluster centroid
    is_new_cluster: bool
    task_id: str = ""
    freshness_iteration: int = 0
    embedding_fallback_reason: str | None = None


@dataclass(slots=True)
class _Cluster:
    cluster_id: str
    centroid: list[float]
    member_count: int = 0
    last_touched_iter: int = 0


@dataclass(slots=True)
class MechanismClusterer:
    """Task-local incremental clusterer.

    Each instance is scoped to one task. Anchors are the base-harness
    mechanisms; candidate mechanisms are matched against the existing
    centroids and either join (similarity >= ``join_threshold``) or spawn a
    new cluster.
    """

    embedder: MechanismEmbedder
    task_id: str = "default"
    join_threshold: float = 0.75
    max_clusters_per_task: int = 12
    _clusters: dict[str, _Cluster] = field(default_factory=dict)
    _next_id: int = 0
    _current_iter: int = 0
    _fallback_embedder: LexicalEmbedder | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.join_threshold <= 1.0:
            raise ValueError("join_threshold must be in [0, 1]")
        if not self.task_id:
            raise ValueError("task_id is required")
        if (
            isinstance(self.max_clusters_per_task, bool)
            or not isinstance(self.max_clusters_per_task, int)
            or self.max_clusters_per_task < 1
        ):
            raise ValueError("max_clusters_per_task must be a positive integer")

    # ------------------------------------------------------------------ #
    # Iteration barriers
    # ------------------------------------------------------------------ #
    def begin_iteration(self, iteration: int) -> None:
        if iteration < 0:
            raise ValueError("iteration must be >= 0")
        if iteration <= self._current_iter:
            raise ValueError(
                f"iteration must increase: current={self._current_iter}, got={iteration}"
            )
        self._current_iter = iteration

    @property
    def current_iteration(self) -> int:
        return self._current_iter

    def refresh_barriers(self) -> None:
        """Called at a refresh barrier.

        Per the architecture doc, cluster create/merge/split occurs at refresh
        barriers. We do not implement automatic merge/split here (it would
        require a full clustering pass over all members); we expose this hook
        so the orchestrator can mark a barrier, and subclasses can override.
        """
        # Hook for future work: re-cluster from scratch if drift detected.
        return None

    # ------------------------------------------------------------------ #
    # Anchors
    # ------------------------------------------------------------------ #
    def add_anchor(self, mechanism: str) -> ClusterAssignment:
        """Add a base-harness mechanism as a new cluster anchor."""
        return self._add(mechanism, force_new=True)

    # ------------------------------------------------------------------ #
    # Incremental assignment
    # ------------------------------------------------------------------ #
    def assign(self, analysis: CausalAnalysis | CausalFinding) -> ClusterAssignment:
        """Assign one analysis's mechanism to a cluster.

        The embedder input is the mechanism string joined with the actor and
        artifact context, mirroring the architecture doc's requirement that
        embedding carry "task, phase/tool, artifact, and counterfactual
        context".
        """
        if isinstance(analysis, CausalFinding):
            return self.assign_finding(analysis)
        text = self._embed_text(analysis)
        return self._add(text, force_new=False)

    def assign_finding(self, finding: CausalFinding) -> ClusterAssignment:
        """Assign a trace-backed :class:`CausalFinding` to a cluster."""
        text = self._embed_text_finding(finding)
        return self._add(text, force_new=False)

    def _embed_text(self, analysis: CausalAnalysis) -> str:
        parts = [analysis.mechanism]
        for n in analysis.blame_graph.nodes:
            parts.append(n.actor_id)
            parts.extend(n.artifacts)
        parts.extend(analysis.counterfactual_evidence)
        return " ".join(parts)

    def _embed_text_finding(self, finding: CausalFinding) -> str:
        parts: list[str] = []
        if finding.mechanism_description:
            parts.append(finding.mechanism_description)
        for n in finding.blame_graph.nodes:
            parts.append(n.actor_id)
            parts.extend(n.artifacts)
        parts.extend(finding.counterfactual_notes)
        return " ".join(parts)

    def _embed(self, text: str) -> tuple[list[float], str | None]:
        """Embed text, falling back to a lexical embedder on provider failure."""
        try:
            return list(self.embedder.embed(text)), None
        except Exception:
            if self._fallback_embedder is None:
                self._fallback_embedder = LexicalEmbedder()
            return list(self._fallback_embedder.embed(text)), "provider_unavailable"

    def _add(self, text: str, force_new: bool) -> ClusterAssignment:
        vec, fallback_reason = self._embed(text)
        if not force_new and self._clusters:
            best_id, best_sim = self._best_match(vec)
            at_cap = len(self._clusters) >= self.max_clusters_per_task
            if best_sim >= self.join_threshold or at_cap:
                self._update_cluster(best_id, vec)
                return ClusterAssignment(
                    cluster_id=best_id,
                    similarity=best_sim,
                    is_new_cluster=False,
                    task_id=self.task_id,
                    freshness_iteration=self._current_iter,
                    embedding_fallback_reason=fallback_reason,
                )
        # New cluster.
        cluster_id = f"c{self._next_id}"
        self._next_id += 1
        self._clusters[cluster_id] = _Cluster(
            cluster_id=cluster_id,
            centroid=vec,
            member_count=1,
            last_touched_iter=self._current_iter,
        )
        return ClusterAssignment(
            cluster_id=cluster_id,
            similarity=1.0,
            is_new_cluster=True,
            task_id=self.task_id,
            freshness_iteration=self._current_iter,
            embedding_fallback_reason=fallback_reason,
        )

    def _best_match(self, vec: list[float]) -> tuple[str, float]:
        best_id = ""
        best_sim = -2.0
        for cid, cluster in self._clusters.items():
            sim = _cosine(vec, cluster.centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = cid
        return best_id, best_sim

    def _update_cluster(self, cluster_id: str, vec: list[float]) -> None:
        c = self._clusters[cluster_id]
        n = c.member_count
        # Running mean update: new_centroid = (old*n + vec) / (n+1)
        c.centroid = [(n * c.centroid[i] + vec[i]) / (n + 1) for i in range(len(vec))]
        c.member_count = n + 1
        c.last_touched_iter = self._current_iter

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    @property
    def cluster_count(self) -> int:
        return len(self._clusters)

    def cluster_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._clusters.keys()))

    def cluster_size(self, cluster_id: str) -> int:
        if cluster_id not in self._clusters:
            raise KeyError(cluster_id)
        return self._clusters[cluster_id].member_count

    def cluster_freshness(self, cluster_id: str) -> int:
        """How stale a cluster is, in iterations.

        Returns ``current_iteration - last_touched_iter``. Higher = staler.
        Stale clusters should have their entropy weight reduced per the
        architecture doc.
        """
        if cluster_id not in self._clusters:
            raise KeyError(cluster_id)
        return self._current_iter - self._clusters[cluster_id].last_touched_iter


@dataclass(slots=True)
class ClusterRegistry:
    """Holds one clusterer per task; the orchestrator-level container."""

    embedder_factory: "callable"
    join_threshold: float = 0.75
    _clusterers: dict[str, MechanismClusterer] = field(default_factory=dict)

    def clusterer_for(self, task_id: str) -> MechanismClusterer:
        if task_id not in self._clusterers:
            self._clusterers[task_id] = MechanismClusterer(
                task_id=task_id,
                embedder=self.embedder_factory(),
                join_threshold=self.join_threshold,
            )
        return self._clusterers[task_id]

    def assign(self, task_id: str, finding: CausalFinding) -> ClusterAssignment:
        """Assign a finding via the per-task clusterer, namespaced by task."""
        assignment = self.clusterer_for(task_id).assign_finding(finding)
        return ClusterAssignment(
            cluster_id=f"{task_id}:{assignment.cluster_id}",
            similarity=assignment.similarity,
            is_new_cluster=assignment.is_new_cluster,
            task_id=task_id,
            freshness_iteration=assignment.freshness_iteration,
            embedding_fallback_reason=assignment.embedding_fallback_reason,
        )

    def begin_iteration(self, iteration: int) -> None:
        for c in self._clusterers.values():
            c.begin_iteration(iteration)

    def refresh_barriers(self) -> None:
        for c in self._clusterers.values():
            c.refresh_barriers()

    def all_cluster_ids(self) -> tuple[str, ...]:
        out: list[str] = []
        for task_id in sorted(self._clusterers.keys()):
            for cid in self._clusterers[task_id].cluster_ids():
                out.append(f"{task_id}:{cid}")
        return tuple(out)
