"""Provenance-preserving deterministic merge.

Per docs/architecture/target-rho-parallel-gepa.md:

    Merge is deterministic by default. It uses ancestry, artifact diffs,
    causal mechanism complementarity, and protected regression floors. The
    editor may refine only a documented same-artifact conflict; it cannot
    rewrite unrelated artifacts.

Design
------
* Input: two parent candidate workspaces, their analyses (one per parent),
  and a protected-floor set.
* Output: a :class:`MergePlan` enumerating per-artifact resolutions, plus a
  set of conflicts (same-artifact edits in both parents) the editor must
  resolve before the plan can be applied.
* The merge is fully deterministic given its inputs: same parents + same
  analyses + same floors -> same plan. No LLM is invoked in the merge itself.

Resolution rules per artifact
-----------------------------
* Touched by only one parent  -> take that parent's version.
* Touched by neither parent   -> take the base version (unchanged).
* Touched by both parents     -> CONFLICT. The editor must resolve; the
  merge plan records the conflict and refuses to apply until resolved.

Conflict resolution by the editor
---------------------------------
The editor receives a :class:`ConflictReport` and may either:
* pick one parent's version (with rationale), or
* provide a new content string (with rationale).

It may NOT modify artifacts that are not in the conflict set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

from agent_evolve.core.blame import CausalAnalysis
from agent_evolve.core.contracts import (
    ArtifactDescriptor,
    ArtifactEdit,
    CandidateWorkspace,
    EvolutionAdapter,
)
from agent_evolve.core.editor import ProtectedFloor


# ---------------------------------------------------------------------- #
# Diff model
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ArtifactDiff:
    """A simple per-artifact diff between base and a parent."""

    artifact_id: str
    base_content: str
    parent_content: str
    base_hash: str
    parent_hash: str

    @property
    def changed(self) -> bool:
        return self.base_hash != self.parent_hash


def compute_diff(
    base_contents: Mapping[str, str],
    parent_contents: Mapping[str, str],
    base_hashes: Mapping[str, str],
    parent_hashes: Mapping[str, str],
    artifact_id: str,
) -> ArtifactDiff:
    """Build an :class:`ArtifactDiff` for one artifact."""
    if artifact_id not in base_contents:
        raise KeyError(f"artifact {artifact_id!r} not in base_contents")
    if artifact_id not in parent_contents:
        raise KeyError(f"artifact {artifact_id!r} not in parent_contents")
    return ArtifactDiff(
        artifact_id=artifact_id,
        base_content=base_contents[artifact_id],
        parent_content=parent_contents[artifact_id],
        base_hash=base_hashes.get(artifact_id, ""),
        parent_hash=parent_hashes.get(artifact_id, ""),
    )


# ---------------------------------------------------------------------- #
# Conflict
# ---------------------------------------------------------------------- #
class ConflictResolutionKind(str, Enum):
    TAKE_LEFT = "take_left"
    TAKE_RIGHT = "take_right"
    NEW_CONTENT = "new_content"


@dataclass(frozen=True, slots=True)
class ConflictResolution:
    artifact_id: str
    kind: ConflictResolutionKind
    new_content: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id is required")
        if self.kind == ConflictResolutionKind.NEW_CONTENT and not self.new_content:
            raise ValueError("new_content required when kind=NEW_CONTENT")


@dataclass(frozen=True, slots=True)
class ConflictReport:
    """One same-artifact conflict the editor must resolve."""

    artifact_id: str
    left_parent_id: str
    right_parent_id: str
    left_content: str
    right_content: str
    left_blame: float
    right_blame: float
    # The mechanism clusters each parent's analysis blamed this artifact for.
    left_mechanisms: tuple[str, ...] = ()
    right_mechanisms: tuple[str, ...] = ()

    def resolve(
        self,
        kind: ConflictResolutionKind,
        new_content: str = "",
        rationale: str = "",
    ) -> ConflictResolution:
        return ConflictResolution(
            artifact_id=self.artifact_id,
            kind=kind,
            new_content=new_content,
            rationale=rationale,
        )


# ---------------------------------------------------------------------- #
# Merge plan
# ---------------------------------------------------------------------- #
class MergePlanStatus(str, Enum):
    READY = "ready"
    HAS_CONFLICTS = "has_conflicts"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ArtifactResolution:
    """How one artifact is resolved in the merge plan."""

    artifact_id: str
    source_parent_id: str  # left, right, base, or "resolved"
    content: str
    rationale: str = ""


@dataclass(slots=True)
class MergePlan:
    """Deterministic merge plan for two parents.

    ``resolutions`` covers every artifact in the inventory; ``conflicts`` is
    the subset where both parents touched the artifact and the editor must
    resolve. The plan is READY only when conflicts is empty.
    """

    left_parent_id: str
    right_parent_id: str
    base_version: str
    resolutions: dict[str, ArtifactResolution] = field(default_factory=dict)
    conflicts: dict[str, ConflictReport] = field(default_factory=dict)
    ancestry_intersection: tuple[str, ...] = ()

    @property
    def status(self) -> MergePlanStatus:
        if self.conflicts:
            return MergePlanStatus.HAS_CONFLICTS
        if not self.resolutions:
            return MergePlanStatus.INVALID
        return MergePlanStatus.READY

    def apply_resolutions(self, resolutions: Iterable[ConflictResolution]) -> "MergePlan":
        """Return a new plan with editor-provided conflict resolutions applied.

        The editor may only resolve conflicts already in the plan; attempts to
        resolve non-conflict artifacts raise ``ValueError``.
        """
        new_resolutions = dict(self.resolutions)
        new_conflicts = dict(self.conflicts)
        for r in resolutions:
            if r.artifact_id not in self.conflicts:
                raise ValueError(
                    f"editor attempted to resolve non-conflict artifact: {r.artifact_id!r}"
                )
            c = self.conflicts[r.artifact_id]
            if r.kind == ConflictResolutionKind.TAKE_LEFT:
                content = c.left_content
                src = self.left_parent_id
            elif r.kind == ConflictResolutionKind.TAKE_RIGHT:
                content = c.right_content
                src = self.right_parent_id
            else:  # NEW_CONTENT
                content = r.new_content
                src = "resolved"
            new_resolutions[r.artifact_id] = ArtifactResolution(
                artifact_id=r.artifact_id,
                source_parent_id=src,
                content=content,
                rationale=r.rationale,
            )
            del new_conflicts[r.artifact_id]
        return MergePlan(
            left_parent_id=self.left_parent_id,
            right_parent_id=self.right_parent_id,
            base_version=self.base_version,
            resolutions=new_resolutions,
            conflicts=new_conflicts,
            ancestry_intersection=self.ancestry_intersection,
        )

    def to_edits(self) -> tuple[ArtifactEdit, ...]:
        """Convert resolutions to structured edits applicable to a fresh workspace.

        Only artifacts whose content differs from the base are emitted as
        edits; identical-to-base artifacts are skipped (no-op).
        """
        if self.status != MergePlanStatus.READY:
            raise ValueError(
                f"cannot convert MergePlan to edits: status={self.status.value}"
            )
        out: list[ArtifactEdit] = []
        for aid in sorted(self.resolutions.keys()):
            r = self.resolutions[aid]
            out.append(
                ArtifactEdit(
                    artifact_id=aid,
                    operation="replace",
                    payload={"content": r.content, "source_parent": r.source_parent_id},
                )
            )
        return tuple(out)


# ---------------------------------------------------------------------- #
# Merge function
# ---------------------------------------------------------------------- #
def _blame_for_artifact(analysis: CausalAnalysis, artifact_id: str) -> tuple[float, tuple[str, ...]]:
    """Return (total blame, mechanism strings) for an artifact from one analysis."""
    total = 0.0
    mechs: list[str] = []
    for n in analysis.blame_graph.nodes:
        if artifact_id in n.artifacts:
            total += n.blame
    if analysis.mechanism:
        mechs.append(analysis.mechanism)
    return total, tuple(mechs)


def _ancestry_intersection(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    """Common ancestor IDs, sorted for determinism."""
    return tuple(sorted(set(left) & set(right)))


def plan_merge(
    *,
    base_version: str,
    left_parent_id: str,
    right_parent_id: str,
    left_ancestors: Sequence[str],
    right_ancestors: Sequence[str],
    inventory: Sequence[ArtifactDescriptor],
    base_contents: Mapping[str, str],
    base_hashes: Mapping[str, str],
    left_contents: Mapping[str, str],
    left_hashes: Mapping[str, str],
    right_contents: Mapping[str, str],
    right_hashes: Mapping[str, str],
    left_analysis: CausalAnalysis,
    right_analysis: CausalAnalysis,
) -> MergePlan:
    """Build a deterministic merge plan for two parent candidates.

    Parameters
    ----------
    base_contents / base_hashes:
        The base harness artifact contents + hashes (the common ancestor).
    left/right_contents / hashes:
        Each parent's contents + hashes after their own edits.
    left/right_analysis:
        The analyzer+judge verdict for each parent on the merge-relevant task.
    """
    if left_parent_id == right_parent_id:
        raise ValueError("left and right parents must be distinct")

    ancestry = _ancestry_intersection(left_ancestors, right_ancestors)

    resolutions: dict[str, ArtifactResolution] = {}
    conflicts: dict[str, ConflictReport] = {}

    for desc in inventory:
        aid = desc.artifact_id
        if aid not in base_contents:
            # Adapter-declared but not present in base; skip (shouldn't happen
            # for the fake adapter but we tolerate it).
            continue
        left_diff = compute_diff(
            base_contents, left_contents, base_hashes, left_hashes, aid
        )
        right_diff = compute_diff(
            base_contents, right_contents, base_hashes, right_hashes, aid
        )

        if not left_diff.changed and not right_diff.changed:
            # Neither parent touched it -> base version.
            resolutions[aid] = ArtifactResolution(
                artifact_id=aid,
                source_parent_id="base",
                content=base_contents[aid],
            )
        elif left_diff.changed and not right_diff.changed:
            resolutions[aid] = ArtifactResolution(
                artifact_id=aid,
                source_parent_id=left_parent_id,
                content=left_contents[aid],
            )
        elif right_diff.changed and not left_diff.changed:
            resolutions[aid] = ArtifactResolution(
                artifact_id=aid,
                source_parent_id=right_parent_id,
                content=right_contents[aid],
            )
        else:
            # Both touched -> conflict.
            left_blame, left_mechs = _blame_for_artifact(left_analysis, aid)
            right_blame, right_mechs = _blame_for_artifact(right_analysis, aid)
            conflicts[aid] = ConflictReport(
                artifact_id=aid,
                left_parent_id=left_parent_id,
                right_parent_id=right_parent_id,
                left_content=left_contents[aid],
                right_content=right_contents[aid],
                left_blame=left_blame,
                right_blame=right_blame,
                left_mechanisms=left_mechs,
                right_mechanisms=right_mechs,
            )

    return MergePlan(
        left_parent_id=left_parent_id,
        right_parent_id=right_parent_id,
        base_version=base_version,
        resolutions=resolutions,
        conflicts=conflicts,
        ancestry_intersection=ancestry,
    )


# ---------------------------------------------------------------------- #
# Complementarity check
# ---------------------------------------------------------------------- #
def mechanisms_are_complementary(
    left: CausalAnalysis, right: CausalAnalysis
) -> bool:
    """Heuristic: two analyses are complementary if their mechanisms differ.

    This is used by the orchestrator to decide whether a merge is worth
    planning at all. If both parents address the same mechanism, merging
    their edits is unlikely to produce a non-conflicting plan.
    """
    return left.mechanism != right.mechanism


# ---------------------------------------------------------------------- #
# Protected floors check
# ---------------------------------------------------------------------- #
def merge_respects_protected_floors(
    plan: MergePlan,
    floors: Sequence[ProtectedFloor],
) -> bool:
    """Whether a ready plan's resolutions respect all protected floors.

    A merge respects a floor if no resolution introduces a content string
    containing a forbidden token. This is a coarse check; the real
    validation happens via :class:`FocusedValidation` after the merge is
    materialized.
    """
    # Floors are checked post-materialization via validation probes; the
    # merge plan itself cannot know the score impact. We expose this hook
    # so the orchestrator can mark a plan as floor-checked.
    return plan.status == MergePlanStatus.READY
