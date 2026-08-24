"""IDX2: the signed mechanism index (D5.4).

Maps a mechanism cluster to its members, RANKED for complementary-parenthood
lookup:

1. **Solvers first** (``valence=-1``), strongest by severity descending.
2. **Then faults** (``valence=+1``), least-bad first -- severity ascending.

The index is built from the TS2 cross-attempt trace store by
``SequentialGepaRunner.signed_mechanism_index``; this module holds only the
pure structure and its ranking contract, so it stays unit-testable without a
runner. Clustering itself is NOT done here: entries arrive already carrying
their full namespaced cluster id (``"<task_id>:c<N>"``), assigned through the
existing ``MechanismClusterer`` whose ``assign`` accepts both fault analyses
and strength findings -- the shared namespace D5.1 requires.

Honesty rules mirrored from the rest of the pipeline: an entry exists only
when a real, scorable, diagnosed observation produced it; nothing is invented
to fill a cluster.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One ranked member of a mechanism cluster."""

    valence: int  # -1 solver (strength), +1 fault
    severity: float  # magnitude in [0, 1], direction lives in valence
    candidate_id: str
    task_id: str
    #: Full namespaced id ("task-a:c3") matching the entropy convention.
    cluster_id: str
    #: Artifacts the diagnosis/strength attributes -- what the editor tool
    #: will offer to read via ``read_parent_artifact``.
    artifact_ids: tuple[str, ...]
    trace_id: str

    def __post_init__(self) -> None:
        if self.valence not in (1, -1):
            raise ValueError(f"valence must be +1 or -1, got {self.valence!r}")
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if not self.cluster_id:
            raise ValueError("cluster_id is required")


@dataclass(slots=True)
class SignedMechanismIndex:
    """cluster key -> ranked members."""

    _members: dict[tuple[str, str], list[IndexEntry]]

    def __init__(self) -> None:
        self._members = {}

    # ------------------------------------------------------------------ #
    def add(self, entry: IndexEntry) -> None:
        self._members.setdefault((entry.task_id, entry.cluster_id), []).append(entry)

    def members_for(
        self, task_id: str, cluster_id: str, *, limit: int | None = None
    ) -> tuple[IndexEntry, ...]:
        """Ranked members of one cluster.

        Solvers (``-1``) first by severity DESCENDING; then faults (``+1``)
        by severity ASCENDING -- "least-bad failures" last. Empty when the
        cluster does not exist; that is an honest absence, not an error.
        """
        members = self._members.get((task_id, cluster_id), [])
        solvers = sorted(
            (m for m in members if m.valence == -1),
            key=lambda m: -(m.severity or 0.0),
        )
        faults = sorted(
            (m for m in members if m.valence == 1),
            key=lambda m: (m.severity or 0.0),
        )
        ranked = (*solvers, *faults)
        return ranked[:limit] if limit is not None else ranked

    def clusters(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._members.keys())

    def __len__(self) -> int:
        return len(self._members)


__all__ = ["IndexEntry", "SignedMechanismIndex"]
