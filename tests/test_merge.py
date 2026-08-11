"""Tests for provenance-preserving deterministic merge."""
from __future__ import annotations

import pytest

from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis
from agent_evolve.core.contracts import ArtifactDescriptor
from agent_evolve.core.merge import (
    ArtifactDiff,
    ArtifactResolution,
    ConflictReport,
    ConflictResolution,
    ConflictResolutionKind,
    MergePlan,
    MergePlanStatus,
    compute_diff,
    mechanisms_are_complementary,
    merge_respects_protected_floors,
    plan_merge,
)


def _desc(aid: str) -> ArtifactDescriptor:
    return ArtifactDescriptor(
        artifact_id=aid,
        kind="skill",
        format="text/plain",
        version_hash=f"sha256:{aid}",
        readable=True,
        writable=True,
        merge_strategy="replace-overwrites",
    )


def _analysis(mechanism: str, blame_map: dict[str, float] | None = None) -> CausalAnalysis:
    nodes = tuple(
        BlameNode(actor_id=aid, blame=b, artifacts=(aid,))
        for aid, b in (blame_map or {}).items()
    )
    return CausalAnalysis(
        mechanism=mechanism,
        severity=0.5,
        score=0.5,
        blame_graph=BlameGraph(nodes=nodes),
    )


def _contents(mapping: dict[str, str]) -> dict[str, str]:
    return dict(mapping)


def _hashes(mapping: dict[str, str]) -> dict[str, str]:
    return {k: f"h-{v}" for k, v in mapping.items()}


def _baseline_inventory() -> tuple[ArtifactDescriptor, ...]:
    return (_desc("skills/a"), _desc("skills/b"), _desc("skills/c"))


def _baseline_contents() -> dict[str, str]:
    return {"skills/a": "A-base", "skills/b": "B-base", "skills/c": "C-base"}


# ---------------------------------------------------------------------- #
# compute_diff
# ---------------------------------------------------------------------- #
def test_compute_diff_detects_change():
    d = compute_diff(
        base_contents={"a": "x"}, parent_contents={"a": "y"},
        base_hashes={"a": "hx"}, parent_hashes={"a": "hy"},
        artifact_id="a",
    )
    assert d.changed is True
    assert d.parent_content == "y"


def test_compute_diff_detects_no_change():
    d = compute_diff(
        base_contents={"a": "x"}, parent_contents={"a": "x"},
        base_hashes={"a": "hx"}, parent_hashes={"a": "hx"},
        artifact_id="a",
    )
    assert d.changed is False


def test_compute_diff_raises_on_missing_artifact():
    with pytest.raises(KeyError):
        compute_diff(
            base_contents={"a": "x"}, parent_contents={"a": "y"},
            base_hashes={}, parent_hashes={},
            artifact_id="missing",
        )


# ---------------------------------------------------------------------- #
# ConflictResolution
# ---------------------------------------------------------------------- #
def test_conflict_resolution_new_content_requires_content():
    with pytest.raises(ValueError):
        ConflictResolution(
            artifact_id="a", kind=ConflictResolutionKind.NEW_CONTENT, new_content=""
        )


def test_conflict_resolution_take_left_allows_empty_content():
    r = ConflictResolution(artifact_id="a", kind=ConflictResolutionKind.TAKE_LEFT)
    assert r.new_content == ""


# ---------------------------------------------------------------------- #
# plan_merge — non-conflicting
# ---------------------------------------------------------------------- #
def test_plan_merge_no_conflicts_takes_changed_parent():
    base = _baseline_contents()
    base_hashes = _hashes(base)
    left = _contents({"skills/a": "A-left", "skills/b": "B-base", "skills/c": "C-base"})
    right = _contents({"skills/a": "A-base", "skills/b": "B-right", "skills/c": "C-base"})

    plan = plan_merge(
        base_version="base-v0",
        left_parent_id="left", right_parent_id="right",
        left_ancestors=("base-v0",), right_ancestors=("base-v0",),
        inventory=_baseline_inventory(),
        base_contents=base, base_hashes=base_hashes,
        left_contents=left, left_hashes=_hashes(left),
        right_contents=right, right_hashes=_hashes(right),
        left_analysis=_analysis("m-left"),
        right_analysis=_analysis("m-right"),
    )
    assert plan.status == MergePlanStatus.READY
    assert plan.resolutions["skills/a"].source_parent_id == "left"
    assert plan.resolutions["skills/a"].content == "A-left"
    assert plan.resolutions["skills/b"].source_parent_id == "right"
    assert plan.resolutions["skills/b"].content == "B-right"
    assert plan.resolutions["skills/c"].source_parent_id == "base"
    assert plan.resolutions["skills/c"].content == "C-base"
    assert plan.conflicts == {}


def test_plan_merge_status_invalid_when_no_resolutions():
    plan = MergePlan(
        left_parent_id="l", right_parent_id="r", base_version="b",
    )
    assert plan.status == MergePlanStatus.INVALID


# ---------------------------------------------------------------------- #
# plan_merge — conflicts
# ---------------------------------------------------------------------- #
def test_plan_merge_records_conflict_when_both_parents_changed_same_artifact():
    base = _baseline_contents()
    base_hashes = _hashes(base)
    left = _contents({"skills/a": "A-left", "skills/b": "B-base", "skills/c": "C-base"})
    right = _contents({"skills/a": "A-right", "skills/b": "B-base", "skills/c": "C-base"})

    plan = plan_merge(
        base_version="base-v0",
        left_parent_id="left", right_parent_id="right",
        left_ancestors=("base-v0",), right_ancestors=("base-v0",),
        inventory=_baseline_inventory(),
        base_contents=base, base_hashes=base_hashes,
        left_contents=left, left_hashes=_hashes(left),
        right_contents=right, right_hashes=_hashes(right),
        left_analysis=_analysis("m-left", blame_map={"skills/a": 0.6}),
        right_analysis=_analysis("m-right", blame_map={"skills/a": 0.4}),
    )
    assert plan.status == MergePlanStatus.HAS_CONFLICTS
    assert "skills/a" in plan.conflicts
    c = plan.conflicts["skills/a"]
    assert c.left_content == "A-left"
    assert c.right_content == "A-right"
    assert c.left_blame == 0.6
    assert c.right_blame == 0.4
    assert c.left_mechanisms == ("m-left",)
    assert c.right_mechanisms == ("m-right",)


def test_plan_merge_rejects_same_parent_id():
    with pytest.raises(ValueError):
        plan_merge(
            base_version="base-v0",
            left_parent_id="same", right_parent_id="same",
            left_ancestors=(), right_ancestors=(),
            inventory=(),
            base_contents={}, base_hashes={},
            left_contents={}, left_hashes={},
            right_contents={}, right_hashes={},
            left_analysis=_analysis("m"),
            right_analysis=_analysis("m"),
        )


def test_plan_merge_ancestry_intersection_sorted():
    base = _baseline_contents()
    plan = plan_merge(
        base_version="base-v0",
        left_parent_id="l", right_parent_id="r",
        left_ancestors=("z", "a", "m"), right_ancestors=("m", "a", "b"),
        inventory=_baseline_inventory(),
        base_contents=base, base_hashes=_hashes(base),
        left_contents=base, left_hashes=_hashes(base),
        right_contents=base, right_hashes=_hashes(base),
        left_analysis=_analysis("m"),
        right_analysis=_analysis("m"),
    )
    assert plan.ancestry_intersection == ("a", "m")


# ---------------------------------------------------------------------- #
# apply_resolutions
# ---------------------------------------------------------------------- #
def test_apply_resolutions_take_left():
    base = _baseline_contents()
    left = _contents({"skills/a": "A-left", "skills/b": "B-base", "skills/c": "C-base"})
    right = _contents({"skills/a": "A-right", "skills/b": "B-base", "skills/c": "C-base"})
    plan = plan_merge(
        base_version="base-v0",
        left_parent_id="left", right_parent_id="right",
        left_ancestors=("base-v0",), right_ancestors=("base-v0",),
        inventory=_baseline_inventory(),
        base_contents=base, base_hashes=_hashes(base),
        left_contents=left, left_hashes=_hashes(left),
        right_contents=right, right_hashes=_hashes(right),
        left_analysis=_analysis("m-left"),
        right_analysis=_analysis("m-right"),
    )
    assert plan.status == MergePlanStatus.HAS_CONFLICTS

    resolved = plan.apply_resolutions([
        ConflictResolution(
            artifact_id="skills/a",
            kind=ConflictResolutionKind.TAKE_LEFT,
            rationale="left has higher blame",
        )
    ])
    assert resolved.status == MergePlanStatus.READY
    assert resolved.resolutions["skills/a"].content == "A-left"
    assert resolved.resolutions["skills/a"].source_parent_id == "left"


def test_apply_resolutions_new_content():
    base = _baseline_contents()
    left = _contents({"skills/a": "A-left", "skills/b": "B-base", "skills/c": "C-base"})
    right = _contents({"skills/a": "A-right", "skills/b": "B-base", "skills/c": "C-base"})
    plan = plan_merge(
        base_version="base-v0",
        left_parent_id="left", right_parent_id="right",
        left_ancestors=("base-v0",), right_ancestors=("base-v0",),
        inventory=_baseline_inventory(),
        base_contents=base, base_hashes=_hashes(base),
        left_contents=left, left_hashes=_hashes(left),
        right_contents=right, right_hashes=_hashes(right),
        left_analysis=_analysis("m-left"),
        right_analysis=_analysis("m-right"),
    )
    resolved = plan.apply_resolutions([
        ConflictResolution(
            artifact_id="skills/a",
            kind=ConflictResolutionKind.NEW_CONTENT,
            new_content="A-merged",
            rationale="combine both fixes",
        )
    ])
    assert resolved.status == MergePlanStatus.READY
    assert resolved.resolutions["skills/a"].content == "A-merged"
    assert resolved.resolutions["skills/a"].source_parent_id == "resolved"


def test_apply_resolutions_rejects_non_conflict_artifact():
    plan = MergePlan(
        left_parent_id="l", right_parent_id="r", base_version="b",
        resolutions={"x": ArtifactResolution(artifact_id="x", source_parent_id="l", content="x")},
    )
    with pytest.raises(ValueError):
        plan.apply_resolutions([
            ConflictResolution(artifact_id="x", kind=ConflictResolutionKind.TAKE_LEFT)
        ])


# ---------------------------------------------------------------------- #
# to_edits
# ---------------------------------------------------------------------- #
def test_to_edits_rejects_plan_with_conflicts():
    plan = MergePlan(
        left_parent_id="l", right_parent_id="r", base_version="b",
        conflicts={"x": ConflictReport(
            artifact_id="x", left_parent_id="l", right_parent_id="r",
            left_content="lx", right_content="rx", left_blame=0.5, right_blame=0.5,
        )},
    )
    with pytest.raises(ValueError):
        plan.to_edits()


def test_to_edits_returns_one_edit_per_resolution():
    base = _baseline_contents()
    left = _contents({"skills/a": "A-left", "skills/b": "B-base", "skills/c": "C-base"})
    right = _contents({"skills/a": "A-base", "skills/b": "B-right", "skills/c": "C-base"})
    plan = plan_merge(
        base_version="base-v0",
        left_parent_id="left", right_parent_id="right",
        left_ancestors=("base-v0",), right_ancestors=("base-v0",),
        inventory=_baseline_inventory(),
        base_contents=base, base_hashes=_hashes(base),
        left_contents=left, left_hashes=_hashes(left),
        right_contents=right, right_hashes=_hashes(right),
        left_analysis=_analysis("m-left"),
        right_analysis=_analysis("m-right"),
    )
    edits = plan.to_edits()
    assert len(edits) == 3
    artifact_ids = {e.artifact_id for e in edits}
    assert artifact_ids == {"skills/a", "skills/b", "skills/c"}


# ---------------------------------------------------------------------- #
# Complementarity + floors
# ---------------------------------------------------------------------- #
def test_mechanisms_complementary_true_when_different():
    a = _analysis("m1")
    b = _analysis("m2")
    assert mechanisms_are_complementary(a, b) is True


def test_mechanisms_complementary_false_when_same():
    a = _analysis("m1")
    b = _analysis("m1")
    assert mechanisms_are_complementary(a, b) is False


def test_merge_respects_protected_floors_true_when_ready():
    base = _baseline_contents()
    left = _contents({"skills/a": "A-left", "skills/b": "B-base", "skills/c": "C-base"})
    right = _contents({"skills/a": "A-base", "skills/b": "B-right", "skills/c": "C-base"})
    plan = plan_merge(
        base_version="base-v0",
        left_parent_id="left", right_parent_id="right",
        left_ancestors=("base-v0",), right_ancestors=("base-v0",),
        inventory=_baseline_inventory(),
        base_contents=base, base_hashes=_hashes(base),
        left_contents=left, left_hashes=_hashes(left),
        right_contents=right, right_hashes=_hashes(right),
        left_analysis=_analysis("m-left"),
        right_analysis=_analysis("m-right"),
    )
    assert merge_respects_protected_floors(plan, floors=()) is True


def test_merge_respects_protected_floors_false_when_conflicts():
    plan = MergePlan(
        left_parent_id="l", right_parent_id="r", base_version="b",
        conflicts={"x": ConflictReport(
            artifact_id="x", left_parent_id="l", right_parent_id="r",
            left_content="lx", right_content="rx", left_blame=0.5, right_blame=0.5,
        )},
    )
    assert merge_respects_protected_floors(plan, floors=()) is False
