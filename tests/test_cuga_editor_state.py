"""EditStagingArea: the editor's write boundary (spec §5, §9).

Every rule here is enforced in a tool body at staging time so the agent gets
per-artifact feedback while it works. Rejections are returned, never raised:
raising inside a CUGA tool body can abort the agent run.
"""
from __future__ import annotations

from agent_evolve.adapters.cuga_editor_state import EditStagingArea


def _area(**kwargs) -> EditStagingArea:
    defaults = dict(write_set=("skills/retrieval", "instructions"))
    defaults.update(kwargs)
    return EditStagingArea(**defaults)


# ------------------------------------------------------------------ #
# stage_replace authorization
# ------------------------------------------------------------------ #
def test_stage_replace_accepts_write_set_member() -> None:
    area = _area()
    outcome = area.stage_replace("skills/retrieval", "new body")
    assert outcome.accepted
    assert area.staged_ids() == ("skills/retrieval",)


def test_stage_replace_rejects_id_outside_write_set() -> None:
    area = _area()
    outcome = area.stage_replace("policies/execution", "x")
    assert not outcome.accepted
    assert "not in the authorized write set" in outcome.reason
    assert area.staged_ids() == ()


def test_stage_replace_returns_rejection_rather_than_raising() -> None:
    area = _area()
    # Must not raise: an exception inside a CUGA tool body can abort the run.
    outcome = area.stage_replace("nope/absent", "x")
    assert outcome.accepted is False


# ------------------------------------------------------------------ #
# stage_create namespace
# ------------------------------------------------------------------ #
def test_stage_create_accepts_namespaced_skill_id() -> None:
    area = _area()
    outcome = area.stage_create("skills/generated-recovery", "body")
    assert outcome.accepted
    assert area.created_count == 1


def test_stage_create_rejects_flat_generated_prefix() -> None:
    """A flat 'generated/' id would raise ValueError in _harness_slot.

    cuga_adapter._harness_slot accepts only 'instructions' or a
    skills|policies|memory/<name> prefix, so the CUGA group must come first.
    """
    area = _area()
    outcome = area.stage_create("generated/recovery", "body")
    assert not outcome.accepted
    assert "skills/generated-" in outcome.reason


def test_stage_create_rejects_policies_and_memory_namespaces() -> None:
    area = _area()
    for artifact_id in ("policies/generated-x", "memory/generated-x"):
        outcome = area.stage_create(artifact_id, "body")
        assert not outcome.accepted, artifact_id


def test_stage_create_rejects_existing_write_set_id() -> None:
    area = _area()
    outcome = area.stage_create("skills/retrieval", "body")
    assert not outcome.accepted
    assert "already exists" in outcome.reason


# ------------------------------------------------------------------ #
# caps
# ------------------------------------------------------------------ #
def test_stage_create_enforces_per_attempt_cap_of_two() -> None:
    area = _area()
    assert area.stage_create("skills/generated-a", "a").accepted
    assert area.stage_create("skills/generated-b", "b").accepted
    third = area.stage_create("skills/generated-c", "c")
    assert not third.accepted
    assert "per-attempt" in third.reason
    assert area.created_count == 2


def test_stage_create_enforces_pool_wide_cap() -> None:
    area = _area(pool_created_count=10, pool_create_cap=10)
    outcome = area.stage_create("skills/generated-a", "a")
    assert not outcome.accepted
    assert "pool" in outcome.reason


def test_pool_cap_counts_existing_plus_staged() -> None:
    area = _area(pool_created_count=9, pool_create_cap=10)
    assert area.stage_create("skills/generated-a", "a").accepted
    second = area.stage_create("skills/generated-b", "b")
    assert not second.accepted


# ------------------------------------------------------------------ #
# unstage / edits
# ------------------------------------------------------------------ #
def test_unstage_removes_a_staged_edit() -> None:
    area = _area()
    area.stage_replace("skills/retrieval", "x")
    assert area.unstage("skills/retrieval").accepted
    assert area.staged_ids() == ()


def test_unstage_rejects_unknown_id() -> None:
    area = _area()
    assert not area.unstage("skills/retrieval").accepted


def test_unstage_frees_a_create_slot() -> None:
    area = _area()
    area.stage_create("skills/generated-a", "a")
    area.stage_create("skills/generated-b", "b")
    area.unstage("skills/generated-b")
    assert area.stage_create("skills/generated-c", "c").accepted


def test_edits_carry_the_correct_operation() -> None:
    area = _area()
    area.stage_replace("skills/retrieval", "r")
    area.stage_create("skills/generated-a", "a")
    ops = {e.artifact_id: e.operation for e in area.edits()}
    assert ops == {
        "skills/retrieval": "replace",
        "skills/generated-a": "create",
    }


def test_edits_are_sorted_for_determinism() -> None:
    area = _area()
    area.stage_replace("instructions", "i")
    area.stage_replace("skills/retrieval", "r")
    assert [e.artifact_id for e in area.edits()] == [
        "instructions",
        "skills/retrieval",
    ]


def test_restaging_the_same_id_replaces_the_content() -> None:
    area = _area()
    area.stage_replace("skills/retrieval", "first")
    area.stage_replace("skills/retrieval", "second")
    assert area.staged_ids() == ("skills/retrieval",)
    assert area.edits()[0].payload["content"] == "second"


# ------------------------------------------------------------------ #
# authored-content normalization
# ------------------------------------------------------------------ #
# An agent that writes artifact bodies from inside an indented Python string
# literal carries that indentation into the artifact. Observed live: every line
# after the first arrived prefixed with four spaces, which Markdown renders as
# a code block, so the skill body silently degrades into a literal listing.
def test_staged_content_is_dedented_when_uniformly_indented() -> None:
    area = _area()
    body = "# Title\n    \n    Step one.\n    Step two.\n"
    area.stage_create("skills/generated-a", body)
    content = area.edits()[0].payload["content"]
    assert "\n    Step one." not in content
    assert "Step one." in content and "Step two." in content


def test_staged_replacement_content_is_dedented_too() -> None:
    area = _area()
    area.stage_replace("skills/retrieval", "# T\n    body line\n")
    assert area.edits()[0].payload["content"] == "# T\nbody line"


def test_dedent_preserves_relative_markdown_indentation() -> None:
    """Nested list structure is meaning, not accidental indentation."""
    area = _area()
    body = "# Title\n\n1. Outer step.\n   - nested detail\n2. Second step.\n"
    area.stage_create("skills/generated-a", body)
    assert "   - nested detail" in area.edits()[0].payload["content"]


def test_dedent_leaves_already_flush_content_unchanged() -> None:
    area = _area()
    body = "# Title\n\n1. One.\n2. Two.\n"
    area.stage_replace("skills/retrieval", body)
    assert area.edits()[0].payload["content"] == body.strip()


def test_dedent_preserves_indented_fenced_code_block_contents() -> None:
    """A fenced block's own indentation must survive relative to the fence."""
    area = _area()
    body = "# T\n    \n    ```python\n    x = 1\n    if x:\n        pass\n    ```\n"
    area.stage_create("skills/generated-a", body)
    content = area.edits()[0].payload["content"]
    assert "```python\nx = 1" in content
    assert "    pass" in content


# ------------------------------------------------------------------ #
# parent read ledger (provenance, spec §9)
# ------------------------------------------------------------------ #
def test_parents_read_is_empty_before_any_read() -> None:
    assert _area().parents_read() == ()


def test_parents_read_records_reads_deduplicated_and_sorted() -> None:
    area = _area()
    area.record_parent_read("cand-b")
    area.record_parent_read("cand-a")
    area.record_parent_read("cand-b")
    assert area.parents_read() == ("cand-a", "cand-b")
