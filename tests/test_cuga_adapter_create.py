"""Adapter support for artifact creation (spec §5).

Creation must map onto a real CUGA harness slot. A flat 'generated/<name>' id
would raise ValueError in _harness_slot, so the CUGA group comes first:
'skills/generated-<name>'.
"""
from __future__ import annotations

import pytest

from agent_evolve.adapters.cuga_adapter import CugaAdapter
from agent_evolve.core.contracts import ArtifactEdit
from agent_evolve.cuga_wrapper import CugaWrapper, InMemoryRuntime, RuntimeSettings


def _adapter() -> CugaAdapter:
    adapter = CugaAdapter(
        wrapper=CugaWrapper(InMemoryRuntime(), RuntimeSettings(model="test-model"))
    )
    adapter.register_candidate("base-v0", {"skills/retrieval": "body"})
    return adapter


def test_create_adds_a_new_artifact() -> None:
    adapter = _adapter()
    ws = adapter.materialize_candidate("base-v0", "att-1")
    result = adapter.apply_structured_edits(
        ws,
        (
            ArtifactEdit(
                artifact_id="skills/generated-recovery",
                operation="create",
                payload={"content": "new skill body"},
            ),
        ),
    )
    assert result["skills/generated-recovery"] == "new skill body"


def test_created_artifact_appears_in_inventory() -> None:
    adapter = _adapter()
    ws = adapter.materialize_candidate("base-v0", "att-1")
    adapter.apply_structured_edits(
        ws,
        (
            ArtifactEdit(
                artifact_id="skills/generated-recovery",
                operation="create",
                payload={"content": "b"},
            ),
        ),
    )
    ids = [d.artifact_id for d in adapter.artifact_inventory(ws.version)]
    assert "skills/generated-recovery" in ids


def test_created_artifact_reaches_the_harness_config() -> None:
    """A created skill must actually be delivered to CUGA, not merely stored."""
    adapter = _adapter()
    ws = adapter.materialize_candidate("base-v0", "att-1")
    adapter.apply_structured_edits(
        ws,
        (
            ArtifactEdit(
                artifact_id="skills/generated-recovery",
                operation="create",
                payload={"content": "b"},
            ),
        ),
    )
    from agent_evolve.core.contracts import EvolutionTask

    config = adapter._harness_config(
        ws.version, EvolutionTask(task_id="t", input_text="i")
    )
    assert config["skills"]["generated-recovery"] == "b"


def test_create_rejects_an_unmappable_id() -> None:
    adapter = _adapter()
    ws = adapter.materialize_candidate("base-v0", "att-1")
    with pytest.raises(ValueError, match="does not map to a CUGA harness slot"):
        adapter.apply_structured_edits(
            ws,
            (
                ArtifactEdit(
                    artifact_id="generated/recovery",
                    operation="create",
                    payload={"content": "b"},
                ),
            ),
        )


def test_create_rejects_an_existing_id() -> None:
    adapter = _adapter()
    ws = adapter.materialize_candidate("base-v0", "att-1")
    with pytest.raises(ValueError, match="already exists"):
        adapter.apply_structured_edits(
            ws,
            (
                ArtifactEdit(
                    artifact_id="skills/retrieval",
                    operation="create",
                    payload={"content": "b"},
                ),
            ),
        )


def test_replace_still_rejects_an_absent_id() -> None:
    adapter = _adapter()
    ws = adapter.materialize_candidate("base-v0", "att-1")
    with pytest.raises(KeyError):
        adapter.apply_structured_edits(
            ws,
            (
                ArtifactEdit(
                    artifact_id="skills/absent",
                    operation="replace",
                    payload={"content": "b"},
                ),
            ),
        )


def test_created_artifact_count_tracks_generated_prefix() -> None:
    adapter = _adapter()
    ws = adapter.materialize_candidate("base-v0", "att-1")
    assert adapter.created_artifact_count(ws.version) == 0
    adapter.apply_structured_edits(
        ws,
        (
            ArtifactEdit(
                artifact_id="skills/generated-a",
                operation="create",
                payload={"content": "a"},
            ),
        ),
    )
    assert adapter.created_artifact_count(ws.version) == 1


def test_creatable_prefix_is_declared() -> None:
    assert CugaAdapter.creatable_prefix == "skills/generated-"
