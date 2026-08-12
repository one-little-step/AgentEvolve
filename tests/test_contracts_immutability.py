"""Standard Pydantic behavior for persisted contract models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_evolve.core.contracts import ScoreCell


def score_cell_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "candidate_id": "candidate-1",
        "task_id": "task-1",
        "mechanism_cluster_id": "cluster-1",
        "score": 0.75,
        "severity": 0.8,
        "confidence": 0.9,
        "stability": 0.5,
        "rollout_count": 2,
        "rollout_ids": ("rollout-1", "rollout-2"),
        "verdict_refs": ("verdict-1",),
        "artifact_versions": {"artifact-1": "sha256:abcdef"},
        "evaluator_id": "judge-v1",
        "coverage": "evaluated",
    }
    values.update(changes)
    return values


def test_standard_constructor_and_model_validate_are_supported() -> None:
    values = score_cell_values()

    assert ScoreCell(**values).candidate_id == "candidate-1"
    assert ScoreCell.model_validate(values).candidate_id == "candidate-1"


def test_normal_pydantic_validation_apis_and_coercion_are_supported() -> None:
    values = score_cell_values(score="0.75")

    assert ScoreCell.model_validate_json(ScoreCell(**values).model_dump_json()).score == 0.75
    with pytest.raises(ValidationError):
        ScoreCell.model_validate_strings(values)


def test_frozen_model_blocks_attribute_assignment_but_nested_mapping_is_mutable() -> None:
    cell = ScoreCell(**score_cell_values())

    with pytest.raises(ValidationError):
        cell.score = 0.0

    assert isinstance(cell.artifact_versions, dict)
    cell.artifact_versions["artifact-2"] = "sha256:abcdef"
    assert cell.artifact_versions["artifact-2"] == "sha256:abcdef"


def test_content_hash_validation_uses_standard_validation_error() -> None:
    with pytest.raises(ValidationError):
        ScoreCell(**score_cell_values(artifact_versions={"artifact-1": "not-a-hash"}))
