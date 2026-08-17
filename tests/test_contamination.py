"""Tests for the post-hoc GT-contamination detector.

This does not restrict the editor. It reports, after the fact, that an evolved
artifact contains a literal from the dataset's answer keys.

Every literal used here is synthetic. No real dataset answer key appears in this
file.
"""
from __future__ import annotations

from agent_evolve.core.contamination import (
    literals_from_regexes,
    scan_artifacts,
)


def test_extracts_literals_from_word_boundary_regexes() -> None:
    literals = literals_from_regexes(
        [r"(?i)\b17\b", r"(?i)\b0\.1777\b", r"(?i)\bClaude\ Shannon\b"]
    )

    assert "17" in literals
    assert "0.1777" in literals
    assert "Claude Shannon" in literals


def test_ignores_the_placeholder_regex() -> None:
    literals = literals_from_regexes([r"(?i)\?"])

    assert literals == ()


def test_flags_a_long_distinctive_literal_at_high_confidence() -> None:
    artifacts = {
        "instructions": (
            "When asked about the paper, answer "
            "Mapping Human Oriented Information to Software Agents."
        )
    }

    hits = scan_artifacts(
        artifacts, ["Mapping Human Oriented Information to Software Agents"]
    )

    assert len(hits) == 1
    assert hits[0].artifact_id == "instructions"
    assert hits[0].confidence == "high"


def test_flags_a_short_numeric_at_low_confidence() -> None:
    artifacts = {"instructions": "Take at most 17 steps before answering."}

    hits = scan_artifacts(artifacts, ["17"])

    assert len(hits) == 1
    assert hits[0].confidence == "low"


def test_clean_artifacts_produce_no_hits() -> None:
    artifacts = {"instructions": "Always verify before answering."}

    hits = scan_artifacts(artifacts, ["17", "0.1777"])

    assert hits == ()


def test_scans_every_artifact_and_reports_the_id() -> None:
    artifacts = {
        "instructions": "clean",
        "skills/generated-lookup": "the answer is 0.1777",
    }

    hits = scan_artifacts(artifacts, ["0.1777"])

    assert len(hits) == 1
    assert hits[0].artifact_id == "skills/generated-lookup"


def test_matching_is_case_insensitive() -> None:
    artifacts = {"instructions": "answer CLAUDE SHANNON"}

    hits = scan_artifacts(artifacts, ["Claude Shannon"])

    assert len(hits) == 1


def test_no_literals_is_a_no_op() -> None:
    assert scan_artifacts({"instructions": "anything"}, []) == ()
