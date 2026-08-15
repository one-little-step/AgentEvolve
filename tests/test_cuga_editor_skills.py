"""Editor agent instructions and skills (spec §6)."""
from __future__ import annotations

from agent_evolve.adapters.cuga_editor_skills import (
    EDITOR_INSTRUCTIONS,
    EDITOR_SKILLS,
    build_editor_prompt,
)


def test_four_skills_are_defined() -> None:
    assert set(EDITOR_SKILLS) == {
        "refine-artifact",
        "combine-parents",
        "create-artifact",
        "learn-from-history",
    }


def test_instructions_state_the_authorization_invariant() -> None:
    assert "authorized" in EDITOR_INSTRUCTIONS.lower()


def test_instructions_require_finalizing_even_when_declining() -> None:
    lowered = EDITOR_INSTRUCTIONS.lower()
    assert "submit_edit_plan" in lowered
    assert "declin" in lowered


def test_instructions_nudge_both_refine_and_combine() -> None:
    lowered = EDITOR_INSTRUCTIONS.lower()
    assert "refine" in lowered
    assert "combine" in lowered


def test_instructions_never_mention_the_expected_contract() -> None:
    """The editor must not be told an expected answer exists to look for."""
    lowered = EDITOR_INSTRUCTIONS.lower()
    for banned in ("expected_contract", "expected answer", "expected_substring"):
        assert banned not in lowered


def test_no_skill_mentions_the_expected_contract() -> None:
    for name, body in EDITOR_SKILLS.items():
        lowered = body.lower()
        for banned in ("expected_contract", "expected answer", "expected_substring"):
            assert banned not in lowered, f"{name} leaks {banned}"


def test_every_skill_is_non_trivial() -> None:
    for name, body in EDITOR_SKILLS.items():
        assert len(body.strip()) > 200, f"{name} is too thin to guide behavior"


def test_prompt_embeds_the_evidence_summary() -> None:
    prompt = build_editor_prompt("MECHANISM: skill never loaded")
    assert "MECHANISM: skill never loaded" in prompt


def test_prompt_directs_the_agent_to_finalize() -> None:
    assert "submit_edit_plan" in build_editor_prompt("x")


def test_prompt_uses_explicit_code_execution_phrasing() -> None:
    """Vague 'use the tools' wording measured 0/2 tool execution on this model;
    an explicit write-and-execute instruction measured 2/2. Without this, the
    agent never reaches the sandbox and every attempt is a no_tool_call."""
    prompt = build_editor_prompt("x").lower()
    assert "write and execute" in prompt
    assert "python code" in prompt


def test_instructions_state_the_one_code_block_per_turn_contract() -> None:
    """CUGA executes only the first fenced block in a response.

    Observed live: the model emitted 8 blocks in one turn, only get_mechanism
    ran, and it then concluded the tools were broken and refused to finalize.
    """
    # Collapse whitespace: the contract is prose that wraps across lines.
    flat = " ".join(EDITOR_INSTRUCTIONS.lower().split())
    assert "one fenced python block per turn" in flat
    assert "executes only the first" in flat
