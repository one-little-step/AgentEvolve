"""Editor agent instructions and skills (spec §6)."""
from __future__ import annotations

from pathlib import Path

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


def test_prompt_demands_code_on_the_very_first_turn() -> None:
    """Narrating a plan before executing it produced a 0-tool-call run.

    The model announced seven investigation steps in prose, emitted no fenced
    block, never reached the sandbox, and was routed straight to a final
    answer. The prompt must make the first concrete action a code block.
    """
    flat = " ".join(build_editor_prompt("EVIDENCE").lower().split())
    assert "very next message" in flat
    assert "narration without a fenced block" in flat


def test_editor_skills_materialize_with_loadable_frontmatter(tmp_path) -> None:
    """Every editor skill must reach disk with frontmatter CUGA can load.

    Two silent failures this pins, both observed:
      1. The skills were never materialized at all, so a live editor run
         loaded a stale global web-research skill and none of its own.
      2. A body starting with '# Heading' yielded `description: None`, because
         an unquoted '#' opens a YAML comment. CUGA's loader rejects a skill
         with no description, so the file existed but never reached the model.
    """
    import yaml

    from agent_evolve.adapters.cuga_editor import materialize_editor_skills

    root = Path(materialize_editor_skills(tmp_path)) / "skills"
    found = {p.parent.name for p in root.rglob("SKILL.md")}
    assert found == set(EDITOR_SKILLS)

    for path in root.rglob("SKILL.md"):
        text = path.read_text(encoding="utf-8")
        meta = yaml.safe_load(text.split("---")[1])
        assert meta.get("name"), f"{path} has no name"
        assert meta.get("description"), f"{path} has no description"
        assert text.split("---")[2].strip(), f"{path} has an empty body"


def test_editor_agent_kwargs_bind_the_editor_skills_folder(tmp_path) -> None:
    """cuga_folder must never be None: CUGA then resolves its skill root to
    <cwd>/.cuga and picks up whatever a previous run left there."""
    from agent_evolve.adapters.cuga_editor import editor_agent_kwargs

    kwargs = editor_agent_kwargs(str(tmp_path))
    assert kwargs["cuga_folder"] == str(tmp_path)
    assert kwargs["skills_folder"] == str(tmp_path)
    assert kwargs["enable_skills"] is True


# ---------------------------------------------------------------------- #
# Surface literacy
#
# All four editable surfaces reach the rollout model by different mechanics,
# and choosing the wrong one produces an edit that validates, materializes,
# and changes nothing. That is a silent failure: no test outside this block
# would notice if the surface guidance were dropped, so these pin it.
# ---------------------------------------------------------------------- #
_SURFACE_DOC = "\n".join([EDITOR_INSTRUCTIONS, *EDITOR_SKILLS.values()]).lower()


def test_all_four_editable_surfaces_are_described() -> None:
    for surface in ("instructions", "skill", "polic", "memory"):
        assert surface in _SURFACE_DOC, f"surface {surface} never described"


def test_instructions_surface_is_described_as_unconditional() -> None:
    """`instructions` is delivered on every turn with no opt-in by the model.

    Verified this session with an unguessable marker: the artifact's text
    reached the rollout model's context unconditionally. It is therefore the
    only surface that can govern turn-level behavior, and the editor must know
    it is editable at all -- it was previously read-only.
    """
    flat = " ".join(_SURFACE_DOC.split())
    assert "unconditional" in flat
    assert "every turn" in flat


def test_the_skill_opt_in_trap_is_stated() -> None:
    """A skill body is only read if the model chooses to call load_skill.

    So a skill cannot repair a failure that prevents tool invocation in the
    first place; that is circular. An editor that does not know this authors a
    plausible skill for a mechanism the skill can never reach.
    """
    flat = " ".join(_SURFACE_DOC.split())
    assert "load_skill" in flat
    assert "opt in" in flat or "opt into" in flat or "choose to" in flat


def test_the_skill_description_selection_criterion_is_stated() -> None:
    """The description is what the model selects on, not decoration.

    Trigger-oriented descriptions ("Use when ...") were selected; passive
    titles were not, so the skill existed on disk and was never loaded.
    """
    flat = " ".join(_SURFACE_DOC.split())
    assert "first line" in flat
    assert "description" in flat


def test_the_always_true_policy_trap_is_stated() -> None:
    """A policy whose only trigger is `always: true` loads and never matches.

    Verified against cuga 0.3.1: no evaluator selects an AlwaysTrigger. A
    policy needs a real intent trigger or it is inert.
    """
    flat = " ".join(_SURFACE_DOC.split())
    assert "trigger" in flat
    assert "never matches" in flat or "never match" in flat


def test_memory_is_described_as_facts_not_behavior() -> None:
    flat = " ".join(_SURFACE_DOC.split())
    assert "memory" in flat
    assert "not behavior" in flat or "not for behavior" in flat


def test_the_directness_heuristic_is_stated() -> None:
    """Prefer the surface with the most direct unconditional path to the model.

    Without this the editor picks by familiarity and produces valid-looking
    edits with zero measurable effect.
    """
    flat = " ".join(_SURFACE_DOC.split())
    assert "most direct" in flat


def test_the_instructions_surface_is_stated_to_be_editable() -> None:
    """`instructions` became editable this session; before that the editor could
    only ever write skills. A prompt that does not say so leaves the strongest
    lever unused, and nothing else in the suite would catch the omission."""
    flat = " ".join(EDITOR_INSTRUCTIONS.lower().split())
    assert "instructions" in flat
    assert "may edit" in flat or "editable" in flat
    assert "highest leverage" in flat or "highest-leverage" in flat


def test_the_create_skill_warns_that_creation_is_confined_to_one_surface() -> None:
    """The creatable prefix is a skills prefix, so a created artifact inherits
    the opt-in delivery route. An editor that reaches for creation to fix a
    turn-level mechanism produces an artifact that can never apply."""
    flat = " ".join(EDITOR_SKILLS["create-artifact"].lower().split())
    assert "surface" in flat
    assert "load_skill" in flat or "opt in" in flat or "selected" in flat


def test_the_refine_skill_requires_choosing_a_surface_deliberately() -> None:
    flat = " ".join(EDITOR_SKILLS["refine-artifact"].lower().split())
    assert "surface" in flat


def test_surface_guidance_prescribes_no_specific_remedy() -> None:
    """The prompts must teach reasoning, not carry the answer.

    If the fix for the current measured bug were written here, the editor would
    be scored on a recipe rather than on its ability to derive one, and the
    result would not generalize to the next mechanism.
    """
    banned = (
        "add a fence directive",
        "always emit a fenced block",
        "roy white",
        "at-bats",
        "519",
        "gaia",
        "web_fetch",
        "wikipedia_search",
        "calculator",
    )
    for phrase in banned:
        assert phrase not in _SURFACE_DOC, f"prompt encodes a specific remedy: {phrase}"
