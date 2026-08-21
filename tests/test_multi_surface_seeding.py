"""SV-8 — the optimizer must be offered every editable surface, not just one.

Two independent blockers produced the observed "every candidate edits only
``instructions``" behaviour, and fixing either alone leaves the defect standing:

1. **An empty roster.** ``VANILLA_HARNESS`` — the only builtin — carries an
   ``instructions`` string and *nothing* else, so ``_harness_artifacts`` yielded
   a one-entry map and ``list_artifacts`` truthfully reported a single surface.
   The optimizer was not ignoring ``memory/`` and ``policies/``; it was never
   shown them.
2. **Single-surface creation.** ``creatable_prefix`` was the scalar
   ``"skills/generated-"``, so even with a fuller roster the only artifact the
   optimizer could *create* lived under ``skills/``.

These tests are behavioural: they asserts what the optimizer can see and stage,
not the wording of any prompt. Prompt-substring assertions were deliberately
avoided — they pass whenever the phrasing survives and say nothing about whether
a surface is reachable.
"""

from __future__ import annotations

import json

from agent_evolve import pipeline
from agent_evolve.adapters.cuga_adapter import CugaAdapter
from agent_evolve.adapters.cuga_editor_state import (
    DEFAULT_CREATABLE_PREFIXES,
    EditStagingArea,
)
from agent_evolve.benchmarks.cuga_executor import VANILLA_HARNESS, HarnessVersion
from agent_evolve.cuga_wrapper import CugaWrapper, InMemoryRuntime, RuntimeSettings

# --------------------------------------------------------------------------- #
# 1. The roster: every surface is present and editable
# --------------------------------------------------------------------------- #

SURFACES = ("skills", "memory", "policies")


def _artifacts(harness: HarnessVersion) -> dict[str, str]:
    return pipeline._harness_artifacts(harness)


def test_a_bare_harness_exposes_every_surface_not_just_one_skill() -> None:
    """The old fallback handed back exactly ``{"skills/generated-evolved": ""}``.

    A bare harness is the *live* shape: ``HarnessVersion(instructions=...)`` with
    empty ``skills``/``memory``/``policies``. Seeding one slot per surface is
    what turns "the optimizer only ever edits instructions" from a structural
    certainty into a choice it makes.
    """
    artifacts = _artifacts(HarnessVersion(version="bare"))

    for surface in SURFACES:
        seeded = [a for a in artifacts if a.startswith(f"{surface}/")]
        assert seeded, f"no editable {surface}/ slot was seeded: {sorted(artifacts)}"


def test_the_vanilla_harness_offers_all_four_surfaces() -> None:
    """The live default must show four surfaces, not one.

    This is the condition the offline stack failed to reproduce: it had several
    surfaces already, so the starved roster never appeared in a dry run.
    """
    artifacts = _artifacts(VANILLA_HARNESS)

    assert "instructions" in artifacts
    for surface in SURFACES:
        assert any(a.startswith(f"{surface}/") for a in artifacts), (
            f"{surface}/ missing from the vanilla roster: {sorted(artifacts)}"
        )


def test_seeded_slots_are_empty_and_carry_no_authored_prior() -> None:
    """Empty slots, not starter content.

    Substantive seed text would be a hand-written prior, and any measured gain
    afterwards could not be attributed to evolution rather than to the seed.
    """
    artifacts = _artifacts(HarnessVersion(version="bare"))

    for artifact_id, content in artifacts.items():
        if artifact_id == "instructions":
            continue
        assert content == "", f"{artifact_id} was seeded with authored content"


def test_real_harness_content_is_never_replaced_by_a_seed() -> None:
    """Seeding fills gaps; it must not overwrite what a harness already owns."""
    harness = HarnessVersion(
        version="rich",
        instructions="do the thing",
        skills={"retrieval": "real skill body"},
        memory={"notes": "real memory body"},
        policies={"safety": "real policy body"},
    )

    artifacts = _artifacts(harness)

    assert artifacts["skills/retrieval"] == "real skill body"
    assert artifacts["memory/notes"] == "real memory body"
    assert artifacts["policies/safety"] == "real policy body"
    assert artifacts["instructions"] == "do the thing"


def test_a_harness_with_one_surface_still_gets_the_others_seeded() -> None:
    """Partial coverage is the common real case and must not block the rest."""
    harness = HarnessVersion(
        version="partial",
        instructions="x",
        skills={"retrieval": "body"},
    )

    artifacts = _artifacts(harness)

    assert artifacts["skills/retrieval"] == "body"
    assert any(a.startswith("memory/") for a in artifacts)
    assert any(a.startswith("policies/") for a in artifacts)


def test_seeding_is_deterministic() -> None:
    """Two builds of the same harness must produce identical rosters, or a
    candidate's write set depends on invocation order."""
    a = _artifacts(VANILLA_HARNESS)
    b = _artifacts(VANILLA_HARNESS)

    assert a == b


# --------------------------------------------------------------------------- #
# 2. Every seeded surface is reachable through the adapter
# --------------------------------------------------------------------------- #


def _adapter() -> CugaAdapter:
    return CugaAdapter(
        wrapper=CugaWrapper(InMemoryRuntime(), RuntimeSettings(model="test-model"))
    )


def test_every_seeded_surface_registers_as_writable() -> None:
    """A surface absent from the adapter inventory is a surface the optimizer
    cannot edit, whatever the roster claims."""
    adapter = _adapter()
    adapter.register_candidate("base", _artifacts(VANILLA_HARNESS))

    writable = {
        d.artifact_id for d in adapter.artifact_inventory("base") if d.writable
    }

    assert "instructions" in writable
    for surface in SURFACES:
        assert any(a.startswith(f"{surface}/") for a in writable), (
            f"no writable {surface}/ artifact: {sorted(writable)}"
        )


def test_every_seeded_surface_maps_to_a_real_cuga_harness_slot() -> None:
    """Guards against seeding an id CUGA cannot receive.

    ``_harness_slot`` raises for unmappable ids, so a bad seed would surface as
    a registration crash rather than a silently dropped artifact.
    """
    adapter = _adapter()

    for artifact_id in _artifacts(VANILLA_HARNESS):
        key, _member = adapter._harness_slot(artifact_id)
        assert key in ("instructions", *SURFACES)


# --------------------------------------------------------------------------- #
# 3. Creation is no longer confined to skills/
# --------------------------------------------------------------------------- #


def test_creation_is_allowed_under_every_surface() -> None:
    """The second blocker. With a scalar ``skills/generated-`` prefix the
    optimizer could only ever create a skill, so ``memory/`` and ``policies/``
    were reachable by replacement alone.

    The cap is raised here only to test all three surfaces in one attempt; the
    default cap of 2 is asserted separately below.
    """
    area = EditStagingArea(write_set=("instructions",), per_attempt_create_cap=3)

    for surface in SURFACES:
        outcome = area.stage_create(f"{surface}/generated-probe", "body")
        assert outcome.accepted, f"{surface}/ creation rejected: {outcome.reason}"


def test_creation_still_requires_the_generated_marker() -> None:
    """Widening the surfaces must not widen *what* may be created: provenance
    depends on generated artifacts being identifiable by id."""
    area = EditStagingArea(write_set=("instructions",))

    outcome = area.stage_create("memory/handwritten", "body")

    assert not outcome.accepted
    assert "memory/generated-" in outcome.reason


def test_creation_outside_a_known_surface_is_still_rejected() -> None:
    """A flat ``generated-x`` maps to no CUGA slot and would raise at
    registration, making the creation path dead code."""
    area = EditStagingArea(write_set=("instructions",))

    assert not area.stage_create("generated-flat", "body").accepted
    assert not area.stage_create("tools/generated-x", "body").accepted


def test_a_created_artifact_still_needs_a_name_after_the_prefix() -> None:
    area = EditStagingArea(write_set=("instructions",))

    outcome = area.stage_create("memory/generated-", "body")

    assert not outcome.accepted


def test_the_per_attempt_creation_cap_survives_the_widening() -> None:
    """Three surfaces must not become three times the creation budget."""
    area = EditStagingArea(write_set=("instructions",), per_attempt_create_cap=2)

    assert area.stage_create("skills/generated-a", "a").accepted
    assert area.stage_create("memory/generated-b", "b").accepted
    denied = area.stage_create("policies/generated-c", "c")

    assert not denied.accepted
    assert "cap" in denied.reason


def test_disabling_creation_still_disables_it_entirely() -> None:
    """The genetic editor defaults to creation disabled; an empty prefix set
    must not be read as "any prefix is acceptable"."""
    area = EditStagingArea(write_set=("instructions",), creatable_prefixes=())

    assert not area.stage_create("skills/generated-a", "a").accepted
    assert not area.stage_create("memory/generated-b", "b").accepted


def test_the_declared_default_prefixes_cover_every_surface() -> None:
    assert {p.split("/", 1)[0] for p in DEFAULT_CREATABLE_PREFIXES} == set(SURFACES)


def test_the_adapter_counts_generated_artifacts_across_all_surfaces() -> None:
    """The pool-wide creation cap reads this count; missing a surface would let
    generated artifacts accumulate past the cap unnoticed."""
    adapter = _adapter()
    adapter.register_candidate(
        "v1",
        {
            "instructions": "x",
            "skills/generated-a": "a",
            "memory/generated-b": "b",
            "policies/generated-c": "c",
            "skills/handwritten": "h",
        },
    )

    assert adapter.created_artifact_count("v1") == 3


# --------------------------------------------------------------------------- #
# 4. The optimizer's own tool surface reports the widened roster
# --------------------------------------------------------------------------- #


def test_list_artifacts_reports_every_surface_and_all_prefixes() -> None:
    """What the optimizer can *see*. Asserts the tool payload rather than the
    prompt text, so it fails when reachability regresses, not when wording
    changes."""
    from agent_evolve.adapters.cuga_rho_optimizer import build_optimizer_callables

    base = _artifacts(VANILLA_HARNESS)
    staging = EditStagingArea(write_set=tuple(sorted(base)))
    plan: dict = {}
    callables = build_optimizer_callables(base, (), staging, plan)

    payload = json.loads(callables["list_artifacts"]())
    listed = payload["artifacts"]

    assert "instructions" in listed
    for surface in SURFACES:
        assert any(a.startswith(f"{surface}/") for a in listed), (
            f"{surface}/ absent from list_artifacts: {listed}"
        )
    prefixes = payload["creatable_prefixes"]
    assert {p.split("/", 1)[0] for p in prefixes} == set(SURFACES)


def test_the_optimizer_can_read_every_seeded_surface() -> None:
    """A listed-but-unreadable artifact would stall the optimizer mid-attempt."""
    from agent_evolve.adapters.cuga_rho_optimizer import build_optimizer_callables

    base = _artifacts(VANILLA_HARNESS)
    staging = EditStagingArea(write_set=tuple(sorted(base)))
    plan: dict = {}
    callables = build_optimizer_callables(base, (), staging, plan)

    for artifact_id in base:
        payload = json.loads(callables["read_artifact"](artifact_id))
        assert payload["status"] == "ok", payload


def test_the_optimizer_can_stage_a_replacement_on_a_non_skill_surface() -> None:
    """End of the SV-8 chain: a seeded ``memory/`` slot must be replaceable, or
    the surface is visible but inert."""
    from agent_evolve.adapters.cuga_rho_optimizer import build_optimizer_callables

    base = _artifacts(VANILLA_HARNESS)
    staging = EditStagingArea(write_set=tuple(sorted(base)))
    plan: dict = {}
    callables = build_optimizer_callables(base, (), staging, plan)

    target = next(a for a in sorted(base) if a.startswith("memory/"))
    payload = json.loads(callables["stage_replace"](target, "learned note"))

    assert payload["accepted"], payload
    assert target in staging.staged_ids()
