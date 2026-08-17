"""Tests for the RHO candidate optimizer (Interface B x N).

N independent workspace-agent invocations, not one sampled request. Candidates
come from staged artifacts, never from parsed final text.

Diagnoses are accepted structurally (mapping keys OR attributes) so this module
does not import ``cuga_rho_diagnoser``; both shapes are exercised here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from agent_evolve.adapters.cuga_rho_optimizer import (
    APP_NAMES,
    CREATABLE_PREFIX,
    DISCARD_STATUSES,
    OPTIMIZER_INSTRUCTIONS,
    ProposalReport,
    ProposedCandidate,
    RhoOptimizer,
    build_optimizer_prompt,
)

BASE = {
    "instructions": "Answer the question.",
    "skills/search": "Use search.",
}


def _diagnosis(
    task_id: str = "gaia-1",
    severity: float = 0.9,
    *,
    observed: bool = True,
) -> dict[str, object]:
    """A diagnosis in plain-mapping form."""
    return {
        "task_id": task_id,
        "recurring_failure_mode": "narrates without executing code",
        "severity": severity,
        "improvement_direction": "require an executable step",
        "candidate_surfaces": ("instructions",),
        "rollouts_seen": 3,
        "observed": observed,
    }


@dataclass(frozen=True, slots=True)
class _AttrDiagnosis:
    """Stand-in for the diagnoser's ``GroupDiagnosis`` dataclass."""

    task_id: str
    recurring_failure_mode: str = ""
    severity: float = 0.0
    improvement_direction: str = ""
    candidate_surfaces: tuple[str, ...] = ()
    rollouts_seen: int = 0
    disagreements: tuple[str, ...] = ()
    self_validation_observed: bool = False
    observed: bool = False
    error: str = ""


# ------------------------------------------------------------------ #
# N independence
# ------------------------------------------------------------------ #


def test_issues_n_independent_invocations() -> None:
    prompts: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        prompts.append(prompt)
        callables["stage_replace"](
            artifact_id="instructions", content=f"v{len(prompts)}"
        )
        callables["submit_candidate"](rationale="r")
        return "done"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 3)

    assert isinstance(report, ProposalReport)
    assert len(prompts) == 3
    assert report.requested == 3
    assert report.distinct == 3
    assert all(isinstance(c, ProposedCandidate) for c in report.candidates)
    assert [c.candidate_index for c in report.candidates] == [0, 1, 2]


def test_each_invocation_gets_a_fresh_isolated_staging_area() -> None:
    """Candidate 1 must not inherit candidate 0's staged edits."""
    seen: list[tuple[str, ...]] = []

    def factory(callables: dict, prompt: str) -> str:
        seen.append(tuple(json.loads(callables["list_staged"]())["staged"]))
        callables["stage_replace"](artifact_id="instructions", content=f"v{len(seen)}")
        callables["submit_candidate"](rationale="r")
        return "done"

    RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 3)

    assert seen == [(), (), ()]


def test_all_n_survivors_are_retained_never_pruned_to_best_of_n() -> None:
    """Every distinct candidate survives; there is no best-of-N selection."""
    seq = {"n": 0}

    def factory(callables: dict, prompt: str) -> str:
        seq["n"] += 1
        callables["stage_replace"](
            artifact_id="instructions", content=f"variant-{seq['n']}"
        )
        callables["submit_candidate"](rationale=f"r{seq['n']}")
        return "done"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 5)

    assert report.requested == 5
    assert report.distinct == 5
    assert report.discarded == ()
    assert [c.rationale for c in report.candidates] == ["r1", "r2", "r3", "r4", "r5"]


# ------------------------------------------------------------------ #
# Capture from staged artifacts
# ------------------------------------------------------------------ #


def test_candidate_comes_from_staged_artifacts_not_answer_text() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["stage_replace"](
            artifact_id="instructions", content="STAGED CONTENT"
        )
        callables["submit_candidate"](rationale="r")
        return json.dumps({"instructions": "ANSWER TEXT CONTENT"})

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)

    assert report.candidates[0].artifacts["instructions"] == "STAGED CONTENT"
    assert "ANSWER TEXT" not in json.dumps(dict(report.candidates[0].artifacts))


def test_untouched_artifacts_are_carried_forward() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["stage_replace"](artifact_id="instructions", content="new")
        callables["submit_candidate"](rationale="r")
        return "done"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)

    assert report.candidates[0].artifacts["skills/search"] == BASE["skills/search"]
    assert report.candidates[0].artifacts["instructions"] == "new"


def test_base_artifacts_are_not_mutated() -> None:
    base = dict(BASE)

    def factory(callables: dict, prompt: str) -> str:
        callables["stage_replace"](artifact_id="instructions", content="new")
        callables["submit_candidate"](rationale="r")
        return "done"

    RhoOptimizer(agent_factory=factory).propose(base, (_diagnosis(),), 1)

    assert base == BASE


def test_created_artifact_is_captured_and_prefix_enforced() -> None:
    outcomes: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        outcomes.append(
            callables["stage_create"](artifact_id="skills/nope", content="x")
        )
        outcomes.append(
            callables["stage_create"](
                artifact_id=f"{CREATABLE_PREFIX}verify", content="verify first"
            )
        )
        callables["submit_candidate"](rationale="r")
        return "done"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)

    assert json.loads(outcomes[0])["accepted"] is False
    assert json.loads(outcomes[1])["accepted"] is True
    artifacts = report.candidates[0].artifacts
    assert artifacts[f"{CREATABLE_PREFIX}verify"] == "verify first"


def test_rationale_and_tool_ledger_are_recorded() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["list_artifacts"]()
        callables["read_artifact"](artifact_id="instructions")
        callables["stage_replace"](artifact_id="instructions", content="new")
        callables["submit_candidate"](rationale="because evidence")
        return "done"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)
    candidate = report.candidates[0]

    assert candidate.rationale == "because evidence"
    assert candidate.observed is True
    assert candidate.error == ""
    assert candidate.tools_called == (
        "list_artifacts",
        "read_artifact",
        "stage_replace",
        "submit_candidate",
    )


def test_unstage_drops_a_staged_edit() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["stage_replace"](artifact_id="instructions", content="dropped")
        callables["stage_replace"](artifact_id="skills/search", content="kept")
        callables["unstage"](artifact_id="instructions")
        callables["submit_candidate"](rationale="r")
        return "done"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)
    artifacts = report.candidates[0].artifacts

    assert artifacts["instructions"] == BASE["instructions"]
    assert artifacts["skills/search"] == "kept"


# ------------------------------------------------------------------ #
# Discard rules and failure observability
# ------------------------------------------------------------------ #


def test_staging_nothing_yields_no_candidate() -> None:
    def factory(callables: dict, prompt: str) -> str:
        return "I considered it and changed nothing."

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 2)

    assert report.candidates == ()
    assert report.distinct == 0
    assert len(report.discarded) == 2
    assert all(reason.startswith("NO_TOOL_CALL") for _, reason in report.discarded)


def test_never_finalizing_discards_the_candidate() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["stage_replace"](artifact_id="instructions", content="x")
        return "forgot to submit"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)

    assert report.candidates == ()
    assert "submit_candidate" in report.discarded[0][1]
    assert report.discarded[0][1].startswith("NOT_FINALIZED")


def test_submitting_without_staging_is_a_no_op() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["list_artifacts"]()
        callables["submit_candidate"](rationale="nothing needed")
        return "done"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)

    assert report.candidates == ()
    assert report.discarded[0][1].startswith("NO_OP")


def test_byte_identical_to_base_is_discarded_as_a_no_op() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["stage_replace"](
            artifact_id="instructions", content=BASE["instructions"]
        )
        callables["submit_candidate"](rationale="r")
        return "done"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)

    assert report.candidates == ()
    assert "identical" in report.discarded[0][1]
    assert report.discarded[0][1].startswith("IDENTICAL")


def test_duplicate_candidates_are_deduplicated() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["stage_replace"](artifact_id="instructions", content="SAME")
        callables["submit_candidate"](rationale="r")
        return "done"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 3)

    assert report.requested == 3
    assert report.distinct == 1
    assert len(report.discarded) == 2
    assert all(r.startswith("DUPLICATE") for _, r in report.discarded)


def test_unauthorized_artifact_is_rejected_at_staging() -> None:
    outcomes: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        outcomes.append(
            callables["stage_replace"](artifact_id="not/a/slot", content="x")
        )
        callables["stage_replace"](artifact_id="instructions", content="ok")
        callables["submit_candidate"](rationale="r")
        return "done"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)

    assert "not in the authorized" in outcomes[0]
    assert json.loads(outcomes[0])["accepted"] is False
    assert report.distinct == 1


def test_one_failing_invocation_does_not_lose_the_others() -> None:
    calls = {"n": 0}

    def factory(callables: dict, prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first one exploded")
        callables["stage_replace"](
            artifact_id="instructions", content=f"v{calls['n']}"
        )
        callables["submit_candidate"](rationale="r")
        return "done"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 3)

    assert report.distinct == 2
    assert len(report.discarded) == 1
    assert report.discarded[0][0] == 0
    assert report.discarded[0][1].startswith("UNAVAILABLE")
    assert "first one exploded" in report.discarded[0][1]


def test_every_discard_reason_uses_the_published_status_vocabulary() -> None:
    def factory(callables: dict, prompt: str) -> str:
        return "nothing"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)

    for _, reason in report.discarded:
        assert reason.split(":", 1)[0] in DISCARD_STATUSES


def test_report_status_counts_make_a_collapse_visible() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["stage_replace"](artifact_id="instructions", content="SAME")
        callables["submit_candidate"](rationale="r")
        return "done"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 3)

    assert report.status_counts() == {"DUPLICATE": 2}
    assert report.collapsed is True


# ------------------------------------------------------------------ #
# Evidence rendering
# ------------------------------------------------------------------ #


def test_diagnoses_are_severity_ordered_in_the_prompt() -> None:
    prompts: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        prompts.append(prompt)
        callables["stage_replace"](artifact_id="instructions", content="x")
        callables["submit_candidate"](rationale="r")
        return "done"

    low = _diagnosis("low-task", 0.2)
    high = _diagnosis("high-task", 0.95)
    RhoOptimizer(agent_factory=factory).propose(BASE, (low, high), 1)

    body = prompts[0]
    assert body.index("high-task") < body.index("low-task")


def test_unobserved_diagnoses_are_excluded_from_the_prompt() -> None:
    prompts: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        prompts.append(prompt)
        callables["stage_replace"](artifact_id="instructions", content="x")
        callables["submit_candidate"](rationale="r")
        return "done"

    bad = _AttrDiagnosis(task_id="broken-task", error="agent failed")
    RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(), bad), 1)

    assert "broken-task" not in prompts[0]


def test_attribute_style_diagnoses_are_accepted() -> None:
    """The diagnoser's dataclass works without this module importing it."""
    prompts: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        prompts.append(prompt)
        listed = json.loads(callables["list_diagnoses"]())["diagnoses"]
        assert listed[0]["task_id"] == "attr-task"
        detail = json.loads(callables["read_diagnosis"](task_id="attr-task"))
        assert detail["recurring_failure_mode"] == "skips verification"
        callables["stage_replace"](artifact_id="instructions", content="x")
        callables["submit_candidate"](rationale="r")
        return "done"

    diag = _AttrDiagnosis(
        task_id="attr-task",
        recurring_failure_mode="skips verification",
        severity=0.8,
        improvement_direction="verify",
        candidate_surfaces=("instructions",),
        rollouts_seen=3,
        observed=True,
    )
    report = RhoOptimizer(agent_factory=factory).propose(BASE, (diag,), 1)

    assert report.distinct == 1
    assert "attr-task" in prompts[0]
    assert "skips verification" in prompts[0]


def test_read_diagnosis_rejects_an_unknown_task() -> None:
    captured: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        captured.append(callables["read_diagnosis"](task_id="nope"))
        callables["stage_replace"](artifact_id="instructions", content="x")
        callables["submit_candidate"](rationale="r")
        return "done"

    RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)

    assert json.loads(captured[0])["status"] == "error"


def test_prompt_lists_artifact_ids_and_the_creatable_prefix() -> None:
    prompt = build_optimizer_prompt(BASE, ())

    assert "instructions" in prompt
    assert "skills/search" in prompt
    assert CREATABLE_PREFIX in prompt
    assert "submit_candidate" in prompt


def test_prompt_survives_an_empty_diagnosis_bundle() -> None:
    prompt = build_optimizer_prompt(BASE, ())

    assert "no observed diagnoses" in prompt


def test_artifact_content_with_braces_does_not_break_prompt_formatting() -> None:
    """Artifact bodies are data, never a format template."""
    prompt = build_optimizer_prompt({"instructions": "use {json} {0} {{x}}"}, ())

    assert "instructions" in prompt


# ------------------------------------------------------------------ #
# Prompt quality: the entire delta of this stage
# ------------------------------------------------------------------ #


def test_instructions_explain_surface_delivery_semantics() -> None:
    text = OPTIMIZER_INSTRUCTIONS

    assert "load_skill" in text
    assert "instructions" in text
    assert "policies/" in text
    assert "memory/" in text
    # The load-bearing insight: an unloaded skill changes nothing.
    assert "NOTHING" in text or "nothing" in text


def test_instructions_state_the_cuga_execution_mechanism() -> None:
    text = OPTIMIZER_INSTRUCTIONS

    assert "CugaLiteSubgraph" in text
    assert "sandbox" in text
    assert "fenced Python" in text


def test_instructions_ban_cosmetic_edits_with_concrete_examples() -> None:
    text = OPTIMIZER_INSTRUCTIONS

    assert "Think step by step" in text
    assert "TRIGGER" in text and "ACTION" in text and "CHECK" in text


def test_instructions_forbid_task_specific_hardcoding() -> None:
    lowered = OPTIMIZER_INSTRUCTIONS.lower()

    assert "task-specific" in lowered
    assert "expected answer" in lowered or "expected answers" in lowered


def test_instructions_warn_that_stage_replace_overwrites_wholesale() -> None:
    assert "does not append" in OPTIMIZER_INSTRUCTIONS


def test_the_n_invocations_do_not_receive_identical_prompts() -> None:
    """N identical prompts are ONE sample repeated N times, not N samples.

    Tool invocation is a deterministic, all-or-nothing function of prompt
    wording, and reasoning models skip temperature, so decoding is effectively
    greedy. Two live rounds proved the consequence: one discarded 3 of 3 as
    NO_TOOL_CALL, another 3 of 3 as NO_OP -- every invocation failing together
    because every invocation read the same bytes. It also caps ``distinct`` at 1
    even when all N succeed, which defeats the diversity this stage exists to
    produce.
    """
    seen: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        seen.append(prompt)
        callables["stage_replace"]("instructions", "edited body")
        return callables["submit_candidate"]("why")

    report = RhoOptimizer(agent_factory=factory).propose(
        {"instructions": "base body"}, (), 3
    )

    assert len(seen) == 3
    assert len(set(seen)) == 3, "all N invocations received the same prompt"
    # The shared evidence must still be present in every one of them.
    for prompt in seen:
        assert "instructions" in prompt
    assert report.requested == 3


def test_shared_contract_keeps_the_live_round_verified_wording() -> None:
    """The contract's wording is a live-round measurement, not a style choice.

    Tool invocation is a deterministic, all-or-nothing function of prompt
    wording on ``azure/gpt-5.6-luna``. Two measurements, same dataset:

      long contract + optimizer "write and execute" tail
          -> 2/2 diagnoses observed, 3/3 distinct candidates
      two-line contract, optimizer tail removed
          -> 0/3 candidates, all discarded NO_TOOL_CALL

    A shorter form won on a one-tool toy probe, which did NOT transfer to the
    real agents. This test pins the configuration that was verified on a live
    round so a readability edit cannot silently zero every Interface B agent's
    tool ledger.
    """
    from agent_evolve.adapters.cuga_workspace_agent import (
        WORKSPACE_AGENT_TOOL_CONTRACT,
    )

    assert "Write and execute Python code" in WORKSPACE_AGENT_TOOL_CONTRACT
    assert "ONE fenced" in WORKSPACE_AGENT_TOOL_CONTRACT
    assert "never" in WORKSPACE_AGENT_TOOL_CONTRACT
    # The optimizer's own trailing execute directive was present in the 3/3 run.
    assert "Write and execute Python code" in build_optimizer_prompt(
        {"instructions": "body"}, ()
    )
    assert "two fenced" not in OPTIMIZER_INSTRUCTIONS.lower()
    assert "multiple fenced" not in OPTIMIZER_INSTRUCTIONS.lower()


def test_optimizer_instructions_are_passed_to_the_runner() -> None:
    """The doctrine must actually reach the agent, not sit unused in the module."""
    import agent_evolve.adapters.cuga_rho_optimizer as mod

    seen: dict[str, object] = {}
    real = mod.run_workspace_agent

    def spy(callables, prompt, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return real(callables, prompt, **kwargs)

    def factory(callables: dict, prompt: str) -> str:
        callables["stage_replace"](artifact_id="instructions", content="x")
        callables["submit_candidate"](rationale="r")
        return "done"

    mod.run_workspace_agent = spy  # type: ignore[assignment]
    try:
        RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)
    finally:
        mod.run_workspace_agent = real  # type: ignore[assignment]

    assert seen["special_instructions"] is OPTIMIZER_INSTRUCTIONS
    assert seen["app_names"]


def test_every_tool_callable_has_a_docstring_and_a_real_signature() -> None:
    """LangChain's @tool raises without a docstring and needs a signature."""
    import inspect

    captured: dict[str, object] = {}

    def factory(callables: dict, prompt: str) -> str:
        captured.update(callables)
        callables["stage_replace"](artifact_id="instructions", content="x")
        callables["submit_candidate"](rationale="r")
        return "done"

    RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)

    assert set(captured) == set(APP_NAMES)
    for name, fn in captured.items():
        assert fn.__doc__, f"{name} has no docstring"
        params = inspect.signature(fn).parameters
        assert "args" not in params, f"{name} lost its typed signature"
        for param in params.values():
            assert param.annotation is not inspect.Parameter.empty


# ------------------------------------------------------------------ #
# Guardrails
# ------------------------------------------------------------------ #


def test_temperature_zero_is_refused() -> None:
    def factory(callables: dict, prompt: str) -> str:
        return "done"

    with pytest.raises(ValueError, match="temperature=0.0"):
        RhoOptimizer(agent_factory=factory, temperature=0.0).propose(
            BASE, (_diagnosis(),), 1
        )


def test_temperature_defaults_to_unset() -> None:
    assert RhoOptimizer().temperature is None


def test_n_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        RhoOptimizer(agent_factory=lambda c, p: "x").propose(BASE, (_diagnosis(),), 0)


def test_empty_base_artifacts_are_refused() -> None:
    with pytest.raises(ValueError, match="base_artifacts"):
        RhoOptimizer(agent_factory=lambda c, p: "x").propose({}, (_diagnosis(),), 1)


def test_reports_are_immutable() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["stage_replace"](artifact_id="instructions", content="x")
        callables["submit_candidate"](rationale="r")
        return "done"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)

    with pytest.raises(Exception):
        report.candidates = ()  # type: ignore[misc]
    with pytest.raises(Exception):
        report.candidates[0].rationale = "no"  # type: ignore[misc]


def test_module_does_not_import_cuga_at_top_level() -> None:
    from pathlib import Path

    import agent_evolve.adapters.cuga_rho_optimizer as mod

    source = Path(str(mod.__file__)).read_text(encoding="utf-8")
    for line in source.splitlines():
        if line.startswith(("import ", "from ")):
            assert "cuga import" not in line
            assert "langchain" not in line


def test_artifact_ids_map_to_cuga_harness_slots() -> None:
    """Every id a survivor can carry must be registerable by the adapter."""

    def factory(callables: dict, prompt: str) -> str:
        callables["stage_replace"](artifact_id="instructions", content="x")
        callables["stage_create"](
            artifact_id=f"{CREATABLE_PREFIX}probe", content="y"
        )
        callables["submit_candidate"](rationale="r")
        return "done"

    report = RhoOptimizer(agent_factory=factory).propose(BASE, (_diagnosis(),), 1)

    for artifact_id in report.candidates[0].artifacts:
        assert artifact_id == "instructions" or artifact_id.split("/")[0] in {
            "skills",
            "policies",
            "memory",
        }
    assert CREATABLE_PREFIX.startswith("skills/generated-")


def test_unused_import_guard() -> None:
    """`field` is imported by the test module only for the attr-diagnosis stub."""
    assert field is not None
