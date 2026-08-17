"""Tests for RHO group diagnosis (Interface B).

Self-validation and self-consistency are two signals from ONE invocation per
task, not two calls. Every test injects an ``agent_factory``; no test touches the
network.

The diagnosis is captured from the terminal tool's SIDE EFFECT, never parsed out
of the agent's final prose. A model that emits a perfectly-shaped JSON object
without executing a tool is an observable ``NO_TOOL_CALL``, not a success: that
distinction is the whole reason Interface B exists, so it is asserted directly.
"""
from __future__ import annotations

import json

from agent_evolve.adapters.cuga_rho_diagnoser import (
    BANNED_MECHANISM_PHRASES,
    DIAGNOSER_INSTRUCTIONS,
    DIAGNOSIS_PROMPT,
    GroupDiagnosis,
    RhoGroupDiagnoser,
    RolloutAssessment,
    VALID_SURFACES,
    build_diagnosis_prompt,
    validate_diagnosis_payload,
)
from agent_evolve.core.contracts import ExecutionTrace, TraceEvent


# --------------------------------------------------------------- fixtures
def _event(event_id: str, kind: str, actor_id: str, **payload: object) -> TraceEvent:
    return TraceEvent(
        event_id=event_id,
        kind=kind,
        actor_id=actor_id,
        parent_event_id=None,
        payload=payload,
    )


def _trace(
    trace_id: str,
    output: str,
    *,
    events: tuple[TraceEvent, ...] | None = None,
    status: str = "failure",
) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        candidate_id="base",
        task_id="gaia-1",
        events=events
        if events is not None
        else (_event("e1", "llm_call_end", "call_model", text="thinking"),),
        final_output=output,
        status=status,
    )


def _traces() -> tuple[ExecutionTrace, ...]:
    return (_trace("t1", "17"), _trace("t2", "unknown"), _trace("t3", "18"))


def _good() -> dict:
    return {
        "recurring_failure_mode": "narrates a plan without executing any tool",
        "disagreements": ["two rollouts committed different numbers"],
        "self_validation_observed": False,
        "severity": 0.8,
        "improvement_direction": "require an executable step before answering",
        "candidate_surfaces": ["instructions"],
    }


def _note_all(callables: dict, traces=None) -> None:
    """Satisfy the per-rollout reading gate the way a compliant agent would."""
    for trace in _traces() if traces is None else traces:
        callables["note_rollout"](
            rollout_id=trace.trace_id,
            likely_successful=False,
            verified_own_answer=False,
            issue="committed an answer without re-deriving it",
        )


def _acting_factory(traces=None, **overrides):
    """A fake agent that behaves correctly: notes each rollout, then submits."""

    def factory(callables: dict, prompt: str) -> str:
        _note_all(callables, traces)
        return callables["submit_diagnosis"](**{**_good(), **overrides})

    return factory


# ------------------------------------------------------- happy-path shape
def test_one_invocation_per_task() -> None:
    calls: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        calls.append(prompt)
        _note_all(callables)
        return callables["submit_diagnosis"](**_good())

    diagnosis = RhoGroupDiagnoser(agent_factory=factory).diagnose(
        "gaia-1", "how many albums", _traces()
    )

    assert len(calls) == 1
    assert isinstance(diagnosis, GroupDiagnosis)
    assert diagnosis.observed is True
    assert diagnosis.status == "OK"
    assert diagnosis.task_id == "gaia-1"
    assert diagnosis.error == ""


def test_extracts_both_self_validation_and_self_consistency() -> None:
    diagnosis = RhoGroupDiagnoser(agent_factory=_acting_factory()).diagnose(
        "gaia-1", "q", _traces()
    )

    assert diagnosis.self_validation_observed is False
    assert diagnosis.disagreements == ("two rollouts committed different numbers",)
    assert diagnosis.severity == 0.8
    assert diagnosis.recurring_failure_mode == (
        "narrates a plan without executing any tool"
    )
    assert diagnosis.improvement_direction == (
        "require an executable step before answering"
    )
    assert diagnosis.candidate_surfaces == ("instructions",)


def test_records_how_many_rollouts_were_seen() -> None:
    two = _traces()[:2]
    diagnosis = RhoGroupDiagnoser(agent_factory=_acting_factory(two)).diagnose(
        "gaia-1", "q", two
    )

    assert diagnosis.rollouts_seen == 2
    assert diagnosis.observed is True


def test_per_rollout_assessments_are_captured() -> None:
    diagnosis = RhoGroupDiagnoser(agent_factory=_acting_factory()).diagnose(
        "gaia-1", "q", _traces()
    )

    assert len(diagnosis.per_rollout) == 3
    assert isinstance(diagnosis.per_rollout[0], RolloutAssessment)
    assert diagnosis.per_rollout[0].rollout_id == "t1"
    assert diagnosis.per_rollout[0].verified_own_answer is False


def test_tools_actually_executed_are_recorded() -> None:
    diagnosis = RhoGroupDiagnoser(agent_factory=_acting_factory()).diagnose(
        "gaia-1", "q", _traces()
    )

    assert "submit_diagnosis" in diagnosis.tools_called
    assert diagnosis.tools_called.count("note_rollout") == 3


# ------------------------------------------------------------- evidence tools
def test_all_g_trajectories_are_available_to_the_agent() -> None:
    seen: list[dict] = []

    def factory(callables: dict, prompt: str) -> str:
        listed = json.loads(callables["list_rollouts"]())
        seen.append(listed)
        _note_all(callables)
        return callables["submit_diagnosis"](**_good())

    RhoGroupDiagnoser(agent_factory=factory).diagnose("gaia-1", "q", _traces())

    assert len(seen[0]["rollouts"]) == 3
    assert [r["rollout_id"] for r in seen[0]["rollouts"]] == ["t1", "t2", "t3"]


def test_get_task_returns_the_task_input() -> None:
    seen: list[dict] = []

    def factory(callables: dict, prompt: str) -> str:
        seen.append(json.loads(callables["get_task"]()))
        _note_all(callables)
        return callables["submit_diagnosis"](**_good())

    RhoGroupDiagnoser(agent_factory=factory).diagnose(
        "gaia-1", "how many studio albums", _traces()
    )

    assert seen[0]["input"] == "how many studio albums"
    assert seen[0]["task_id"] == "gaia-1"


def test_rollout_listing_separates_narration_from_real_tool_execution() -> None:
    """The narrated-vs-failed distinction needs tool observations, not prose."""
    narrated = _trace(
        "narrated",
        "I would call the search tool",
        events=(_event("e1", "llm_call_end", "call_model", text="I will search"),),
    )
    executed = _trace(
        "executed",
        "17",
        events=(
            _event("e1", "llm_call_end", "call_model", text="searching"),
            _event("e2", "tool_call", "sandbox", name="search", output="none"),
        ),
    )
    seen: list[dict] = []

    def factory(callables: dict, prompt: str) -> str:
        seen.append(json.loads(callables["list_rollouts"]()))
        _note_all(callables, (narrated, executed))
        return callables["submit_diagnosis"](**_good())

    RhoGroupDiagnoser(agent_factory=factory).diagnose(
        "gaia-1", "q", (narrated, executed)
    )

    by_id = {r["rollout_id"]: r for r in seen[0]["rollouts"]}
    assert by_id["narrated"]["tool_observations"] == 0
    assert by_id["narrated"]["executed_any_tool"] is False
    assert by_id["executed"]["tool_observations"] == 1
    assert by_id["executed"]["executed_any_tool"] is True


def test_read_rollout_events_returns_that_rollouts_events() -> None:
    seen: list[dict] = []

    def factory(callables: dict, prompt: str) -> str:
        seen.append(json.loads(callables["read_rollout_events"](rollout_id="t2")))
        _note_all(callables)
        return callables["submit_diagnosis"](**_good())

    RhoGroupDiagnoser(agent_factory=factory).diagnose("gaia-1", "q", _traces())

    assert seen[0]["rollout_id"] == "t2"
    assert seen[0]["events"][0]["kind"] == "llm_call_end"


def test_read_rollout_events_on_unknown_id_returns_error_not_raise() -> None:
    seen: list[dict] = []

    def factory(callables: dict, prompt: str) -> str:
        seen.append(json.loads(callables["read_rollout_events"](rollout_id="nope")))
        _note_all(callables)
        return callables["submit_diagnosis"](**_good())

    diagnosis = RhoGroupDiagnoser(agent_factory=factory).diagnose(
        "gaia-1", "q", _traces()
    )

    assert seen[0]["status"] == "error"
    assert diagnosis.observed is True


def test_long_event_payloads_are_truncated_to_keep_the_prompt_bounded() -> None:
    big = _trace(
        "big",
        "x",
        events=(_event("e1", "llm_call_end", "call_model", text="z" * 50_000),),
    )
    seen: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        seen.append(callables["read_rollout_events"](rollout_id="big"))
        _note_all(callables, (big,))
        return callables["submit_diagnosis"](**_good())

    RhoGroupDiagnoser(agent_factory=factory).diagnose("gaia-1", "q", (big,))

    assert len(seen[0]) < 10_000
    assert "truncated" in seen[0]


# --------------------------------------------------- in-loop validation gate
def test_submit_is_rejected_until_every_rollout_has_been_assessed() -> None:
    replies: list[dict] = []

    def factory(callables: dict, prompt: str) -> str:
        replies.append(json.loads(callables["submit_diagnosis"](**_good())))
        _note_all(callables)
        replies.append(json.loads(callables["submit_diagnosis"](**_good())))
        return "done"

    diagnosis = RhoGroupDiagnoser(agent_factory=factory).diagnose(
        "gaia-1", "q", _traces()
    )

    assert replies[0]["status"] == "rejected"
    assert "note_rollout" in replies[0]["reason"]
    assert replies[1]["status"] == "ok"
    assert diagnosis.observed is True


def test_severity_out_of_range_is_rejected() -> None:
    diagnosis = RhoGroupDiagnoser(
        agent_factory=_acting_factory(severity=5.0)
    ).diagnose("gaia-1", "q", _traces())

    assert diagnosis.observed is False
    assert "severity" in diagnosis.error
    assert diagnosis.status == "REJECTED"


def test_unmappable_surface_is_rejected() -> None:
    diagnosis = RhoGroupDiagnoser(
        agent_factory=_acting_factory(candidate_surfaces=["nonsense"])
    ).diagnose("gaia-1", "q", _traces())

    assert diagnosis.observed is False
    assert "surface" in diagnosis.error


def test_every_valid_surface_maps_to_a_real_cuga_harness_slot() -> None:
    assert VALID_SURFACES == frozenset(
        {"instructions", "skills", "policies", "memory"}
    )
    for surface in sorted(VALID_SURFACES):
        diagnosis = RhoGroupDiagnoser(
            agent_factory=_acting_factory(candidate_surfaces=[surface])
        ).diagnose("gaia-1", "q", _traces())
        assert diagnosis.observed is True, surface


def test_empty_mechanism_is_rejected() -> None:
    diagnosis = RhoGroupDiagnoser(
        agent_factory=_acting_factory(recurring_failure_mode="   ")
    ).diagnose("gaia-1", "q", _traces())

    assert diagnosis.observed is False
    assert "recurring_failure_mode" in diagnosis.error


def test_vague_non_discriminative_mechanism_is_rejected() -> None:
    """A mechanism the optimizer cannot act on is not a diagnosis."""
    diagnosis = RhoGroupDiagnoser(
        agent_factory=_acting_factory(
            recurring_failure_mode="the agent did not follow instructions carefully"
        )
    ).diagnose("gaia-1", "q", _traces())

    assert diagnosis.observed is False
    assert "vague" in diagnosis.error.lower()


def test_one_word_mechanism_is_rejected() -> None:
    diagnosis = RhoGroupDiagnoser(
        agent_factory=_acting_factory(recurring_failure_mode="bad")
    ).diagnose("gaia-1", "q", _traces())

    assert diagnosis.observed is False
    assert "recurring_failure_mode" in diagnosis.error


def test_rejection_reason_is_actionable_and_reaches_the_agent() -> None:
    replies: list[dict] = []

    def factory(callables: dict, prompt: str) -> str:
        _note_all(callables)
        replies.append(
            json.loads(callables["submit_diagnosis"](**{**_good(), "severity": 9.0}))
        )
        replies.append(json.loads(callables["submit_diagnosis"](**_good())))
        return "done"

    diagnosis = RhoGroupDiagnoser(agent_factory=factory).diagnose(
        "gaia-1", "q", _traces()
    )

    assert replies[0]["status"] == "rejected"
    assert "severity" in replies[0]["reason"]
    # A rejection is recoverable inside the same invocation.
    assert replies[1]["status"] == "ok"
    assert diagnosis.observed is True


def test_note_rollout_on_unknown_id_is_refused() -> None:
    replies: list[dict] = []

    def factory(callables: dict, prompt: str) -> str:
        replies.append(
            json.loads(
                callables["note_rollout"](
                    rollout_id="ghost",
                    likely_successful=False,
                    verified_own_answer=False,
                    issue="x",
                )
            )
        )
        return "done"

    RhoGroupDiagnoser(agent_factory=factory).diagnose("gaia-1", "q", _traces())

    assert replies[0]["status"] == "error"


# ------------------------------------------------------ failure observability
def test_json_in_the_answer_without_a_tool_call_is_not_a_success() -> None:
    """Prose is not evidence. A shaped answer with no executed tool is NO_TOOL_CALL."""

    def factory(callables: dict, prompt: str) -> str:
        return json.dumps(_good())

    diagnosis = RhoGroupDiagnoser(agent_factory=factory).diagnose(
        "gaia-1", "q", _traces()
    )

    assert diagnosis.observed is False
    assert diagnosis.status == "NO_TOOL_CALL"


def test_malformed_json_is_unobserved() -> None:
    def factory(callables: dict, prompt: str) -> str:
        return "not json"

    diagnosis = RhoGroupDiagnoser(agent_factory=factory).diagnose(
        "gaia-1", "q", _traces()
    )

    assert diagnosis.observed is False
    assert diagnosis.status == "NO_TOOL_CALL"


def test_tools_used_but_never_submitted_is_a_no_op() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["list_rollouts"]()
        return "I have finished my analysis."

    diagnosis = RhoGroupDiagnoser(agent_factory=factory).diagnose(
        "gaia-1", "q", _traces()
    )

    assert diagnosis.observed is False
    assert diagnosis.status == "NO_OP"
    assert diagnosis.tools_called == ("list_rollouts",)


def test_agent_failure_is_unobserved_not_raised() -> None:
    def factory(callables: dict, prompt: str) -> str:
        raise RuntimeError("boom")

    diagnosis = RhoGroupDiagnoser(agent_factory=factory).diagnose(
        "gaia-1", "q", _traces()
    )

    assert diagnosis.observed is False
    assert "boom" in diagnosis.error
    assert diagnosis.status == "UNAVAILABLE"


def test_tool_evidence_survives_an_agent_crash() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["list_rollouts"]()
        raise RuntimeError("boom")

    diagnosis = RhoGroupDiagnoser(agent_factory=factory).diagnose(
        "gaia-1", "q", _traces()
    )

    assert diagnosis.status == "UNAVAILABLE"
    assert diagnosis.tools_called == ("list_rollouts",)


def test_empty_trace_group_is_unobserved() -> None:
    called: list[str] = []

    def factory(callables: dict, prompt: str) -> str:
        called.append("ran")
        return "x"

    diagnosis = RhoGroupDiagnoser(agent_factory=factory).diagnose("gaia-1", "q", ())

    assert diagnosis.observed is False
    assert "no rollouts" in diagnosis.error
    assert diagnosis.status == "NO_ROLLOUTS"
    assert called == []  # no invocation is spent on an empty group


def test_a_diagnosis_submitted_after_a_crash_is_still_kept() -> None:
    """Capture is a side effect, so it survives a later failure in the same run."""

    def factory(callables: dict, prompt: str) -> str:
        _note_all(callables)
        callables["submit_diagnosis"](**_good())
        raise RuntimeError("late boom")

    diagnosis = RhoGroupDiagnoser(agent_factory=factory).diagnose(
        "gaia-1", "q", _traces()
    )

    assert diagnosis.observed is True
    assert diagnosis.status == "OK"
    assert "late boom" in diagnosis.error


# ------------------------------------------------------------ prompt quality
def test_every_tool_is_a_real_typed_documented_callable() -> None:
    """LangChain's @tool needs a docstring, and a signature to build args from."""
    import inspect

    captured: dict = {}

    def factory(callables: dict, prompt: str) -> str:
        captured.update(callables)
        return "x"

    RhoGroupDiagnoser(agent_factory=factory).diagnose("gaia-1", "q", _traces())

    assert set(captured) == {
        "get_task",
        "list_rollouts",
        "read_rollout_events",
        "note_rollout",
        "submit_diagnosis",
    }
    for name, fn in captured.items():
        assert (fn.__doc__ or "").strip(), f"{name} has no docstring"
        signature = inspect.signature(fn)
        assert "args" not in signature.parameters, f"{name} lost its signature"

    assert "rollout_id" in inspect.signature(captured["note_rollout"]).parameters
    assert (
        "recurring_failure_mode"
        in inspect.signature(captured["submit_diagnosis"]).parameters
    )


def test_prompt_states_the_real_sdk_graph_and_the_fence_rule() -> None:
    prompt = build_diagnosis_prompt("gaia-1", "q", _traces())

    assert "call_model" in prompt and "sandbox" in prompt
    assert "FinalAnswerAgent" in prompt
    assert "fenced" in prompt.lower()
    assert "3" in prompt  # the group size the agent must account for


def test_prompt_forbids_inferring_tool_failure_from_prose() -> None:
    prompt = build_diagnosis_prompt("gaia-1", "q", _traces())
    lowered = prompt.lower()

    assert "tool observation" in lowered
    assert "narrat" in lowered


def test_prompt_demands_a_mechanism_not_a_symptom() -> None:
    prompt = build_diagnosis_prompt("gaia-1", "q", _traces())
    lowered = prompt.lower()

    assert "mechanism" in lowered
    assert "symptom" in lowered
    # A negative example list is what keeps mechanisms discriminative.
    assert any(phrase in lowered for phrase in BANNED_MECHANISM_PHRASES)


def test_prompt_anchors_the_severity_scale() -> None:
    prompt = build_diagnosis_prompt("gaia-1", "q", _traces())

    # Without anchors every severity collapses to 0.7/0.8 and the ordering
    # signal phase 6 depends on is dead.
    for anchor in ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0"):
        assert anchor in prompt


def test_instructions_carry_the_invariants_and_do_not_fight_the_contract() -> None:
    lowered = DIAGNOSER_INSTRUCTIONS.lower()

    assert "ground truth" in lowered
    assert "agreement" in lowered  # agreement is not correctness
    assert "harness" in lowered
    assert "expected answer" in lowered  # no answer leakage into reusable text
    # The shared contract owns the fence rule; instructions must not contradict.
    assert "two fenced" not in lowered
    assert "multiple fenced blocks" not in lowered


def test_prompt_requires_generality_of_the_improvement_direction() -> None:
    prompt = build_diagnosis_prompt("gaia-1", "q", _traces())

    assert "general" in prompt.lower()
    assert "task-specific" in prompt.lower()


def test_prompt_scales_its_stated_group_size() -> None:
    assert " 2 independent rollouts" in build_diagnosis_prompt(
        "gaia-1", "q", _traces()[:2]
    )
    assert " 3 independent rollouts" in build_diagnosis_prompt(
        "gaia-1", "q", _traces()
    )


def test_prompt_template_and_builder_agree() -> None:
    assert "{count}" in DIAGNOSIS_PROMPT
    built = build_diagnosis_prompt("gaia-1", "q", _traces())
    assert built.startswith(DIAGNOSIS_PROMPT.format(count=3, task_id="gaia-1"))


def test_prompt_ends_with_an_explicit_execute_directive() -> None:
    """The last thing the model reads must tell it to emit a fenced block.

    Measured on ``azure/gpt-5.6-luna``: tool invocation is a deterministic
    function of prompt wording, and a prompt that ends on the submission schema
    produced a complete narration -- including "Diagnosis submitted
    successfully" -- with an empty tool ledger. Reordering alone flipped the
    same task to seven executed calls, so the trailing directive is load
    bearing, not decoration.
    """
    built = build_diagnosis_prompt("gaia-1", "q", _traces())
    tail = built[-500:]
    assert "Write and execute Python code" in tail
    assert "fenced" in tail


def test_instructions_are_passed_as_special_instructions(monkeypatch) -> None:
    """Interface B behavioural config must actually reach the agent."""
    import agent_evolve.adapters.cuga_rho_diagnoser as module

    seen: dict = {}
    real = module.run_workspace_agent

    def spy(callables, prompt, **kwargs):
        seen.update(kwargs)
        return real(callables, prompt, **kwargs)

    monkeypatch.setattr(module, "run_workspace_agent", spy)
    RhoGroupDiagnoser(agent_factory=_acting_factory()).diagnose(
        "gaia-1", "q", _traces()
    )

    assert seen["special_instructions"] == DIAGNOSER_INSTRUCTIONS
    assert set(seen["app_names"]) >= {"submit_diagnosis", "list_rollouts"}


# ------------------------------------------------------------ pure validator
def test_validator_is_reusable_and_returns_empty_string_on_success() -> None:
    assert validate_diagnosis_payload(_good()) == ""
    assert "severity" in validate_diagnosis_payload({**_good(), "severity": "abc"})
    assert "severity" in validate_diagnosis_payload({**_good(), "severity": -1})
    assert "list" in validate_diagnosis_payload(
        {**_good(), "candidate_surfaces": "instructions"}
    )
