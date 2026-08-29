"""Tests for the CUGA-backed positivity judge (D5/J2B adapter).

Every test injects a fake ``completion_fn``; no test makes a network call.
(The live path was verified separately against the real endpoint.)

What is defended here
---------------------
Mirror of ``test_cuga_analyzer.py`` for the success side:

1. A specific causal strength sentence survives into ``mechanism_description``.
2. Nothing the model invents -- actors, event ids, artifacts -- produces a
   finding that claims trace-back it does not have.
3. POLARITY IS CODE-STAMPED: whatever the model returns, every emitted finding
   has ``valence == -1``, and a client-supplied ``valence`` field in the
   payload cannot change that.
4. Abstention is first-class: unparseable or thin responses become
   non-observed findings (still strengths by polarity), never exceptions that
   would launder a transport problem into a trajectory verdict.
"""
from __future__ import annotations

import json

import pytest

from agent_evolve.adapters.cuga_positivity_judge import (
    CugaPositivityJudge,
)
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace, TraceEvent

GOOD_STRENGTH = (
    "planner decomposed the multi-city itinerary first, so api_agent's flight "
    "search ran against a fully specified query and returned the answer token"
)

ANSWER_KEY = "Zq7Denver-Intl-4417"


def _task(task_id: str = "task-1") -> EvolutionTask:
    return EvolutionTask(
        task_id=task_id,
        input_text="book a flight from Boston to Denver",
        expected_contract={"expected_substring": ANSWER_KEY},
    )


def _trace(trace_id: str = "trace-1") -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        candidate_id="cand-1",
        task_id="task-1",
        events=(
            TraceEvent(
                event_id="e1",
                kind="tool_call",
                actor_id="planner",
                parent_event_id=None,
                payload={"tool": "plan", "code": "itinerary()", "skill": "skills/retrieval"},
            ),
            TraceEvent(
                event_id="e2",
                kind="tool_call",
                actor_id="api_agent",
                parent_event_id="e1",
                payload={"tool": "search_flights", "origin": "Denver"},
            ),
        ),
        final_output=f"flight booked: {ANSWER_KEY}",
        status="success",
    )


def _fake(payload: object, *, calls: list[dict] | None = None):
    body = payload if isinstance(payload, str) else json.dumps(payload)

    def completion_fn(**request):
        if calls is not None:
            calls.append(request)
        return {"choices": [{"message": {"content": body}}]}

    return completion_fn


def _judge(completion_fn, **overrides) -> CugaPositivityJudge:  # type: ignore[no-untyped-def]
    # House convention (mirrors test_cuga_analyzer): inject completion_fn AND
    # an explicit model so no credential resolution touches the environment.
    overrides.setdefault("model", "test-positivity-model")
    return CugaPositivityJudge(completion_fn=completion_fn, **overrides)


# ---------------------------------------------------------------------- #
# Happy path: grounded strengths
# ---------------------------------------------------------------------- #
def _good_payload():
    return {
        "findings": [
            {
                "status": "observed",
                "mechanism": GOOD_STRENGTH,
                "severity": 0.9,
                "confidence": 0.85,
                "blamed_actors": [
                    {"actor_id": "planner", "blame": 1.0, "artifacts": []}
                ],
                "evidence_event_ids": ["e1"],
                "rationale": "grounded in the planning step",
            }
        ]
    }


def test_emits_a_strength_with_the_model_sentence_intact() -> None:
    judge = _judge(_fake(_good_payload()))

    findings = judge.analyze_success(_task(), _trace())

    assert len(findings) == 1
    f = findings[0]
    assert f.status == "observed"
    assert f.mechanism_description == GOOD_STRENGTH
    assert f.severity == 0.9
    # House convention: event refs are qualified as "<trace_id>#<event_id>".
    assert any(ref.endswith("#e1") for ref in f.evidence_refs)


def test_polarity_is_stamped_by_code_not_by_the_model() -> None:
    """The payload carries a client 'valence'; the adapter must override."""
    payload = _good_payload()
    payload["findings"][0]["valence"] = 1  # smuggled through the JSON

    findings = _judge(_fake(payload)).analyze_success(_task(), _trace())

    assert findings[0].valence == -1


def test_protocol_shape_holds() -> None:
    judge = _judge(_fake(_good_payload()))
    assert hasattr(judge, "analyzer_model_id")
    assert callable(getattr(judge, "analyze_success"))


def test_clusters_and_stored_traces_reach_the_prompt() -> None:
    """S1.3: the judge must be told the generated clusters and the other
    candidates' traces, so it can decide *which cluster is solved better by
    which candidate*."""
    calls: list[dict] = []
    judge = _judge(_fake(_good_payload(), calls=calls))

    other = ExecutionTrace(
        trace_id="trace-other", candidate_id="other-cand", task_id="task-1",
        events=(), final_output="the other candidate's answer", status="success",
    )

    judge.analyze_success(
        _task(), _trace(),
        clusters=("the planner failed to decompose the itinerary",),
        stored_traces=(other,),
    )

    prompt = calls[0]["messages"][1]["content"]
    assert "KNOWN MECHANISM CLUSTERS" in prompt
    assert "failed to decompose the itinerary" in prompt
    assert "OTHER CANDIDATES" in prompt
    assert "other-cand" in prompt


# ---------------------------------------------------------------------- #
# Grounding: inventions are dropped, never trace-backed
# ---------------------------------------------------------------------- #
def test_invented_actor_and_events_are_dropped() -> None:
    payload = {
        "findings": [
            {
                "status": "observed",
                "mechanism": "a strength attributed to someone absent",
                "severity": 0.8,
                "confidence": 0.9,
                "blamed_actors": [
                    {"actor_id": "ghost_actor", "blame": 1.0, "artifacts": []}
                ],
                "evidence_event_ids": ["e999"],
                "rationale": "hallucinated",
            },
            _good_payload()["findings"][0],
        ]
    }

    findings = _judge(_fake(payload)).analyze_success(_task(), _trace())

    observed = [f for f in findings if f.status == "observed"]
    assert len(observed) == 1
    assert observed[0].mechanism_description == GOOD_STRENGTH
    # The dropped one stays visible as an abstaining strength, not deleted.
    dropped = [f for f in findings if f.status != "observed"]
    assert dropped and all(f.valence == -1 for f in dropped)


def test_artifact_only_attached_when_literal_in_evidence() -> None:
    payload = _good_payload()
    payload["findings"][0]["blamed_actors"][0]["artifacts"] = [
        "skills/retrieval",
        "skills/totally-invented",
    ]

    findings = _judge(_fake(payload)).analyze_success(_task(), _trace())

    node_artifacts = {
        a for n in findings[0].blame_graph.nodes for a in n.artifacts
    }
    assert "skills/retrieval" in node_artifacts
    assert "skills/totally-invented" not in node_artifacts


# ---------------------------------------------------------------------- #
# Abstention: untrustworthy responses never raise, never invent
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "payload", ["not json at all", {"unexpected": "shape"}, []]
)
def test_untrustworthy_responses_become_abstaining_strengths(payload) -> None:
    findings = _judge(_fake(payload)).analyze_success(_task(), _trace())

    assert findings
    assert all(f.valence == -1 for f in findings)
    assert all(f.status != "observed" for f in findings)


def test_transport_errors_propagate_not_laundered() -> None:
    def boom(**request):
        raise ConnectionError("endpoint down")

    with pytest.raises(ConnectionError):
        _judge(boom).analyze_success(_task(), _trace())
