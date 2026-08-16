"""Tests for the LLM-backed trajectory analyzer.

Every test injects a fake ``completion_fn``; no test makes a network call.

What is actually being defended here
------------------------------------
The analyzer replaces a placeholder that emitted one templated mechanism per
task, which made every failure on a task indistinguishable downstream. So the
tests care about two properties above all: (1) a real causal sentence survives
intact into ``mechanism_description``, and (2) nothing the model invents --
actors, artifacts, event ids, out-of-range numbers, non-JSON -- can produce a
finding that claims to be trace-backed when it is not.
"""
from __future__ import annotations

import json
import threading

import pytest

from agent_evolve.adapters.cuga_analyzer import (
    UNASSIGNED_MECHANISM_CLUSTER_ID,
    AnalyzerConfigurationError,
    CugaTrajectoryAnalyzer,
)
from agent_evolve.core.analysis import RolloutGroupReport
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace, TraceEvent
from agent_evolve.core.evidence import rollout_group_report
from agent_evolve.core.parallel_analysis import ParallelAnalysisRunner

# ---------------------------------------------------------------------- #
# Fixtures / builders
# ---------------------------------------------------------------------- #
GOOD_MECHANISM = (
    "the planner emitted code that never called load_skill, so the required "
    "skill body was never in the model context when it produced an answer"
)


#: A distinctive answer-key token, so a leak test cannot pass or fail by
#: accident on a common substring appearing in the prompt's own instructions.
ANSWER_KEY = "Zq7Denver-Intl-4417"


def _task(task_id: str = "task-1") -> EvolutionTask:
    return EvolutionTask(
        task_id=task_id,
        input_text="book a flight from Boston to Denver",
        expected_contract={"expected_substring": ANSWER_KEY},
    )


def _trace(
    trace_id: str = "trace-1",
    *,
    candidate_id: str = "cand-1",
    task_id: str = "task-1",
    status: str = "failure",
) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        candidate_id=candidate_id,
        task_id=task_id,
        events=(
            TraceEvent(
                event_id="e1",
                kind="tool_call",
                actor_id="planner",
                parent_event_id=None,
                payload={"tool": "plan", "code": "answer()"},
            ),
            TraceEvent(
                event_id="e2",
                kind="tool_call",
                actor_id="api_agent",
                parent_event_id="e1",
                payload={"tool": "search_flights", "origin": "Denver"},
            ),
        ),
        final_output="a flight to Boston",
        status=status,
    )


def _report(*traces: ExecutionTrace, task: EvolutionTask | None = None):
    return rollout_group_report(task or _task(), list(traces) or [_trace()])


def _fake(payload: object, *, calls: list[dict] | None = None):
    """A completion_fn returning ``payload`` (dict/list -> JSON, str verbatim)."""
    body = payload if isinstance(payload, str) else json.dumps(payload)

    def completion_fn(**request):
        if calls is not None:
            calls.append(request)
        return {"choices": [{"message": {"content": body}}]}

    return completion_fn


def _analyzer(payload: object, *, calls: list[dict] | None = None, **kwargs):
    return CugaTrajectoryAnalyzer(
        completion_fn=_fake(payload, calls=calls), model="test/model", **kwargs
    )


def _observed_payload(**overrides) -> dict:
    finding = {
        "trace_id": "trace-1",
        "status": "observed",
        "mechanism": GOOD_MECHANISM,
        "severity": 0.9,
        "confidence": 0.75,
        "rationale": "e1 shows the planner emitting code with no load_skill call",
        "blamed_actors": [
            {"actor_id": "planner", "blame": 0.8, "artifacts": ["plan"]},
            {"actor_id": "api_agent", "blame": 0.2},
        ],
        "causal_links": [
            {
                "from_actor": "planner",
                "to_actor": "api_agent",
                "mechanism": "passed a plan with no skill load step",
            }
        ],
        "evidence_event_ids": ["e1", "e2"],
        "counterfactual_notes": ["loading the skill first would supply the body"],
    }
    finding.update(overrides)
    return {"findings": [finding]}


# ---------------------------------------------------------------------- #
# Well-formed observed finding
# ---------------------------------------------------------------------- #
def test_observed_finding_carries_the_verbatim_causal_mechanism():
    findings = _analyzer(_observed_payload()).analyze(_report())

    assert len(findings) == 1
    f = findings[0]
    assert f.status == "observed"
    # The point of the module: the causal sentence is preserved verbatim, not
    # normalized into a category label.
    assert f.mechanism_description == GOOD_MECHANISM
    assert f.severity == 0.9
    assert f.confidence == 0.75
    assert f.mechanism_cluster_id == UNASSIGNED_MECHANISM_CLUSTER_ID
    assert f.candidate_id == "cand-1"
    assert f.task_id == "task-1"
    assert f.trace_id == "trace-1"
    assert f.counterfactual_notes == (
        "loading the skill first would supply the body",
    )


def test_observed_finding_grounds_blame_nodes_and_event_refs():
    f = _analyzer(_observed_payload()).analyze(_report())[0]

    assert [n.actor_id for n in f.blame_graph.nodes] == ["planner", "api_agent"]
    assert f.blame_graph.artifacts_for("planner") == ("plan",)
    # Event refs are namespaced by trace so a multi-trace group cannot conflate
    # two rollouts' events.
    assert "trace-1#e1" in f.evidence_refs
    assert "trace-1#e2" in f.evidence_refs
    # The artifact must be in evidence_refs or CausalFinding would have rejected
    # the whole finding; assert it explicitly so a regression is legible.
    assert "plan" in f.evidence_refs
    assert len(f.blame_graph.edges) == 1


def test_analyzer_never_sends_the_answer_key_to_the_model():
    """The bridge sanitizes, but the prompt is where a leak would land."""
    calls: list[dict] = []
    _analyzer(_observed_payload(), calls=calls).analyze(_report())

    blob = json.dumps(calls[0]["messages"])
    assert ANSWER_KEY not in blob
    assert "a flight to Boston" not in blob  # final_output
    assert "book a flight from Boston to Denver" in blob  # task input is allowed


# ---------------------------------------------------------------------- #
# Abstention
# ---------------------------------------------------------------------- #
def test_abstention_maps_to_insufficient_evidence_without_a_mechanism():
    payload = {
        "findings": [
            {
                "trace_id": "trace-1",
                "status": "insufficient_evidence",
                "rationale": "no tool payloads were visible; cannot attribute",
            }
        ]
    }
    f = _analyzer(payload).analyze(_report())[0]

    assert f.status == "insufficient_evidence"
    assert f.mechanism_description is None
    assert f.mechanism_cluster_id is None
    assert f.blame_graph.nodes == ()
    assert "cannot attribute" in f.rationale


@pytest.mark.parametrize("alias", ["abstain", "INSUFFICIENT-EVIDENCE", "unknown"])
def test_abstention_aliases_are_accepted(alias):
    payload = {
        "findings": [
            {"trace_id": "trace-1", "status": alias, "rationale": "not enough"}
        ]
    }
    assert _analyzer(payload).analyze(_report())[0].status == "insufficient_evidence"


def test_abstention_never_invents_a_mechanism_even_if_the_model_supplies_one():
    """A model that abstains but still writes a mechanism must not be promoted."""
    payload = {
        "findings": [
            {
                "trace_id": "trace-1",
                "status": "insufficient_evidence",
                "mechanism": GOOD_MECHANISM,
                "blamed_actors": [{"actor_id": "planner", "blame": 1.0}],
                "rationale": "guessing",
            }
        ]
    }
    f = _analyzer(payload).analyze(_report())[0]

    assert f.status == "insufficient_evidence"
    assert f.mechanism_description is None
    assert f.blame_graph.nodes == ()


# ---------------------------------------------------------------------- #
# Malformed responses
# ---------------------------------------------------------------------- #
def test_malformed_json_becomes_a_malformed_finding_not_an_exception():
    f = _analyzer("this is not json at all").analyze(_report())[0]

    assert f.status == "malformed"
    assert f.mechanism_description is None
    assert "not valid JSON" in f.rationale
    assert f.trace_id == "trace-1"


def test_truncated_json_becomes_a_malformed_finding():
    f = _analyzer('{"findings": [{"trace_id": "trace-1", "status": "obs')
    finding = f.analyze(_report())[0]
    assert finding.status == "malformed"


def test_empty_response_body_becomes_malformed():
    f = _analyzer("").analyze(_report())[0]
    assert f.status == "malformed"
    assert "empty response" in f.rationale


def test_json_wrapped_in_a_markdown_fence_is_parsed():
    body = "Here you go:\n```json\n" + json.dumps(_observed_payload()) + "\n```"
    f = _analyzer(body).analyze(_report())[0]
    assert f.status == "observed"


def test_prose_around_a_bare_json_object_is_tolerated():
    body = "Analysis follows. " + json.dumps(_observed_payload()) + " Done."
    f = _analyzer(body).analyze(_report())[0]
    assert f.status == "observed"


def test_json_of_the_wrong_shape_becomes_malformed():
    f = _analyzer("[1, 2, 3]").analyze(_report())[0]
    assert f.status == "malformed"


def test_unrecognized_status_becomes_malformed():
    payload = {
        "findings": [
            {"trace_id": "trace-1", "status": "kinda_bad", "rationale": "x"}
        ]
    }
    f = _analyzer(payload).analyze(_report())[0]
    assert f.status == "malformed"
    assert "kinda_bad" in f.rationale


def test_observed_without_a_mechanism_becomes_malformed():
    payload = _observed_payload()
    del payload["findings"][0]["mechanism"]
    f = _analyzer(payload).analyze(_report())[0]
    assert f.status == "malformed"
    assert "no mechanism" in f.rationale


def test_a_response_carrying_no_finding_for_a_trace_is_reported_per_trace():
    """Multi-trace reports must not silently reuse another trace's finding."""
    payload = _observed_payload()  # only trace-1
    findings = _analyzer(payload).analyze(_report(_trace("trace-1"), _trace("trace-2")))

    assert [f.trace_id for f in findings] == ["trace-1", "trace-2"]
    assert findings[0].status == "observed"
    assert findings[1].status == "malformed"
    assert "no finding for this trace" in findings[1].rationale


# ---------------------------------------------------------------------- #
# Out-of-range and non-numeric scalars
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("severity", [7, -0.5, 1.0001, "high", True, float("nan")])
def test_out_of_range_or_non_numeric_severity_becomes_malformed(severity):
    payload = _observed_payload(severity=severity)
    f = _analyzer(payload).analyze(_report())[0]

    assert f.status == "malformed"
    assert "severity" in f.rationale


def test_out_of_range_confidence_becomes_malformed():
    f = _analyzer(_observed_payload(confidence=1.5)).analyze(_report())[0]
    assert f.status == "malformed"
    assert "confidence" in f.rationale


def test_numeric_strings_within_range_are_accepted():
    f = _analyzer(_observed_payload(severity="0.4")).analyze(_report())[0]
    assert f.status == "observed"
    assert f.severity == 0.4


def test_missing_severity_downgrades_observed_to_uncertain():
    payload = _observed_payload()
    del payload["findings"][0]["severity"]
    f = _analyzer(payload).analyze(_report())[0]

    assert f.status == "uncertain"
    assert "missing severity" in f.rationale
    # The mechanism is retained: it is still information, just not promoted.
    assert f.mechanism_description == GOOD_MECHANISM
    assert f.mechanism_cluster_id is None


def test_out_of_range_blame_weight_drops_only_that_node():
    payload = _observed_payload(
        blamed_actors=[
            {"actor_id": "planner", "blame": 0.8},
            {"actor_id": "api_agent", "blame": 4.0},
        ],
        causal_links=[],
    )
    f = _analyzer(payload).analyze(_report())[0]

    assert f.status == "observed"
    assert [n.actor_id for n in f.blame_graph.nodes] == ["planner"]
    assert "invalid weight" in f.rationale


# ---------------------------------------------------------------------- #
# Hallucination filtering
# ---------------------------------------------------------------------- #
def test_hallucinated_actor_is_dropped_and_recorded():
    payload = _observed_payload(
        blamed_actors=[
            {"actor_id": "planner", "blame": 0.6},
            {"actor_id": "ghost_agent", "blame": 0.9},
        ],
        causal_links=[],
    )
    f = _analyzer(payload).analyze(_report())[0]

    actors = [n.actor_id for n in f.blame_graph.nodes]
    assert actors == ["planner"]
    assert "ghost_agent" in f.rationale
    assert "absent from the trace evidence" in f.rationale


def test_hallucinated_artifact_is_dropped_so_the_finding_stays_valid():
    """The strict CausalFinding validator would reject an unbacked artifact.

    Constructing an ``observed`` finding whose blame node names an artifact that
    is not in ``evidence_refs`` raises ValidationError, so an ungrounded artifact
    must be filtered before construction -- not passed through and crashed on.
    """
    payload = _observed_payload(
        blamed_actors=[
            {
                "actor_id": "planner",
                "blame": 0.9,
                "artifacts": ["plan", "skills/imaginary_skill.md"],
            }
        ],
        causal_links=[],
    )
    f = _analyzer(payload).analyze(_report())[0]

    assert f.status == "observed"
    assert f.blame_graph.artifacts_for("planner") == ("plan",)
    assert "skills/imaginary_skill.md" in f.rationale
    # The invariant the validator enforces, asserted directly.
    referenced = {a for n in f.blame_graph.nodes for a in n.artifacts}
    assert referenced <= set(f.evidence_refs)


def test_hallucinated_event_ids_are_dropped():
    payload = _observed_payload(evidence_event_ids=["e1", "e99"])
    f = _analyzer(payload).analyze(_report())[0]

    assert "trace-1#e1" in f.evidence_refs
    assert "trace-1#e99" not in f.evidence_refs
    assert "e99" in f.rationale


def test_no_real_event_ids_downgrades_observed_to_uncertain():
    payload = _observed_payload(evidence_event_ids=["e77", "e88"])
    f = _analyzer(payload).analyze(_report())[0]

    assert f.status == "uncertain"
    assert "not trace-backed" in f.rationale
    assert f.mechanism_cluster_id is None


def test_causal_link_to_an_ungrounded_actor_is_dropped():
    payload = _observed_payload(
        blamed_actors=[{"actor_id": "planner", "blame": 0.9}],
        causal_links=[
            {
                "from_actor": "planner",
                "to_actor": "ghost_agent",
                "mechanism": "handed off",
            }
        ],
    )
    f = _analyzer(payload).analyze(_report())[0]

    assert f.status == "observed"
    assert f.blame_graph.edges == ()
    assert "dropped 1 causal link" in f.rationale


# ---------------------------------------------------------------------- #
# Anti-degeneracy gate
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mechanism",
    [
        "failed-to-match-task-1",
        "tool error",
        "planning failure",
        "the agent failed the task",
        "incorrect answer",
        "unknown",
        "N/A",
        "the output did not match the expected answer",
    ],
)
def test_degenerate_mechanisms_are_not_promoted_to_observed(mechanism):
    """This is exactly the pathology the module exists to eliminate."""
    f = _analyzer(_observed_payload(mechanism=mechanism)).analyze(_report())[0]

    assert f.status == "uncertain"
    assert f.mechanism_cluster_id is None
    assert "generic or a restatement" in f.rationale


def test_a_causal_sentence_that_starts_with_a_generic_phrase_is_still_observed():
    mechanism = (
        "the agent failed to call load_skill before answering, so the skill "
        "body was absent from its context window"
    )
    f = _analyzer(_observed_payload(mechanism=mechanism)).analyze(_report())[0]
    assert f.status == "observed"


# ---------------------------------------------------------------------- #
# Multi-trace reports and instance reuse
# ---------------------------------------------------------------------- #
def test_multiple_traces_yield_one_finding_each_in_evidence_order():
    payload = {
        "findings": [
            # Deliberately out of order in the response.
            {
                "trace_id": "trace-2",
                "status": "insufficient_evidence",
                "rationale": "second rollout had no tool payloads",
            },
            {
                "trace_id": "trace-1",
                "status": "observed",
                "mechanism": GOOD_MECHANISM,
                "severity": 0.5,
                "confidence": 0.5,
                "rationale": "see e1",
                "blamed_actors": [{"actor_id": "planner", "blame": 1.0}],
                "evidence_event_ids": ["e1"],
            },
        ]
    }
    report = _report(_trace("trace-1"), _trace("trace-2"))
    findings = _analyzer(payload).analyze(report)

    assert [f.trace_id for f in findings] == ["trace-1", "trace-2"]
    assert findings[0].status == "observed"
    assert findings[1].status == "insufficient_evidence"


def test_single_trace_report_accepts_a_finding_with_a_mislabeled_trace_id():
    payload = _observed_payload(trace_id="whatever-the-model-said")
    f = _analyzer(payload).analyze(_report())[0]

    assert f.status == "observed"
    assert f.trace_id == "trace-1"


def test_repeated_analyze_calls_on_one_instance_are_independent():
    analyzer = _analyzer(_observed_payload())

    first = analyzer.analyze(_report(_trace("trace-1")))
    second = analyzer.analyze(_report(_trace("trace-1")))
    third = analyzer.analyze(_report(_trace("trace-1")))

    assert len(first) == len(second) == len(third) == 1
    assert (
        first[0].mechanism_description
        == second[0].mechanism_description
        == third[0].mechanism_description
        == GOOD_MECHANISM
    )
    assert first[0].evidence_refs == third[0].evidence_refs


def test_an_empty_report_produces_no_findings_and_no_model_call():
    calls: list[dict] = []
    analyzer = _analyzer(_observed_payload(), calls=calls)
    empty = RolloutGroupReport(
        candidate_id="cand-1",
        task_id="task-1",
        trace_refs=(),
        rollout_ids=(),
        sanitized_evidence=(),
    )
    assert analyzer.analyze(empty) == ()
    assert calls == []


# ---------------------------------------------------------------------- #
# Request construction / configuration
# ---------------------------------------------------------------------- #
def test_request_carries_the_configured_model_and_connection_settings():
    calls: list[dict] = []
    CugaTrajectoryAnalyzer(
        completion_fn=_fake(_observed_payload(), calls=calls),
        model="test/model",
        base_url="https://example.invalid/v1",
        api_key="secret",
        temperature=0.0,
    ).analyze(_report())

    request = calls[0]
    assert request["model"] == "test/model"
    assert request["api_base"] == "https://example.invalid/v1"
    assert request["api_key"] == "secret"
    assert request["temperature"] == 0.0
    assert [m["role"] for m in request["messages"]] == ["system", "user"]

def test_temperature_is_omitted_by_default_and_forwarded_when_set():
    """Pinning temperature=0 broke a live model, so the default sends nothing.

    ``azure/gpt-5.6-luna`` rejects the whole request with "Unsupported value:
    'temperature' does not support 0.0 with this model". Omitting the parameter
    is the only default that works across providers.
    """
    calls: list[dict] = []
    _analyzer(_observed_payload(), calls=calls).analyze(_report())
    assert "temperature" not in calls[0]

    calls.clear()
    _analyzer(_observed_payload(), calls=calls, temperature=0.2).analyze(_report())
    assert calls[0]["temperature"] == 0.2


def test_the_system_prompt_forbids_generic_mechanisms():
    """The anti-degeneracy instruction is load-bearing, so pin it."""
    calls: list[dict] = []
    _analyzer(_observed_payload(), calls=calls).analyze(_report())
    system = calls[0]["messages"][0]["content"]

    assert "FORBIDDEN" in system
    assert "category labels" in system
    assert "causal sentence" in system


def test_missing_model_configuration_raises_rather_than_guessing(monkeypatch):
    for var in (
        "CUGA_MODEL",
        "LITELLM_MODEL",
        "CUGA_BASE_URL",
        "LITELLM_BASE_URL",
        "CUGA_API_KEY",
        "LITELLM_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    analyzer = CugaTrajectoryAnalyzer(completion_fn=_fake(_observed_payload()))
    with pytest.raises(AnalyzerConfigurationError):
        analyzer.analyze(_report())


def test_a_transport_failure_propagates_instead_of_faking_a_verdict():
    """A model that could not be reached says nothing about the trajectory."""

    def boom(**_request):
        raise RuntimeError("connection reset")

    analyzer = CugaTrajectoryAnalyzer(completion_fn=boom, model="test/model")
    with pytest.raises(RuntimeError, match="connection reset"):
        analyzer.analyze(_report())


def test_events_are_trimmed_in_the_prompt_with_truncation_flagged():
    calls: list[dict] = []
    analyzer = _analyzer(_observed_payload(), calls=calls, max_events_in_prompt=1)
    analyzer.analyze(_report())

    user = calls[0]["messages"][1]["content"]
    payload = json.loads(user.split("SANITIZED EVIDENCE (JSON):\n")[1].split("\n\nReturn")[0])
    assert len(payload[0]["events"]) == 1
    assert payload[0]["events_truncated"] is True


def test_json_object_response_format_is_opt_in():
    calls: list[dict] = []
    _analyzer(_observed_payload(), calls=calls).analyze(_report())
    assert "response_format" not in calls[0]

    calls.clear()
    _analyzer(
        _observed_payload(), calls=calls, request_json_object=True
    ).analyze(_report())
    assert calls[0]["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------- #
# Composition with ParallelAnalysisRunner
# ---------------------------------------------------------------------- #
def test_factory_returns_a_zero_arg_builder():
    build = CugaTrajectoryAnalyzer.factory(
        completion_fn=_fake(_observed_payload()), model="test/model"
    )
    a, b = build(), build()
    assert isinstance(a, CugaTrajectoryAnalyzer)
    assert a is not b  # one analyzer per worker thread, not a shared instance


def test_composes_with_parallel_analysis_runner_preserving_order():
    threads: set[int] = set()
    lock = threading.Lock()

    def completion_fn(**request):
        with lock:
            threads.add(threading.get_ident())
        # Echo back the trace id the prompt asked about so a mixed-up mapping
        # would show as a malformed finding.
        user = request["messages"][1]["content"]
        trace_id = user.split("traces to analyze (one finding each): ['")[1].split(
            "'"
        )[0]
        body = json.dumps(
            {
                "findings": [
                    {
                        "trace_id": trace_id,
                        "status": "observed",
                        "mechanism": (
                            f"the api_agent called search_flights with the "
                            f"destination in the origin field on {trace_id}, so "
                            f"every itinerary departed from the wrong city"
                        ),
                        "severity": 0.7,
                        "confidence": 0.6,
                        "rationale": f"see e2 in {trace_id}",
                        "blamed_actors": [{"actor_id": "api_agent", "blame": 0.9}],
                        "evidence_event_ids": ["e2"],
                    }
                ]
            }
        )
        return {"choices": [{"message": {"content": body}}]}

    reports = [
        _report(_trace(f"trace-{i}", task_id=f"task-{i}"), task=_task(f"task-{i}"))
        for i in range(12)
    ]
    runner = ParallelAnalysisRunner(
        analyzer_factory=CugaTrajectoryAnalyzer.factory(
            completion_fn=completion_fn, model="test/model"
        ),
        max_workers=4,
    )
    outcomes = runner.run(reports)

    assert len(outcomes) == 12
    assert all(o.ok for o in outcomes), [o.error for o in outcomes if not o.ok]
    # Input order out, not completion order.
    assert [o.report.task_id for o in outcomes] == [f"task-{i}" for i in range(12)]
    findings = ParallelAnalysisRunner.flatten(outcomes)
    assert len(findings) == 12
    assert all(f.status == "observed" for f in findings)
    assert [f.trace_id for f in findings] == [f"trace-{i}" for i in range(12)]
    # Mechanisms differ per trace, which is what the placeholder could not do.
    assert len({f.mechanism_description for f in findings}) == 12
    assert len(threads) > 1  # actually ran concurrently


def test_runner_records_a_transport_failure_as_a_failed_outcome():
    def boom(**_request):
        raise RuntimeError("provider 503")

    runner = ParallelAnalysisRunner(
        analyzer_factory=CugaTrajectoryAnalyzer.factory(
            completion_fn=boom, model="test/model"
        ),
        max_workers=2,
    )
    outcomes = runner.run([_report(), _report(_trace("trace-2"))])

    assert [o.ok for o in outcomes] == [False, False]
    assert all("provider 503" in o.error for o in outcomes)
    assert ParallelAnalysisRunner.flatten(outcomes) == ()
