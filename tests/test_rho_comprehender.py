"""Tests for RHO trajectory comprehension (Interface A).

Every test injects a fake ``completion_fn``; no test makes a network call.

Two properties matter more than schema plumbing here:

1. The summary text that gets embedded must carry no UUIDs, hex hashes, or JSON
   keys, because those dominate a raw trace and destroy DPP diversity.
2. The *prompt* must carry behaviour, not identifiers. Measured on the real
   corpus, a CUGA event payload is
   ``{run_id, parent_run_id, state_before_ref, response_ref, step}`` -- dumping
   raw events reproduces exactly the 60.8%-identifier problem this stage exists
   to remove. The semantic content lives in ``payload.node``,
   ``payload.routed_to``, ``tool_call`` payloads and ``tool_observations``.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

from agent_evolve.adapters.cuga_rho_comprehender import (
    OUTCOME_VALUES,
    RhoComprehender,
    TrajectorySummary,
    _collapse,
)
from agent_evolve.core.rho.cache import JsonDiskCache
from agent_evolve.core.rho.history import HistoricalRecord

UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)
LONG_HEX_RE = re.compile(r"\b[0-9a-f]{16,}\b", re.I)

# Values that must never reach the prompt: they are the identifier noise the
# comprehension stage exists to strip.
RUN_ID = "01a0092c-60dd-7be2-a57a-fedd06a73a94"
STATE_REF = "934a6c823f7b35051b4c0625449944b98d48480d5e00f98990120b37a434cf32"
RESPONSE_REF = "75371a34fdfefc04e789461750cb2fe6a1a68d902282fb88f16a3a178bfb84f1"


# --------------------------------------------------------------------------- #
# Fixtures shaped like the real corpus
# --------------------------------------------------------------------------- #
def _record(task_id: str = "gaia-1") -> HistoricalRecord:
    """A minimal narration-only trajectory: two model turns, no tool ever ran."""
    return HistoricalRecord(
        task_id=task_id,
        input_text="how many albums were released in 1977",
        trace_path=f"/traces/{task_id}/causal-trace.json",
        raw_trace={
            "events": [
                {
                    "event_id": "3f57289b-1e31-4a36-913f-aedd501e44d1",
                    "kind": "llm_call_end",
                    "actor_id": "call_model",
                    "payload": {"text": "I should search for this."},
                }
            ]
        },
        final_output="I was unable to determine the answer.",
        tool_observation_count=0,
        harness_version="vanilla",
        content_hash="sha256:abc123",
    )


def _graph_events(nodes: list[str]) -> list[dict]:
    """graph_node_start/end pairs whose payloads are identifier-only, as real."""
    events: list[dict] = []
    for index, node in enumerate(nodes):
        events.append(
            {
                "event_id": f"graph:{index * 2}",
                "kind": "graph_node_start",
                "actor_id": None,
                "sequence": index * 2,
                "payload": {
                    "node": node,
                    "run_id": RUN_ID,
                    "parent_run_id": RUN_ID,
                    "state_before_ref": STATE_REF,
                    "step": index + 1,
                },
            }
        )
        events.append(
            {
                "event_id": f"graph:{index * 2 + 1}",
                "kind": "graph_node_end",
                "actor_id": None,
                "sequence": index * 2 + 1,
                "payload": {
                    "node": node,
                    "run_id": RUN_ID,
                    "state_after_ref": STATE_REF,
                    "routed_to": nodes[index + 1] if index + 1 < len(nodes) else None,
                },
            }
        )
    return events


def _tool_observation(sequence: int, tool: str, args: dict, result: str) -> dict:
    return {
        "sequence": sequence,
        "tool_name": tool,
        "canonical_arguments": json.dumps(args, sort_keys=True),
        "result": result,
        "error": None,
        "duration_ms": 1529.16,
        "original_bytes": len(result),
        "retained_bytes": len(result),
        "truncated": False,
        "withheld_reason": None,
        "content_digest": f"sha256:{STATE_REF}",
        "replay_eligible": True,
    }


def _tool_record(observations: int = 3) -> HistoricalRecord:
    """A trajectory that really executed tools, shaped like the rich real trace."""
    nodes = ["prepare", "call_model", "sandbox", "call_model", "sandbox", "call_model"]
    events = _graph_events(nodes)
    events.append(
        {
            "event_id": "graph:99",
            "kind": "llm_call_end",
            "actor_id": None,
            "sequence": 99,
            "payload": {"response_ref": RESPONSE_REF, "run_id": RUN_ID},
        }
    )
    obs = [
        _tool_observation(
            i,
            "web_search" if i % 2 == 0 else "web_fetch",
            {"query": f"searched phrasing number {i}", "max_results": 5},
            f"result body number {i} with plain prose only",
        )
        for i in range(observations)
    ]
    return HistoricalRecord(
        task_id="gaia-tools",
        input_text="what was the volume of the bag",
        trace_path="/traces/gaia-tools/causal-trace.json",
        raw_trace={
            "events": events,
            "tool_observations": obs,
            "status": "success",
            "error": None,
            "events_truncated": False,
            "dropped_event_count": 0,
            "capabilities": {
                "tool_observations": {"status": "captured", "reason": None},
                "graph_history": {
                    "status": "unavailable_no_checkpointer",
                    "reason": "no verified active checkpointer",
                },
            },
        },
        final_output="The calculated volume was 0.1777 cubic metres.",
        tool_observation_count=observations,
        harness_version="vanilla",
        content_hash="sha256:tools",
    )


def _response(payload: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def _good_payload() -> dict:
    return {
        "what_was_attempted": "count albums released in a given year",
        "approach_taken": "narrated a search plan in prose",
        "where_it_went_wrong": (
            "never emitted an executable code block, so no tool ran"
        ),
        "tools_used": [],
        "outcome": "no_committed_answer",
    }


def _recording_fake(payload: dict | None = None):
    """Return ``(fake, calls)`` where ``calls`` collects each request kwargs."""
    calls: list[dict] = []

    def fake(**request: object) -> dict:
        calls.append(request)
        return _response(payload if payload is not None else _good_payload())

    return fake, calls


def _prompt(calls: list[dict]) -> str:
    messages = calls[0]["messages"]
    assert isinstance(messages, list)
    return "\n".join(str(m["content"]) for m in messages)


def _user_prompt(calls: list[dict]) -> str:
    messages = calls[0]["messages"]
    return str(next(m for m in messages if m["role"] == "user")["content"])


def _system_prompt(calls: list[dict]) -> str:
    messages = calls[0]["messages"]
    return str(next(m for m in messages if m["role"] == "system")["content"])


# --------------------------------------------------------------------------- #
# Plan-specified behaviour
# --------------------------------------------------------------------------- #
def test_parses_a_valid_summary() -> None:
    fake, calls = _recording_fake()

    summary = RhoComprehender(completion_fn=fake, model="m").comprehend(_record())

    assert isinstance(summary, TrajectorySummary)
    assert summary.observed is True
    assert summary.outcome == "no_committed_answer"
    assert "executable code block" in summary.where_it_went_wrong
    assert summary.task_id == "gaia-1"
    assert not summary.error
    assert len(calls) == 1


def test_embedding_text_carries_no_identifiers() -> None:
    fake, _ = _recording_fake()

    summary = RhoComprehender(completion_fn=fake, model="m").comprehend(_record())

    text = summary.embedding_text
    assert UUID_RE.search(text) is None
    assert LONG_HEX_RE.search(text) is None
    assert '":' not in text
    assert "event_id" not in text


def test_malformed_response_yields_an_unobserved_summary() -> None:
    def fake(**request: object) -> dict:
        return {"choices": [{"message": {"content": "not json at all"}}]}

    summary = RhoComprehender(completion_fn=fake, model="m").comprehend(_record())

    assert summary.observed is False
    assert summary.error


def test_transport_failure_yields_an_unobserved_summary() -> None:
    def fake(**request: object) -> dict:
        raise RuntimeError("endpoint down")

    summary = RhoComprehender(completion_fn=fake, model="m").comprehend(_record())

    assert summary.observed is False
    assert "endpoint down" in summary.error


def test_temperature_is_never_zero_and_omitted_by_default() -> None:
    fake, calls = _recording_fake()

    RhoComprehender(completion_fn=fake, model="m").comprehend(_record())

    assert "temperature" not in calls[0]


def test_cache_hit_skips_the_model_call(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path)
    fake, calls = _recording_fake()

    comprehender = RhoComprehender(completion_fn=fake, model="m", cache=cache)
    first = comprehender.comprehend(_record())
    second = comprehender.comprehend(_record())

    assert len(calls) == 1
    assert second.where_it_went_wrong == first.where_it_went_wrong
    assert second.observed is True
    assert cache.hits == 1


def test_changed_content_hash_misses_the_cache(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path)
    fake, calls = _recording_fake()

    comprehender = RhoComprehender(completion_fn=fake, model="m", cache=cache)
    comprehender.comprehend(_record())

    # NOTE: HistoricalRecord uses slots=True, so it has no ``__dict__``;
    # dataclasses.replace is the only way to derive a variant.
    changed = dataclasses.replace(_record(), content_hash="sha256:different")
    comprehender.comprehend(changed)

    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# Failure observability
# --------------------------------------------------------------------------- #
def test_zero_temperature_is_rejected_rather_than_sent() -> None:
    fake, calls = _recording_fake()

    summary = RhoComprehender(
        completion_fn=fake, model="m", temperature=0.0
    ).comprehend(_record())

    assert calls == []
    assert summary.observed is False
    assert "temperature" in summary.error


def test_nonzero_temperature_is_forwarded() -> None:
    fake, calls = _recording_fake()

    RhoComprehender(completion_fn=fake, model="m", temperature=0.4).comprehend(
        _record()
    )

    assert calls[0]["temperature"] == 0.4


def test_empty_response_body_yields_an_unobserved_summary() -> None:
    def fake(**request: object) -> dict:
        return {"choices": []}

    summary = RhoComprehender(completion_fn=fake, model="m").comprehend(_record())

    assert summary.observed is False
    assert summary.error


def test_failure_is_not_cached(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path)
    attempts: list[int] = []

    def fake(**request: object) -> dict:
        attempts.append(1)
        raise RuntimeError("endpoint down")

    comprehender = RhoComprehender(completion_fn=fake, model="m", cache=cache)
    comprehender.comprehend(_record())
    comprehender.comprehend(_record())

    assert len(attempts) == 2
    assert cache.hits == 0


def test_fenced_json_response_is_parsed() -> None:
    body = "```json\n" + json.dumps(_good_payload()) + "\n```"

    def fake(**request: object) -> dict:
        return {"choices": [{"message": {"content": body}}]}

    summary = RhoComprehender(completion_fn=fake, model="m").comprehend(_record())

    assert summary.observed is True
    assert summary.outcome == "no_committed_answer"


def test_model_echoed_identifiers_are_scrubbed_from_the_summary() -> None:
    payload = _good_payload()
    payload["where_it_went_wrong"] = (
        f'the node {RUN_ID} wrote "state_after_ref": {STATE_REF} and stopped'
    )

    def fake(**request: object) -> dict:
        return _response(payload)

    summary = RhoComprehender(completion_fn=fake, model="m").comprehend(_record())

    text = summary.embedding_text
    assert UUID_RE.search(text) is None
    assert LONG_HEX_RE.search(text) is None
    assert '":' not in text
    assert "and stopped" in text


def test_outcome_vocabulary_is_published_for_downstream_stages() -> None:
    assert "no_committed_answer" in OUTCOME_VALUES
    assert "correct_answer" in OUTCOME_VALUES
    assert "wrong_answer" in OUTCOME_VALUES
    assert "error" in OUTCOME_VALUES


# --------------------------------------------------------------------------- #
# Prompt quality: the actual point of this stage
# --------------------------------------------------------------------------- #
def test_prompt_is_free_of_run_ids_and_state_refs() -> None:
    fake, calls = _recording_fake()

    RhoComprehender(completion_fn=fake, model="m").comprehend(_tool_record())

    prompt = _user_prompt(calls)
    assert RUN_ID not in prompt
    assert STATE_REF not in prompt
    assert RESPONSE_REF not in prompt
    assert "state_before_ref" not in prompt
    assert "parent_run_id" not in prompt


def test_prompt_carries_executed_tool_names_and_arguments() -> None:
    fake, calls = _recording_fake()

    RhoComprehender(completion_fn=fake, model="m").comprehend(_tool_record())

    prompt = _user_prompt(calls)
    assert "web_search" in prompt
    assert "web_fetch" in prompt
    assert "searched phrasing number 0" in prompt
    assert "result body number 0" in prompt


def test_prompt_states_no_tool_ran_when_none_did() -> None:
    fake, calls = _recording_fake()

    RhoComprehender(completion_fn=fake, model="m").comprehend(_record())

    prompt = _user_prompt(calls)
    assert "no tool ever executed" in prompt


def test_prompt_reports_collapsed_control_flow() -> None:
    fake, calls = _recording_fake()

    RhoComprehender(completion_fn=fake, model="m").comprehend(_tool_record())

    prompt = _user_prompt(calls)
    assert "CONTROL FLOW" in prompt
    assert "prepare" in prompt
    # call_model -> sandbox repeats, so the loop must be collapsed, not listed.
    assert "(call_model -> sandbox) x2" in prompt


def test_repeated_cycle_count_survives_collapse() -> None:
    """The repetition count is the diagnostic signal and must not be lost.

    A longest-cycle-first collapse renders the real 8x call_model/sandbox
    alternation as one 8-node cycle repeated twice, which reports "2" for a run
    that looped eight times. Shortest-cycle-first is the correct search order.
    """
    nodes = ["prepare"] + ["call_model", "sandbox"] * 8 + ["FinalAnswerAgent"]

    assert (
        _collapse(nodes)
        == "prepare -> (call_model -> sandbox) x8 -> FinalAnswerAgent"
    )


def test_collapse_drops_cuga_adjacent_duplicate_nodes() -> None:
    # CUGA emits the subgraph node name twice; that is capture shape, not a loop.
    assert _collapse(["prepare", "prepare", "call_model"]) == "prepare -> call_model"


def test_prompt_declares_capture_limits_so_absence_is_not_overread() -> None:
    fake, calls = _recording_fake()

    RhoComprehender(completion_fn=fake, model="m").comprehend(_tool_record())

    prompt = _user_prompt(calls)
    assert "CAPTURE LIMITS" in prompt
    assert "unavailable_no_checkpointer" in prompt


def test_prompt_marks_missing_task_text_explicitly() -> None:
    # 0 of 267 real traces carry input_text, so a blank must be labelled.
    fake, calls = _recording_fake()
    blank = dataclasses.replace(_record(), input_text="")

    RhoComprehender(completion_fn=fake, model="m").comprehend(blank)

    assert "not recorded" in _user_prompt(calls)


def test_prompt_bounds_the_number_of_tool_rows() -> None:
    fake, calls = _recording_fake()

    RhoComprehender(completion_fn=fake, model="m", max_events=2).comprehend(
        _tool_record(observations=5)
    )

    prompt = _user_prompt(calls)
    assert "3 further tool executions not shown" in prompt


def test_system_prompt_forbids_identifiers_and_names_the_output_keys() -> None:
    fake, calls = _recording_fake()

    RhoComprehender(completion_fn=fake, model="m").comprehend(_record())

    system = _system_prompt(calls)
    for key in (
        "what_was_attempted",
        "approach_taken",
        "where_it_went_wrong",
        "tools_used",
        "outcome",
    ):
        assert key in system
    lowered = system.lower()
    assert "uuid" in lowered
    assert "hash" in lowered
    # It must tell the model to judge tool use by execution, not by narration.
    assert "narrat" in lowered


def test_embedding_text_excludes_the_task_id() -> None:
    fake, _ = _recording_fake()

    summary = RhoComprehender(completion_fn=fake, model="m").comprehend(
        _record("gaia-distinctive-id")
    )

    assert "gaia-distinctive-id" not in summary.embedding_text


def test_final_output_reaches_the_prompt() -> None:
    fake, calls = _recording_fake()

    RhoComprehender(completion_fn=fake, model="m").comprehend(_record())

    assert "I was unable to determine the answer." in _user_prompt(calls)


def test_model_and_credentials_are_forwarded() -> None:
    fake, calls = _recording_fake()

    RhoComprehender(
        completion_fn=fake,
        model="azure/gpt",
        base_url="https://example.invalid",
        api_key="k",
    ).comprehend(_record())

    assert calls[0]["model"] == "azure/gpt"
    assert calls[0]["api_base"] == "https://example.invalid"
    assert calls[0]["api_key"] == "k"


def test_tools_used_is_coerced_to_a_tuple_of_strings() -> None:
    payload = _good_payload()
    payload["tools_used"] = ["web_search", 7]

    def fake(**request: object) -> dict:
        return _response(payload)

    summary = RhoComprehender(completion_fn=fake, model="m").comprehend(_record())

    assert summary.tools_used == ("web_search", "7")


def test_default_comprehender_does_not_cache(tmp_path: Path) -> None:
    fake, calls = _recording_fake()

    comprehender = RhoComprehender(completion_fn=fake, model="m")
    comprehender.comprehend(_record())
    comprehender.comprehend(_record())

    assert len(calls) == 2
