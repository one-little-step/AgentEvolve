"""Phase 1 (?14): persist tool invocations verbatim at the graph-callback layer.

RQ2 finding: LangChain hands ``on_tool_start`` an ``inputs`` mapping and
``on_tool_end`` the raw ``output`` -- both were dropped on the floor. Without
them a trace cannot replay tool calls, so external tools (web_search etc.)
can never be taped (design R3).

Contract under test:
* start -> tool_name + args_ref resolving to the VERBATIM inputs;
* end   -> output_ref resolving to the VERBATIM output;
* args fall back to input_str when the inputs kwarg is absent;
* no payload store -> refs are None but events still exist (honest absence);
* exotic output objects never crash capture (_json_safe markers);
* capability reports captured when graph-layer observations exist even
  without the SDK post-hoc recorder.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.cuga_wrapper import (  # noqa: E402
    GraphEventCollector,
    PayloadStore,
    _collector_tool_observations_captured,
    build_graph_callback_handler,
)


def _handler(store: PayloadStore | None = None):
    collector = GraphEventCollector(max_events=1000, payload_store=store)
    return build_graph_callback_handler(collector), collector


def _blob(store: PayloadStore, ref: str | None):
    assert ref is not None, "expected a persisted payload reference"
    return json.loads(store.blobs[ref])


import json  # noqa: E402


def test_on_tool_start_records_name_and_verbatim_args() -> None:
    handler, collector = _handler(PayloadStore())
    inputs = {"query": "meeting notes platform finance", "scope": "agent"}
    handler.on_tool_start(
        serialized={"name": "knowledge_search_knowledge"},
        input_str="meeting notes platform finance",
        run_id="r1",
        parent_run_id="p1",
        inputs=inputs,
    )
    event = next(e for e in collector.events if e["kind"] == "graph_tool_start")
    assert event["tool_name"] == "knowledge_search_knowledge"
    store = collector.payload_store
    assert _blob(store, event["args_ref"]) == inputs


def test_args_fall_back_to_input_str_without_inputs_kwarg() -> None:
    handler, collector = _handler(PayloadStore())
    handler.on_tool_start(
        serialized={"name": "calculator"},
        input_str="40000 + 25000",
        run_id="r2",
        parent_run_id="p1",
    )
    event = next(e for e in collector.events if e["kind"] == "graph_tool_start")
    assert _blob(collector.payload_store, event["args_ref"]) == (
        "40000 + 25000"
    )


def test_on_tool_end_records_verbatim_output() -> None:
    handler, collector = _handler(PayloadStore())
    output = {"results": [{"text": "Program sync - raw notes", "score": 0.59}]}
    handler.on_tool_start(
        serialized={"name": "knowledge_search_knowledge"},
        input_str="q",
        run_id="r3",
        parent_run_id="p1",
        inputs={"query": "q"},
    )
    handler.on_tool_end(output=output, run_id="r3", parent_run_id="p1")
    end = next(e for e in collector.events if e["kind"] == "graph_tool_end")
    assert _blob(collector.payload_store, end["output_ref"]) == output


def test_no_payload_store_yields_none_refs_but_keeps_events() -> None:
    handler, collector = _handler(None)
    handler.on_tool_start(
        serialized={"name": "web_search"},
        input_str="q",
        run_id="r4",
        parent_run_id="p",
        inputs={"q": "q"},
    )
    handler.on_tool_end(output="irrelevant", run_id="r4", parent_run_id="p")
    start = next(e for e in collector.events if e["kind"] == "graph_tool_start")
    end = next(e for e in collector.events if e["kind"] == "graph_tool_end")
    # record() omits None-valued fields: honest absence is a missing key,
    # never a blank string.
    assert "args_ref" not in start
    assert "output_ref" not in end


def test_exotic_output_does_not_crash_capture() -> None:
    handler, collector = _handler(PayloadStore())

    class Unserializable:
        pass

    handler.on_tool_end(
        output={"weird": {Unserializable}}, run_id="r5", parent_run_id="p"
    )
    end = next(e for e in collector.events if e["kind"] == "graph_tool_end")
    # _json_safe reduces unrepresentable values to typed markers instead of
    # raising, so the event must exist with SOME ref or None -- never crash.
    assert "output_ref" in end


def test_capability_captures_graph_layer_without_recorder() -> None:
    handler, collector = _handler(PayloadStore())
    assert _collector_tool_observations_captured(collector) is False
    handler.on_tool_start(
        serialized={"name": "calculator"},
        input_str="1+1",
        run_id="r6",
        parent_run_id="p",
        inputs={"expression": "1+1"},
    )
    handler.on_tool_end(output="2", run_id="r6", parent_run_id="p")
    assert _collector_tool_observations_captured(collector) is True
