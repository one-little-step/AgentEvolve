"""Verify the CugaAdapter against the real live-CUGA reference trace.

Synthetic test fixtures only prove the adapter matches my own assumptions.
This exercises the mapping against the 56-event trace captured from a live
CUGA run, and checks that no payload blob content reaches the core trace.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "src")

from agent_evolve.adapters.cuga_adapter import CugaAdapter  # noqa: E402
from agent_evolve.core.analyzer import FakeAnalyzerJudge  # noqa: E402
from agent_evolve.core.contracts import EvolutionTask  # noqa: E402

TRACE = "data/traces/5d434903-bc26-4dc4-9229-8d886d2c6781"


class ReferenceWrapper:
    def __init__(self) -> None:
        self.harness: dict[str, object] = {}

    def run_task(self, task_id, harness_config):
        self.harness = dict(harness_config)
        return {
            "task_id": task_id,
            "status": "success",
            "final_output": "ALPHA-7924786034\nBETA-2779592008\n858",
            "events": [{"event_id": "thin", "kind": "run_started"}],
            "causal_trace_path": TRACE,
        }

    def get_artifacts(self) -> dict[str, str]:
        return {}

    def update_artifact(self, artifact_id: str, content: str) -> None:
        pass


def main() -> int:
    wrapper = ReferenceWrapper()
    adapter = CugaAdapter(wrapper)
    adapter.register_candidate(
        "base",
        {
            "skills/status-report": "Report status succinctly.",
            "policies/format": "Answer with the checksum only.",
            "instructions": "Be precise.",
        },
    )
    workspace = adapter.materialize_candidate("base", "attempt-1")
    task = EvolutionTask(
        task_id="complete-graph-demo",
        input_text="chain the tools",
        expected_contract={"expected_substring": "NEVER_MATCHES_SO_WE_SEE_BLAME"},
    )
    result = adapter.run_full_rollout(workspace, task, "rollout-1")
    trace = adapter.capture_trace(result)

    print("--- harness delivered to CUGA ---")
    for key in sorted(wrapper.harness):
        value = wrapper.harness[key]
        print(f"  {key}: {value}")

    print("\n--- trace mapping (real 56-event live trace) ---")
    raw = json.loads(open(f"{TRACE}/causal-trace.json").read())
    print("  source events:", len(raw["events"]))
    print("  mapped events:", len(trace.events))
    print("  with actor_id:", sum(1 for e in trace.events if e.actor_id))
    print("  with parent_event_id:", sum(1 for e in trace.events if e.parent_event_id))
    actors = sorted({e.actor_id for e in trace.events if e.actor_id})
    print("  distinct actors:", len(actors))
    print("  actors:", actors)

    analysis = FakeAnalyzerJudge().analyze(task, trace)
    nodes = analysis.blame_graph.nodes
    print("\n--- causal blame (H2) ---")
    print("  blame nodes:", len(nodes))
    print("  blame mass:", round(sum(n.blame for n in nodes), 6))
    print("  synthetic 'unknown' placeholder:", any(n.actor_id == "unknown" for n in nodes))
    print("  top blamed:", max(nodes, key=lambda n: n.blame).actor_id if nodes else None)

    print("\n--- payload leak audit ---")
    serialized = json.dumps([dict(e.payload) for e in trace.events])
    print("  serialized payload bytes:", len(serialized))
    longest = 0
    for event in trace.events:
        for value in dict(event.payload).values():
            longest = max(longest, len(str(value)))
    print("  longest inline payload string:", longest)

    blob_dir = f"{TRACE}/payloads"
    leaked = []
    for name in os.listdir(blob_dir):
        content = open(f"{blob_dir}/{name}").read()
        # Sample distinctive long substrings from each blob body.
        for probe in (content[200:260], content[1000:1060]):
            if len(probe) >= 40 and probe in serialized:
                leaked.append((name, probe[:40]))
    print("  blob bodies leaked into core trace:", len(leaked))
    for name, probe in leaked[:3]:
        print("    LEAK:", name, repr(probe))

    # The real invariant: every *_ref value stays a bare 64-char hash, and the
    # only oversized payloads are tool observations (environment evidence,
    # not model prompts or agent state).
    ref_keys = {"state_before_ref", "state_after_ref", "messages_ref", "response_ref"}
    bad_refs = []
    oversized_non_tool = []
    for event in trace.events:
        for key, value in dict(event.payload).items():
            text = value if isinstance(value, str) else json.dumps(value)
            if key in ref_keys and len(text) != 64:
                bad_refs.append((event.event_id, key, len(text)))
            if key not in ref_keys and len(text) > 64 and key != "tool_call":
                oversized_non_tool.append((event.event_id, key, len(text)))
    print("  refs that are not bare 64-char hashes:", len(bad_refs))
    print("  oversized non-tool_call payloads:", len(oversized_non_tool))
    for item in oversized_non_tool[:3]:
        print("    OVERSIZED:", item)

    ok = (
        len(trace.events) == len(raw["events"])
        and sum(1 for e in trace.events if e.actor_id) > 0
        and sum(1 for e in trace.events if e.parent_event_id) > 0
        and len(nodes) > 0
        and not any(n.actor_id == "unknown" for n in nodes)
        and len(leaked) == 0
        and not bad_refs
        and not oversized_non_tool
        and wrapper.harness.get("skills") == {"status-report": "Report status succinctly."}
        and wrapper.harness.get("policies") == {"format": "Answer with the checksum only."}
        and wrapper.harness.get("instructions") == "Be precise."
    )
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
