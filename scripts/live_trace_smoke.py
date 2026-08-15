"""Run a multistep task through the real CUGA wrapper and emit an inspectable trace.

Tracing is enabled; a task that requires multiple tool invocations is used so
the produced trace contains ordered tool-call events. Model output and trace
paths are printed, but environment values and credentials are never printed.
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from agent_evolve.core.trace import PayloadLevel
from agent_evolve.cuga_wrapper import CugaWrapper, RuntimeSettings, TraceConfig

load_dotenv()

TASK_ID = "multistep-smoke-001"
TASK_INPUT = (
    "Using the calculator tool, compute 1234 * 5678. Then using the calculator "
    "tool again, add 900 to that first result. Report both the product and the "
    "final sum as two separate lines."
)
INSTRUCTIONS = (
    "Use the available calculator tool for every arithmetic step. Do not do "
    "mental arithmetic; invoke the calculator tool once per arithmetic operation."
)

config = TraceConfig(
    enabled=True,
    output_root=Path("data/traces"),
    payload_level=PayloadLevel.CAUSAL_SUFFICIENT,
    max_observation_bytes=1_048_576,
)

wrapper = CugaWrapper.from_cuga(RuntimeSettings.from_env(), trace_config=config)
result = wrapper.run_task(TASK_ID, {"input": TASK_INPUT, "instructions": INSTRUCTIONS})

report = {
    "task_id": TASK_ID,
    "task_input": TASK_INPUT,
    "status": result.get("status"),
    "causal_trace_path": result.get("causal_trace_path"),
    "final_output": result.get("final_output"),
    "events": result.get("events"),
}
print("=== WRAPPER RESULT ===")
print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))

causal_trace_path = result.get("causal_trace_path")
if causal_trace_path:
    trace_dir = Path(causal_trace_path)
    print("=== MANIFEST ===")
    print((trace_dir / "manifest.json").read_text())
    print("=== EVENTS (events.jsonl) ===")
    events_path = trace_dir / "events.jsonl"
    print(events_path.read_text() if events_path.exists() else "(no events.jsonl written)")
    print("=== CAUSAL-TRACE EXPORT (causal-trace.json) ===")
    print((trace_dir / "causal-trace.json").read_text())
