"""Run one task and write its raw JSON-serializable trajectory.

Edit the configuration constants below for each experiment. Environment values
are loaded only for a live CUGA run and are never written to a trajectory.
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from cuga import tracked_tool
from langchain_core.tools import tool

from agent_evolve.cuga_wrapper import CugaWrapper, MockHarnessRuntime, RuntimeSettings

# Experiment configuration. Change these values directly; this runner accepts no CLI arguments.
USE_MOCK_RUNTIME = False
TASK_ID = "baseline-demo-001"
TASK_INPUT = "what tools and skills available to you now?"
OUTPUT_PATH = Path("data/historical_trajectories/baseline-demo-001.json")


@tool
@tracked_tool(app_name="agent-evolve")
def multiply(left: int, right: int) -> int:
    """Multiply two integers when an arithmetic task needs it."""
    return left * right


# Harness configuration. Real CUGA supports `instructions` and LangChain tool
# objects today. Skills, memory, and policies run deterministically with the mock
# runtime until their CUGA APIs are individually verified.
HARNESS = {
    "version": "b0-v1",
    "instructions": "Use the available calculator tool for arithmetic. Answer directly and concisely.",
    "skills": {"arithmetic": "Use tools instead of mental arithmetic when a tool is available."},
    "memory": {"baseline_note": "This is the initial B0 harness."},
    "tools": [multiply],
    "policies": {},
}


def main() -> int:
    if USE_MOCK_RUNTIME:
        wrapper = CugaWrapper(MockHarnessRuntime(), RuntimeSettings(model="mock-model"))
    else:
        load_dotenv()
        wrapper = CugaWrapper.from_cuga(RuntimeSettings.from_env())
    trace = wrapper.run_task(TASK_ID, {**HARNESS, "input": TASK_INPUT})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
