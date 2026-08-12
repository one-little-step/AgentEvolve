"""Run one task and write its raw JSON-serializable trajectory.

Edit the configuration constants below for each experiment. Environment values
are loaded only for a live CUGA run and are never written to a trajectory.
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from agent_evolve.cuga_wrapper import CugaWrapper, InMemoryRuntime, RuntimeSettings

# Experiment configuration. Change these values directly; this runner accepts no CLI arguments.
USE_MOCK_RUNTIME = False
TASK_ID = "baseline-demo-001"
TASK_INPUT = "Tell me a jike about a cat and a dog that are friends."
OUTPUT_PATH = Path("data/historical_trajectories/baseline-demo-001.json")


def main() -> int:
    if USE_MOCK_RUNTIME:
        wrapper = CugaWrapper(InMemoryRuntime(), RuntimeSettings(model="mock-model"))
    else:
        load_dotenv()
        wrapper = CugaWrapper.from_cuga(RuntimeSettings.from_env())
    trace = wrapper.run_task(TASK_ID, {"input": TASK_INPUT})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
