"""Collect the vanilla CUGA B0 trajectory corpus.

Edit the configuration block below to select the baseline tasks. This script has
no command-line arguments so each experimental run is explicit in the file.
Credentials remain in `.env` and never enter the written trajectory JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from agent_evolve.cuga_wrapper import CugaWrapper, InMemoryRuntime, RuntimeSettings


# Baseline configuration. Replace the example tasks with the fixed B0 task set.
USE_MOCK_RUNTIME = False
OUTPUT_DIRECTORY = Path("data/historical_trajectories")
TASKS = (
    {
        "task_id": "b0-001",
        "input": "Summarize the task and provide a direct answer.",
    },
    {
        "task_id": "b0-002",
        "input": "Identify the main constraint and explain the next action.",
    },
    {
        "task_id": "b0-003",
        "input": "Give a concise, evidence-based response to the task.",
    },
    {
        "task_id": "b0-004",
        "input": "State the result and the reasoning needed to support it.",
    },
    {
        "task_id": "b0-005",
        "input": "Resolve the request using only the information available.",
    },
)


def build_wrapper() -> CugaWrapper:
    if USE_MOCK_RUNTIME:
        return CugaWrapper(InMemoryRuntime(), RuntimeSettings(model="mock-model"))
    load_dotenv()
    return CugaWrapper.from_cuga(RuntimeSettings.from_env())


def main() -> int:
    wrapper = build_wrapper()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        task_id = str(task["task_id"])
        trace = wrapper.run_task(task_id, {"input": str(task["input"])})
        output_path = OUTPUT_DIRECTORY / f"{task_id}.json"
        output_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
