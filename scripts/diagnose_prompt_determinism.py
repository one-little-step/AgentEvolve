"""D4 confirmation: execution is a deterministic function of the prompt.

Established so far:
  * Tool construction is irrelevant (1/3 vs 1/3, byte-identical tool metadata) —
    `terminal_output/cuga-tracing/d4-tool-construction.log`
  * Probe vocabulary is not the controlling variable — the arm that scored 3/3 in
    `d4-vocabulary-ab.log` scored 0/3 in `d4-framing-isolation.log` with the same
    wording, differing only in task suffix.
  * Identical prompt hash always produced an identical outcome across both
    construction arms, which is expected: this reasoning model has temperature
    skipped, so decoding is effectively greedy.

Hypothesis under test: for this model, "did the agent invoke the tool" is a
deterministic function of the exact prompt text, NOT a random per-run event. If
so, "flaky tool execution" is a wrong description and repeated identical trials
add no information.

Method: run each of several task phrasings N times. If the hypothesis holds,
every phrasing scores either 0/N or N/N, never in between.

This is the correct way to sample this model: vary the PROMPT, not the trial
index. Ground truth remains the tool function body.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REPEATS = int(os.environ.get("D4_REPEATS", "2"))

BASE = (
    "Call read_build_number and report the exact build number it returns. "
    "The build number changes per run, so it must come from the tool."
)

PHRASINGS = {
    "bare": BASE,
    "respond_only_value": BASE + " Respond with only the value.",
    "return_just_value": BASE + " Return just the value, nothing else.",
    "imperative_first": (
        "First call read_build_number. Then report the exact value it returned, "
        "and nothing else."
    ),
    "explicit_code_instruction": (
        "Write and execute Python code that calls read_build_number(), then report "
        "the exact value it returned."
    ),
}


def build_tool(expected: str, executed: list[str]) -> list:
    from langchain_core.tools import tool

    from cuga import tracked_tool

    @tool
    @tracked_tool(app_name="probe")
    def read_build_number() -> str:
        """Read the current build number for this workspace."""
        executed.append("called")
        return expected

    return [read_build_number]


async def run_once(label: str, task: str, index: int) -> dict:
    from cuga.sdk import CugaAgent

    executed: list[str] = []
    expected = f"BUILD-{random.randint(10**11, 10**12 - 1)}"
    agent = CugaAgent(
        tools=build_tool(expected, executed),
        special_instructions="You are an autonomous agent. Use the available tools.",
        enable_knowledge=False,
        enable_skills=False,
    )
    try:
        result = await agent.invoke(
            task, thread_id=f"d4d-{label}-{index}", track_tool_calls=True
        )
        answer = str(result.answer or "")
        sdk_calls = len(result.tool_calls or [])
    finally:
        try:
            await agent.aclose()
        except Exception:  # noqa: BLE001
            pass

    return {
        "phrasing": label,
        "repeat": index,
        "task_hash": hashlib.sha256(task.encode()).hexdigest()[:10],
        "tool_body_ran": len(executed),
        "value_in_answer": expected in answer,
        "sdk_tool_calls": sdk_calls,
        "answer_head": answer[:140],
    }


async def main() -> None:
    from agent_evolve.cuga_wrapper import RuntimeSettings, prepare_cuga_environment, resolve_skills_root

    prepare_cuga_environment()
    RuntimeSettings.from_env().configure_cuga_environment()
    os.environ["SKILLS_ROOT"] = resolve_skills_root()

    results: list[dict] = []
    for label, task in PHRASINGS.items():
        for index in range(REPEATS):
            outcome = await run_once(label, task, index)
            print(json.dumps(outcome), flush=True)
            results.append(outcome)

    print("=== D4 PROMPT DETERMINISM (ground truth = tool body ran) ===")
    mixed = []
    for label in PHRASINGS:
        subset = [r for r in results if r["phrasing"] == label]
        ran = sum(1 for r in subset if int(r["tool_body_ran"]) > 0)
        verdict = "ALL" if ran == len(subset) else ("NONE" if ran == 0 else "MIXED")
        if verdict == "MIXED":
            mixed.append(label)
        print(f"{label:<26} executed {ran}/{len(subset)}  [{verdict}]")

    print()
    if mixed:
        print(f"NOT deterministic: {mixed} varied across identical prompts.")
    else:
        print(
            "Deterministic: every phrasing was all-or-nothing. Tool execution is a "
            "function of prompt text for this model, not a flaky runtime event."
        )


if __name__ == "__main__":
    asyncio.run(main())
