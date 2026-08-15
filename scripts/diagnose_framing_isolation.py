"""D4: isolate WHICH part of the probe framing suppresses tool execution.

The vocabulary A/B (`terminal_output/cuga-tracing/d4-vocabulary-ab.log`) gave a
clean split with identical wiring:

    secret_token   executed 0/3
    neutral_code   executed 0/3
    neutral_value  executed 3/3   ("read_build_number", mundane framing)

So the failure is prompt-semantic, not a CUGA/tool defect. Three candidate
variables were bundled together in the failing arms. This isolates them by
starting from the WORKING arm and changing exactly one thing at a time:

  baseline_working      : read_build_number  / mundane doc      / mundane task
  plus_cannot_be_known  : + "cannot be known without calling this" in the doc
  plus_random_task      : + "random and cannot be known" in the task
  plus_token_noun       : same wiring, value/noun renamed to "token"
  plus_secret_adjective : same wiring, doc says "secret"

If `plus_cannot_be_known` / `plus_random_task` drop to 0, the anti-guessing
phrasing itself is what tells the model the value is unobtainable. That phrasing
was introduced deliberately (to defeat guessable probes) and would then be
self-defeating.

Ground truth is always the tool function body.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

TRIALS = int(os.environ.get("D4_TRIALS", "3"))

ARMS = (
    "baseline_working",
    "plus_cannot_be_known",
    "plus_random_task",
    "plus_token_noun",
    "plus_secret_adjective",
)


def make_arm(kind: str):
    from langchain_core.tools import tool

    from cuga import tracked_tool

    executed: list[str] = []
    digits = f"{random.randint(10**11, 10**12 - 1)}"
    expected = f"BUILD-{digits}"

    doc = "Read the current build number for this workspace."
    task = (
        "Call read_build_number and report the exact build number it returns. "
        "The build number changes per run, so it must come from the tool."
    )

    if kind == "plus_cannot_be_known":
        doc = (
            "Read the current build number for this workspace. "
            "Cannot be known without calling this."
        )
    elif kind == "plus_random_task":
        task = (
            "Call read_build_number and report the exact build number it returns. "
            "The build number is random and cannot be known without calling the tool."
        )
    elif kind == "plus_token_noun":
        expected = f"TOKEN-{digits}"
        task = (
            "Call read_build_number and report the exact token it returns. "
            "The token changes per run, so it must come from the tool."
        )
    elif kind == "plus_secret_adjective":
        doc = "Read the secret build number for this workspace."
        task = (
            "Call read_build_number and report the exact secret build number it returns. "
            "The build number changes per run, so it must come from the tool."
        )

    @tracked_tool(app_name="probe")
    def read_build_number() -> str:
        executed.append("called")
        return expected

    # ``@tool`` needs a docstring at decoration time; set the description
    # explicitly so the doc text is the only variable that changes per arm.
    read_build_number.__doc__ = doc
    built = tool(read_build_number)
    return [built], task, expected, executed


async def run_arm(kind: str, index: int) -> dict:
    from cuga.sdk import CugaAgent

    tools, task, expected, executed = make_arm(kind)
    agent = CugaAgent(
        tools=tools,
        special_instructions="You are an autonomous agent. Use the available tools.",
        enable_knowledge=False,
        enable_skills=False,
    )
    try:
        result = await agent.invoke(task, thread_id=f"d4iso-{kind}-{index}", track_tool_calls=True)
        answer = str(result.answer or "")
        sdk_calls = len(result.tool_calls or [])
    finally:
        try:
            await agent.aclose()
        except Exception:  # noqa: BLE001
            pass

    return {
        "arm": kind,
        "trial": index,
        "tool_body_ran": len(executed),
        "value_in_answer": expected in answer,
        "sdk_tool_calls": sdk_calls,
        "answer_head": answer[:160],
    }


async def main() -> None:
    from agent_evolve.cuga_wrapper import RuntimeSettings, prepare_cuga_environment, resolve_skills_root

    prepare_cuga_environment()
    RuntimeSettings.from_env().configure_cuga_environment()
    os.environ["SKILLS_ROOT"] = resolve_skills_root()

    results: list[dict] = []
    for index in range(TRIALS):
        for kind in ARMS:
            outcome = await run_arm(kind, index)
            print(json.dumps(outcome), flush=True)
            results.append(outcome)

    print("=== D4 FRAMING ISOLATION (ground truth = tool body ran) ===")
    for kind in ARMS:
        subset = [r for r in results if r["arm"] == kind]
        ran = sum(1 for r in subset if int(r["tool_body_ran"]) > 0)
        print(f"{kind:<24} executed {ran}/{len(subset)}")


if __name__ == "__main__":
    asyncio.run(main())
