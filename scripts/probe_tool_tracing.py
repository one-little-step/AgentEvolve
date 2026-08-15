"""Decisive test: does the live CUGA agent actually invoke a custom tool?

Uses a tool with an observable side effect and an unguessable return value, so
we can tell definitively whether the tool ran and whether CUGA surfaced it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import os

from dotenv import load_dotenv
from langchain_core.tools import tool

from cuga import tracked_tool
from cuga.sdk import CugaAgent

from agent_evolve.cuga_wrapper import RuntimeSettings, prepare_cuga_environment, resolve_skills_root

load_dotenv()
prepare_cuga_environment()
RuntimeSettings.from_env().configure_cuga_environment()
os.environ["SKILLS_ROOT"] = resolve_skills_root()

SIDE_EFFECT_PATH = Path("terminal_output/cuga-tracing/tool-invocation-evidence.jsonl")
SIDE_EFFECT_PATH.parent.mkdir(parents=True, exist_ok=True)
SIDE_EFFECT_PATH.write_text("")

secret = random.randint(100000, 999999)


@tool
@tracked_tool(app_name="probe")
def probe_number() -> str:
    """Return the secret number; you cannot know it without calling this tool."""
    with SIDE_EFFECT_PATH.open("a") as handle:
        handle.write(json.dumps({"secret": secret}) + "\n")
    return str(secret)


async def main() -> None:
    agent = CugaAgent(tools=[probe_number], enable_knowledge=False, enable_skills=False)
    result = await agent.invoke(
        "Use the probe_number tool to get the secret number, then tell me the secret number.",
        track_tool_calls=True,
    )
    print("=== answer ===")
    print(result.answer)
    print("=== tool_calls ===")
    print(json.dumps(result.tool_calls, indent=2))
    print("=== side-effect evidence (lines) ===")
    print(SIDE_EFFECT_PATH.read_text().strip() or "(tool never invoked)")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
