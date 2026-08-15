"""D4: is the controlling variable the TOOL CONSTRUCTION, not the wording?

The framing isolation refuted the vocabulary hypothesis
(`terminal_output/cuga-tracing/d4-framing-isolation.log`): the arm that scored
3/3 in the vocabulary A/B scored 0/3 in the isolation run with identical wording.

The one material difference between the two scripts is how the tool object was
built:

  A (vocabulary script, PASSED 3/3):
        @tool
        @tracked_tool(app_name="probe")
        def read_build_number() -> str:
            \"\"\"literal docstring\"\"\"

  B (isolation script, FAILED 0/3):
        @tracked_tool(app_name="probe")
        def read_build_number() -> str: ...
        read_build_number.__doc__ = doc     # post-hoc
        built = tool(read_build_number)

This runs BOTH constructions against a byte-identical task in one process and
prints the resulting tool metadata (name, description, args schema) plus the
rendered-prompt hash, so a construction difference is visible rather than
inferred.

Second correction applied here: the previous runs were not independent samples.
The random value appears only in the tool's RETURN, never in the prompt, so every
trial sent an identical prompt and this reasoning model decodes greedily
(temperature is skipped) — identical prompt gives identical output. This script
varies the task text per trial so trials are genuinely distinct samples.

Ground truth remains the tool function body.
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

TRIALS = int(os.environ.get("D4_TRIALS", "3"))
DOC = "Read the current build number for this workspace."
PROMPT_HASHES: list[str] = []


def install_prompt_hash_spy() -> None:
    """Record a hash of each rendered prompt to prove trials differ."""
    from cuga.backend.cuga_graph.nodes.cuga_lite.adapter import graph_adapter as ga

    target = None
    for name in dir(ga):
        if name.startswith("_"):
            continue
        candidate = getattr(ga, name)
        if isinstance(candidate, type) and "ainvoke_model" in candidate.__dict__:
            target = candidate
            break
    if target is None:
        return

    original = target.ainvoke_model

    async def spy(self, bound, messages, config):  # noqa: ANN001
        rendered = "\n\n".join(str(getattr(m, "content", m)) for m in (messages or []))
        PROMPT_HASHES.append(hashlib.sha256(rendered.encode()).hexdigest()[:12])
        return await original(self, bound, messages, config)

    target.ainvoke_model = spy


def build_construction_a(expected: str, executed: list[str]) -> list:
    """Literal docstring, ``@tool`` stacked over ``@tracked_tool``."""
    from langchain_core.tools import tool

    from cuga import tracked_tool

    @tool
    @tracked_tool(app_name="probe")
    def read_build_number() -> str:
        """Read the current build number for this workspace."""
        executed.append("called")
        return expected

    return [read_build_number]


def build_construction_b(expected: str, executed: list[str]) -> list:
    """Post-hoc ``__doc__`` assignment, then ``tool(...)`` applied manually."""
    from langchain_core.tools import tool

    from cuga import tracked_tool

    @tracked_tool(app_name="probe")
    def read_build_number() -> str:
        executed.append("called")
        return expected

    read_build_number.__doc__ = DOC
    return [tool(read_build_number)]


def describe_tool(built: object) -> dict:
    return {
        "name": getattr(built, "name", None),
        "description": str(getattr(built, "description", ""))[:200],
        "args_schema_fields": sorted(
            (getattr(built, "args", None) or {}).keys()
        ) if getattr(built, "args", None) else [],
        "has_coroutine": callable(getattr(built, "coroutine", None)),
        "has_func": callable(getattr(built, "func", None)),
    }


async def run_arm(construction: str, index: int) -> dict:
    from cuga.sdk import CugaAgent

    executed: list[str] = []
    expected = f"BUILD-{random.randint(10**11, 10**12 - 1)}"
    builder = build_construction_a if construction == "A_literal_docstring" else build_construction_b
    tools = builder(expected, executed)

    # Vary the task per trial so trials are independent samples, not one
    # identical prompt decoded greedily three times.
    suffix = ("", " Respond with only the value.", " Return just the value, nothing else.")[index % 3]
    task = (
        "Call read_build_number and report the exact build number it returns. "
        "The build number changes per run, so it must come from the tool." + suffix
    )

    before = len(PROMPT_HASHES)
    agent = CugaAgent(
        tools=tools,
        special_instructions="You are an autonomous agent. Use the available tools.",
        enable_knowledge=False,
        enable_skills=False,
    )
    try:
        result = await agent.invoke(
            task, thread_id=f"d4c-{construction}-{index}", track_tool_calls=True
        )
        answer = str(result.answer or "")
        sdk_calls = len(result.tool_calls or [])
    finally:
        try:
            await agent.aclose()
        except Exception:  # noqa: BLE001
            pass

    return {
        "construction": construction,
        "trial": index,
        "tool_body_ran": len(executed),
        "value_in_answer": expected in answer,
        "sdk_tool_calls": sdk_calls,
        "tool_metadata": describe_tool(tools[0]),
        "prompt_hashes": PROMPT_HASHES[before:],
        "answer_head": answer[:160],
    }


async def main() -> None:
    from agent_evolve.cuga_wrapper import RuntimeSettings, prepare_cuga_environment, resolve_skills_root

    prepare_cuga_environment()
    RuntimeSettings.from_env().configure_cuga_environment()
    os.environ["SKILLS_ROOT"] = resolve_skills_root()
    install_prompt_hash_spy()

    constructions = ("A_literal_docstring", "B_posthoc_doc")
    results: list[dict] = []
    for index in range(TRIALS):
        for construction in constructions:
            outcome = await run_arm(construction, index)
            print(json.dumps(outcome), flush=True)
            results.append(outcome)

    print("=== D4 CONSTRUCTION A/B (ground truth = tool body ran) ===")
    for construction in constructions:
        subset = [r for r in results if r["construction"] == construction]
        ran = sum(1 for r in subset if int(r["tool_body_ran"]) > 0)
        print(f"{construction:<22} executed {ran}/{len(subset)}")
    distinct = {h for r in results for h in r["prompt_hashes"]}
    print(f"distinct prompt hashes observed: {len(distinct)} (proves trials differ)")


if __name__ == "__main__":
    asyncio.run(main())
