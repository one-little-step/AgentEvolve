"""D4 evidence: why does the model stop emitting executable CodeAct blocks?

Phase 1 boundary instrumentation, no fix. The failing layer is already narrowed
to ``call_model`` -> ``extract_code_from_model_response``: when extraction
returns "", ``shared_nodes.call_model`` (line 233) never routes to the execute
node, so the tool body cannot run. This records what the model actually emitted
at that boundary.

Ground truth for "did the tool run" is the tool function body, never the model's
claims and never ``InvokeResult.tool_calls``.

Captured per model turn:
  * whether ``content`` / ``reasoning_content`` were non-empty
  * whether either contained a ``` fence at all
  * the extraction result length
  * a redacted head of the content so a prose refusal is distinguishable from
    a formatting failure

Prints only structural facts and short heads; no credentials, no env values.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

TOKEN = f"D4-{random.randint(10**9, 10**10 - 1)}"
EXECUTED: list[str] = []
TURNS: list[dict[str, object]] = []


def install_extraction_spy() -> None:
    """Record every code-extraction decision made during the run."""
    from cuga.backend.cuga_graph.nodes.cuga_agent_core.graph import shared_nodes as sn

    original = sn.extract_code_from_model_response

    def spy(content, reasoning_content, tools_needing_probing=frozenset()):
        code = original(content, reasoning_content, tools_needing_probing)
        content_text = content or ""
        reasoning_text = reasoning_content or ""
        TURNS.append(
            {
                "turn": len(TURNS),
                "content_chars": len(content_text),
                "reasoning_chars": len(reasoning_text),
                "content_has_fence": "```" in content_text,
                "reasoning_has_fence": "```" in reasoning_text,
                "content_mentions_python_fence": "```python" in content_text,
                "extracted_code_chars": len(code or ""),
                "routed_to_execute_node": bool(code),
                "content_head": content_text[:400],
                "reasoning_head": reasoning_text[:200],
            }
        )
        return code

    sn.extract_code_from_model_response = spy


def build_tool() -> list:
    from langchain_core.tools import tool

    from cuga import tracked_tool

    @tool
    @tracked_tool(app_name="probe")
    def get_d4_token() -> str:
        """Return the secret D4 token. Cannot be known without calling this."""
        EXECUTED.append("get_d4_token")
        return TOKEN

    return [get_d4_token]


async def run_once(label: str) -> dict:
    from cuga.sdk import CugaAgent

    EXECUTED.clear()
    TURNS.clear()

    agent = CugaAgent(
        tools=build_tool(),
        special_instructions="You are an autonomous agent. Use the available tools.",
        enable_knowledge=False,
        enable_skills=False,
    )
    try:
        result = await agent.invoke(
            "Call get_d4_token and report the exact token string it returns. "
            "The token is random and cannot be known without calling the tool.",
            thread_id=f"d4-{label}",
            track_tool_calls=True,
        )
        answer = str(result.answer or "")
        sdk_calls = len(result.tool_calls or [])
        error = result.error
    finally:
        try:
            await agent.aclose()
        except Exception:  # noqa: BLE001
            pass

    return {
        "label": label,
        "tool_body_ran": EXECUTED.count("get_d4_token"),
        "token_in_answer": TOKEN in answer,
        "sdk_tool_calls": sdk_calls,
        "sdk_error": error,
        "model_turns": len(TURNS),
        "any_turn_emitted_code": any(bool(t["routed_to_execute_node"]) for t in TURNS),
        "any_turn_had_fence": any(
            bool(t["content_has_fence"]) or bool(t["reasoning_has_fence"]) for t in TURNS
        ),
        "turns": TURNS.copy(),
        "answer_head": answer[:300],
    }


async def main() -> None:
    from agent_evolve.cuga_wrapper import RuntimeSettings, prepare_cuga_environment, resolve_skills_root

    prepare_cuga_environment()
    RuntimeSettings.from_env().configure_cuga_environment()
    os.environ["SKILLS_ROOT"] = resolve_skills_root()

    install_extraction_spy()

    trials = int(os.environ.get("D4_TRIALS", "3"))
    results = [await run_once(f"t{index}") for index in range(trials)]

    print("=== D4 CODE-EMISSION EVIDENCE ===")
    print(json.dumps(results, indent=2))

    ran = sum(1 for r in results if int(r["tool_body_ran"]) > 0)
    emitted = sum(1 for r in results if r["any_turn_emitted_code"])
    fenced = sum(1 for r in results if r["any_turn_had_fence"])
    print("=== SUMMARY ===")
    print(f"tool body executed:        {ran}/{len(results)}")
    print(f"emitted extractable code:  {emitted}/{len(results)}")
    print(f"emitted any ``` fence:     {fenced}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
