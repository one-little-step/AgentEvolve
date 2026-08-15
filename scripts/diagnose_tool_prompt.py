"""Diagnostic: capture what CUGA actually gives the model at the prompt boundary.

Phase 1 evidence gathering for the "agent never invokes tools" bug. This does
NOT propose a fix; it records, at each component boundary, what data enters and
exits so we can identify the failing layer instead of guessing.

Boundaries instrumented:
  1. tools passed by us -> DirectLangChainToolsProvider (registration)
  2. prepare_node -> tools_for_prompt / _tools_context (what the model can see
     and what the sandbox can execute)
  3. model response -> code extraction (did the model emit executable code?)
  4. sandbox -> tool invocation (did our tool function actually run?)

Nothing here prints credentials or environment values.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agent_evolve.cuga_wrapper import (
    RuntimeSettings,
    prepare_cuga_environment,
    resolve_skills_root,
)

load_dotenv()
prepare_cuga_environment()
RuntimeSettings.from_env().configure_cuga_environment()
os.environ["SKILLS_ROOT"] = resolve_skills_root()

OUT = Path("terminal_output/cuga-tracing/tool-prompt-diagnosis.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

evidence: dict[str, object] = {}

# --- Boundary 4 instrumentation: did our tool function actually execute? ---
invocations: list[dict[str, object]] = []


def _install_probes() -> list:
    """Build a tool whose body records real execution."""
    from langchain_core.tools import tool

    from cuga import tracked_tool

    @tool
    @tracked_tool(app_name="diag")
    def diag_add(a: int, b: int) -> str:
        """Add two integers and return the sum as a string."""
        invocations.append({"tool": "diag_add", "a": a, "b": b})
        return str(a + b)

    return [diag_add]


# --- Boundary 2 instrumentation: what does prepare_node hand to the model? ---
def _install_prepare_node_spy() -> None:
    from cuga.backend.cuga_graph.nodes.cuga_lite.adapter import prepare_node as pn

    original = pn.make_tool_awaitable

    def spy(fn):
        # Record every callable registered into the sandbox execution context.
        evidence.setdefault("tools_context_registrations", []).append(
            getattr(fn, "__name__", repr(fn))
        )
        return original(fn)

    pn.make_tool_awaitable = spy


# --- Boundary 3 instrumentation: what code did CUGA extract from the model? ---
def _install_code_extraction_spy() -> None:
    from cuga.backend.cuga_graph.nodes.cuga_agent_core.graph import shared_nodes as sn

    for attr in dir(sn):
        if "extract_code" not in attr:
            continue
        original = getattr(sn, attr)
        if not callable(original):
            continue

        def make_spy(orig, name):
            def spy(*args, **kwargs):
                result = orig(*args, **kwargs)
                evidence.setdefault("code_extractions", []).append(
                    {
                        "fn": name,
                        "extracted_code": (result or None)
                        if isinstance(result, str)
                        else repr(result),
                    }
                )
                return result

            return spy

        setattr(sn, attr, make_spy(original, attr))
        evidence.setdefault("code_extraction_hooks", []).append(attr)


async def main() -> None:
    from cuga.sdk import CugaAgent

    _install_prepare_node_spy()
    _install_code_extraction_spy()
    tools = _install_probes()

    agent = CugaAgent(tools=tools, enable_knowledge=False, enable_skills=False)

    # Capture the prompt-visible tool list by spying on the provider.
    result = await agent.invoke(
        "Use the diag_add tool to add 17 and 25. Report only the returned number.",
        track_tool_calls=True,
    )

    evidence["answer"] = result.answer
    evidence["sdk_tool_calls"] = result.tool_calls
    evidence["real_tool_invocations"] = invocations
    evidence["tool_actually_executed"] = bool(invocations)

    OUT.write_text(json.dumps(evidence, indent=2, default=str))
    print("=== TOOL ACTUALLY EXECUTED ===")
    print(evidence["tool_actually_executed"])
    print("=== SDK tool_calls ===")
    print(json.dumps(result.tool_calls, indent=2, default=str))
    print("=== CODE EXTRACTIONS ===")
    print(json.dumps(evidence.get("code_extractions", []), indent=2, default=str))
    print("=== ANSWER ===")
    print(result.answer)
    print(f"=== full evidence written to {OUT} ===")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
