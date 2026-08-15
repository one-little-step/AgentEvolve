"""D4 evidence: is the probe tool actually present in the prompt the model sees?

The extraction boundary is already ruled out as the cause: the model emits no
``` fence at all and instead says it cannot call the tool
(``terminal_output/cuga-tracing/d4-code-emission.log``, 0/3 with identical
131-char refusals). That points one layer earlier — the prompt.

This records, for the exact prompt string handed to the model:
  * whether the probe tool name appears at all
  * whether the CodeAct/code-block contract text appears
  * which tool-ish sections are present
  * the count of tools registered into the sandbox execution context

No credentials or env values are printed; only structural facts and short heads.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

TOKEN = f"D4P-{random.randint(10**9, 10**10 - 1)}"
EXECUTED: list[str] = []
PROMPTS: list[dict[str, object]] = []
SANDBOX_REGISTRATIONS: list[str] = []


def install_prompt_spy() -> None:
    """Capture the messages actually sent to the model."""
    from cuga.backend.cuga_graph.nodes.cuga_agent_core.graph import shared_nodes as sn

    original = sn.extract_code_from_model_response

    def spy(content, reasoning_content, tools_needing_probing=frozenset()):
        return original(content, reasoning_content, tools_needing_probing)

    sn.extract_code_from_model_response = spy


def install_sandbox_spy() -> None:
    """Record every callable registered into the sandbox execution context."""
    from cuga.backend.cuga_graph.nodes.cuga_lite.adapter import prepare_node as pn

    original = pn.make_tool_awaitable

    def spy(fn):
        SANDBOX_REGISTRATIONS.append(getattr(fn, "__name__", repr(fn)))
        return original(fn)

    pn.make_tool_awaitable = spy


def install_model_input_spy() -> None:
    """Capture the fully-rendered prompt text per model call."""
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
        raise RuntimeError("no adapter class with ainvoke_model found")
    print(f"[spy] instrumenting {target.__name__}.ainvoke_model")

    original = target.ainvoke_model

    async def spy(self, bound, messages, config):  # noqa: ANN001
        rendered = "\n\n".join(str(getattr(m, "content", m)) for m in (messages or []))
        PROMPTS.append(
            {
                "call": len(PROMPTS),
                "message_count": len(messages or []),
                "prompt_chars": len(rendered),
                "mentions_probe_tool": "get_d4p_token" in rendered,
                "mentions_code_fence_contract": "```python" in rendered,
                "mentions_codeact": bool(re.search(r"code\s*act|CodeAct", rendered, re.I)),
                "mentions_run_command": "run_command" in rendered,
                "mentions_load_skill": "load_skill" in rendered,
                "tool_signature_lines": [
                    line.strip()[:160]
                    for line in rendered.splitlines()
                    if "get_d4p_token" in line
                ][:6],
            }
        )
        return await original(self, bound, messages, config)

    target.ainvoke_model = spy


def build_tool() -> list:
    from langchain_core.tools import tool

    from cuga import tracked_tool

    @tool
    @tracked_tool(app_name="probe")
    def get_d4p_token() -> str:
        """Return the secret token. Cannot be known without calling this."""
        EXECUTED.append("get_d4p_token")
        return TOKEN

    return [get_d4p_token]


async def main() -> None:
    from agent_evolve.cuga_wrapper import RuntimeSettings, prepare_cuga_environment, resolve_skills_root

    prepare_cuga_environment()
    RuntimeSettings.from_env().configure_cuga_environment()
    os.environ["SKILLS_ROOT"] = resolve_skills_root()

    install_prompt_spy()
    install_sandbox_spy()
    install_model_input_spy()

    from cuga.sdk import CugaAgent

    agent = CugaAgent(
        tools=build_tool(),
        special_instructions="You are an autonomous agent. Use the available tools.",
        enable_knowledge=False,
        enable_skills=False,
    )
    try:
        result = await agent.invoke(
            "Call get_d4p_token and report the exact token it returns.",
            thread_id="d4-prompt-probe",
            track_tool_calls=True,
        )
        answer = str(result.answer or "")
    finally:
        try:
            await agent.aclose()
        except Exception:  # noqa: BLE001
            pass

    print("=== D4 PROMPT-BOUNDARY EVIDENCE ===")
    print(
        json.dumps(
            {
                "tool_body_ran": EXECUTED.count("get_d4p_token"),
                "token_in_answer": TOKEN in answer,
                "sandbox_registrations": SANDBOX_REGISTRATIONS,
                "sandbox_registration_count": len(SANDBOX_REGISTRATIONS),
                "model_calls_captured": len(PROMPTS),
                "prompts": PROMPTS,
                "answer_head": answer[:300],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
