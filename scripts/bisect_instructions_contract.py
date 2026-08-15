"""Differential test with an UNGUESSABLE probe and repeated trials.

Why this replaces the previous bisect: the earlier probe used ``diag_add(17, 25)``,
whose answer (42) the model can produce by mental arithmetic. A correct final
answer therefore proved nothing, and a "no-exec" result could not be
distinguished from "model chose to answer directly". Any conclusion drawn from a
guessable probe is unsound.

Here the tool returns a per-run random secret that cannot be produced without a
real call, so:
  * tool body ran            -> recorded in ``invocations`` (ground truth)
  * answer contains secret   -> the agent genuinely used the tool result

Each case is repeated to measure flakiness, because a single run of a
nondeterministic agent cannot establish that a config option is the cause.
"""
from __future__ import annotations

import json
import os
import random

from dotenv import load_dotenv

from agent_evolve.cuga_wrapper import (
    DEFAULT_SPECIAL_INSTRUCTIONS,
    RuntimeSettings,
    prepare_cuga_environment,
    resolve_skills_root,
)

load_dotenv()
prepare_cuga_environment()
RuntimeSettings.from_env().configure_cuga_environment()
os.environ["SKILLS_ROOT"] = resolve_skills_root()

TRIALS = 2

TASK = (
    "Call the get_probe_token tool and report the exact token string it returns. "
    "The token is random and cannot be known without calling the tool."
)

# A prompt contract that states the ONLY mechanism by which the agent can act.
# CUGA routes a model response to the sandbox only when a fenced Python block is
# extracted from it; narrative text alone is classified and can finalize early.
CODE_CONTRACT_INSTRUCTIONS = (
    "You act ONLY by emitting a fenced Python code block. Tools are async Python "
    "functions available in the execution environment; call them with await, for "
    "example:\n"
    "```python\n"
    "result = await some_tool(arg)\n"
    "print(result)\n"
    "```\n"
    "Every step that needs a tool MUST be a fenced Python block that awaits the "
    "tool and prints the result. Never describe a call you have not emitted as "
    "code. Never state a tool result you have not seen printed in execution "
    "output. Give a final textual answer only after the required execution "
    "output is present."
)

invocations: list[str] = []


def build_probe(secret: str) -> object:
    from langchain_core.tools import tool

    from cuga import tracked_tool

    @tool
    @tracked_tool(app_name="diag")
    def get_probe_token() -> str:
        """Return the random probe token. It cannot be known without calling this."""
        invocations.append(secret)
        return secret

    return get_probe_token


async def run_case(name: str, **agent_kwargs: object) -> dict[str, object]:
    from cuga.sdk import CugaAgent

    secret = f"TKN-{random.randint(10**9, 10**10 - 1)}"
    invocations.clear()
    agent = CugaAgent(tools=[build_probe(secret)], **agent_kwargs)  # type: ignore[arg-type]
    try:
        result = await agent.invoke(TASK, track_tool_calls=True)
        answer = result.answer or ""
        return {
            "case": name,
            "tool_body_ran": bool(invocations),
            "secret_in_answer": secret in answer,
            "sdk_tool_calls": len(result.tool_calls or []),
            "answer_head": answer[:120],
        }
    except Exception as exc:  # noqa: BLE001 - failure is evidence
        return {"case": name, "error": repr(exc)[:200], "tool_body_ran": False}
    finally:
        try:
            await agent.aclose()
        except Exception:  # noqa: BLE001
            pass


async def main() -> None:
    cases = [
        # Current wrapper behavior: prose-style instructions.
        (
            "wrapper_prose_instructions",
            {
                "enable_knowledge": True,
                "enable_skills": False,
                "special_instructions": DEFAULT_SPECIAL_INSTRUCTIONS,
            },
        ),
        # Single variable changed: instructions state the code-block contract.
        (
            "code_contract_instructions",
            {
                "enable_knowledge": True,
                "enable_skills": False,
                "special_instructions": CODE_CONTRACT_INSTRUCTIONS,
            },
        ),
    ]

    rows: list[dict[str, object]] = []
    for name, kwargs in cases:
        for trial in range(TRIALS):
            label = f"{name}#{trial + 1}"
            print(f"--- {label} ---", flush=True)
            row = await run_case(label, **kwargs)
            rows.append(row)
            print(json.dumps(row), flush=True)

    print("=== SUMMARY (ground truth = tool_body_ran) ===")
    for row in rows:
        ran = "RAN" if row.get("tool_body_ran") else "NOT-RAN"
        used = "used" if row.get("secret_in_answer") else "unused"
        print(f"{str(row['case']):<34} {ran:<8} result-{used}  calls={row.get('sdk_tool_calls')}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
