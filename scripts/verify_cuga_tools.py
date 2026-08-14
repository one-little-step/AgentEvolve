"""Live CUGA wrapper verification: ask the agent about its tools, then force a
five-tool invocation and report per-tool success.

Edit the configuration block below. This runner accepts no command-line
arguments. Credentials stay in ``.env``; nothing secret is written to stdout.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from agent_evolve.cuga_wrapper import (
    CugaWrapper,
    RuntimeSettings,
    prepare_cuga_environment,
)

# --- Configuration -----------------------------------------------------------
PROMPT_TOOLS_AND_SKILLS = "What tools and skills do you have available?"
PROMPT_FIVE_STEP = (
    "Execute the following 5 steps in order and report the result of each:\n"
    "1. Use the calculator tool to multiply 17 by 24.\n"
    "2. Use the web_search tool to find the current capital of Australia.\n"
    "3. Use the wikipedia_search tool to find the summary for 'Albert Einstein'.\n"
    "4. Use the web_fetch tool to retrieve the content of 'http://example.com'.\n"
    "5. Use the save_note tool to save the text 'All 5 tools verified successfully'.\n"
    "Report the exact output or success status for each of the 5 steps."
)
EXPECTED_TOOL_NAMES = ("calculator", "web_search", "wikipedia_search", "web_fetch", "save_note")


def _tool_call_name(call: object) -> str:
    if isinstance(call, dict):
        return str(call.get("name") or call.get("tool_name") or call.get("operation_id") or "")
    return str(getattr(call, "name", "") or getattr(call, "tool_name", "") or getattr(call, "operation_id", "") or "")


def _tool_call_result(call: object) -> str:
    if isinstance(call, dict):
        return json.dumps(call.get("result") or call.get("output") or call.get("error") or "", default=repr)
    result = getattr(call, "result", None)
    if result is None:
        result = getattr(call, "error", None)
    return repr(result) if result is not None else ""


def summarize(trace: dict, label: str) -> dict:
    events = trace.get("events", [])
    tool_calls = [e.get("tool_call") for e in events if isinstance(e, dict) and e.get("kind") == "tool_call"]
    per_tool = {}
    for call in tool_calls:
        name = _tool_call_name(call)
        per_tool[name] = {
            "name": name,
            "result": _tool_call_result(call)[:400],
        }
    return {
        "label": label,
        "status": trace.get("status"),
        "final_output": str(trace.get("final_output", ""))[:2000],
        "tool_calls": per_tool,
    }


def main() -> int:
    prepare_cuga_environment()
    settings = RuntimeSettings.from_env()
    wrapper = CugaWrapper.from_cuga(settings)

    print(json.dumps({"model": settings.public_config()["model"], "started_at": datetime.now(timezone.utc).isoformat()}))

    for index, (task_id, prompt) in enumerate(
        (("verify-tools-skills", PROMPT_TOOLS_AND_SKILLS), ("verify-five-step", PROMPT_FIVE_STEP)),
        1,
    ):
        trace = wrapper.run_task(task_id, {"input": prompt})
        summary = summarize(trace, task_id)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        if index == 2:
            invoked = set(summary["tool_calls"].keys())
            missing = [name for name in EXPECTED_TOOL_NAMES if name not in invoked]
            print(json.dumps({"five_step_invoked": sorted(invoked), "missing": missing}, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
