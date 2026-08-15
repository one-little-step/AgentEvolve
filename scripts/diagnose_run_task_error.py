"""Surface the exception that ``CugaSdkRuntime.run_task`` currently swallows.

``run_task`` converts any invoke exception into ``status="error"`` with an empty
``final_output`` and does not persist the exception text, so a failing live run
leaves no diagnosable evidence. This script re-runs the same wrapper path with
the exception printed, to identify the failure before changing any behavior.
"""
from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

EXECUTED: list[str] = []
TOKEN = "ALPHA-7731900021"


def build_tools() -> list:
    from langchain_core.tools import tool

    from cuga import tracked_tool

    @tool
    @tracked_tool(app_name="probe")
    def fetch_alpha_token() -> str:
        """Return the secret ALPHA token. Cannot be known without calling this."""
        EXECUTED.append("fetch_alpha_token")
        return TOKEN

    return [fetch_alpha_token]


def main() -> None:
    from agent_evolve.cuga_wrapper import (
        DEFAULT_SPECIAL_INSTRUCTIONS,
        RuntimeSettings,
        _construct_agent,
        _execute,
        _require_autonomous_mode,
        resolve_skills_root,
    )
    import os

    settings = RuntimeSettings.from_env()
    settings.configure_cuga_environment()
    os.environ["SKILLS_ROOT"] = resolve_skills_root()
    _require_autonomous_mode()

    tools = build_tools()
    agent = _construct_agent({"tools": tools}, tools, DEFAULT_SPECIAL_INSTRUCTIONS, None)

    from agent_evolve.cuga_wrapper import GraphEventCollector, build_graph_callback_handler

    collector = GraphEventCollector(max_events=10_000)
    thread_id = "diagnose-run-task-error"
    invoke_kwargs = {
        "track_tool_calls": True,
        "thread_id": thread_id,
        "config": {
            "configurable": {"thread_id": thread_id},
            "callbacks": [build_graph_callback_handler(collector)],
        },
    }

    verdict: dict = {}
    try:
        result = asyncio.run(_execute(agent, "Call fetch_alpha_token and report the token.", [], invoke_kwargs))
        verdict.update(
            answer=str(getattr(result, "answer", ""))[:400],
            sdk_error=getattr(result, "error", None),
            sdk_tool_calls=len(getattr(result, "tool_calls", ()) or ()),
        )
    except Exception as exc:  # noqa: BLE001 - the point of this script
        verdict["exception"] = repr(exc)
        verdict["traceback"] = traceback.format_exc()[-3000:]

    verdict["tool_body_ran"] = EXECUTED.count("fetch_alpha_token")
    verdict["collector_event_count"] = len(collector.events)
    verdict["collector_kinds"] = sorted({str(e["kind"]) for e in collector.events})
    verdict["collector_nodes"] = sorted(
        {str(e.get("node")) for e in collector.events if e.get("node")}
    )
    print("=== RUN_TASK ERROR DIAGNOSIS ===")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
