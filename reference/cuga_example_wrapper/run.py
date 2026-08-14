from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool

from cuga import CugaAgent, tracked_tool


ROOT = Path(__file__).resolve().parent
TRACE_DIR = ROOT / "data" / "traces"


def configure_environment() -> dict[str, Any]:
    load_dotenv(ROOT / ".env")

    model = os.getenv("CUGA_MODEL") or os.getenv("MODEL_NAME")
    if not model:
        raise RuntimeError(
            "Set CUGA_MODEL in .env, e.g. CUGA_MODEL=gpt-4o"
        )

    base_url = os.getenv("CUGA_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("CUGA_API_KEY") or os.getenv("OPENAI_API_KEY")

    # CUGA's documented OpenAI configuration.
    os.environ["AGENT_SETTING_CONFIG"] = os.getenv(
        "AGENT_SETTING_CONFIG", "settings.openai.toml"
    )
    os.environ["MODEL_NAME"] = model.removeprefix("openai/")

    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # CUGA skills use a configured root. The default "cuga" means
    # .cuga/skills in the documented CUGA-native layout.
    os.environ["SKILLS_ROOT"] = os.getenv("SKILLS_ROOT", "cuga")

    mcp_file = os.getenv("MCP_SERVERS_FILE")
    if mcp_file:
        os.environ["MCP_SERVERS_FILE"] = str(Path(mcp_file).expanduser().resolve())

    # Optional observability switches.
    if os.getenv("LANGFUSE_TRACING", "").lower() == "true":
        os.environ["LANGFUSE_TRACING"] = "true"
    if os.getenv("OPENLIT", "").lower() == "true":
        os.environ["OPENLIT"] = "true"

    return {
        "model": model,
        "base_url": base_url,
        "configuration": os.environ["AGENT_SETTING_CONFIG"],
        "skills_root": os.environ["SKILLS_ROOT"],
        "mcp_servers_file": os.getenv("MCP_SERVERS_FILE"),
    }


@tool
@tracked_tool(app_name="calculator")
def multiply(left: int, right: int) -> int:
    """Multiply two integers."""
    return left * right


@tool
@tracked_tool(app_name="calculator")
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


@tool
@tracked_tool(app_name="research")
def save_note(note: str) -> str:
    """Save a small local research note for this demo run."""
    path = ROOT / "data" / "notes.txt"
    with path.open("a", encoding="utf-8") as f:
        f.write(note.rstrip() + "\n")
    return f"Saved note to {path}"


CUSTOM_TOOLS = [multiply, add, save_note]


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        return repr(value)


class CugaExperiment:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.agent: CugaAgent | None = None

    async def create_agent(self) -> CugaAgent:
        # These are documented CugaAgent constructor surfaces.
        #
        # Knowledge is enabled explicitly because the SDK exposes
        # agent.knowledge only when enable_knowledge=True.
        #
        # Skills are configured through CUGA's [skills] settings and
        # .cuga/skills root; current SDK docs do not list an
        # enable_skills constructor parameter, so we do NOT invent one.
        self.agent = CugaAgent(
            tools=CUSTOM_TOOLS,
            special_instructions=(
                "You are running inside an experimental CUGA wrapper. "
                "Use available tools when appropriate. "
                "If a capability is not actually available, say so."
            ),
            enable_knowledge=True,
        )
        return self.agent

    async def run(self, prompt: str) -> dict[str, Any]:
        agent = await self.create_agent()

        run_id = str(uuid.uuid4())
        started = datetime.now(timezone.utc).isoformat()
        events: list[dict[str, Any]] = []
        streamed: list[Any] = []

        try:
            # Stream first so the experiment captures intermediate graph
            # outputs exposed by CUGA's stream interface.
            async for state in agent.stream(prompt):
                item = json_safe(state)
                streamed.append(item)
                events.append({
                    "kind": "stream_event",
                    "index": len(streamed) - 1,
                    "state": item,
                })

            # Run invoke separately with tool tracking to obtain the
            # documented InvokeResult/tool_calls surface.
            result = await agent.invoke(prompt, track_tool_calls=True)

            tool_calls = [
                json_safe(call)
                for call in (getattr(result, "tool_calls", None) or [])
            ]

            for index, call in enumerate(tool_calls):
                events.append({
                    "kind": "tool_call",
                    "index": index,
                    "tool_call": call,
                })

            output = {
                "run_id": run_id,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "config": self.config,
                "runtime": {
                    "cuga_agent": "CugaAgent",
                    "knowledge_enabled": True,
                    "custom_tools": [t.name for t in CUSTOM_TOOLS],
                    "skill_root": os.getenv("SKILLS_ROOT"),
                    "mcp_servers_file": os.getenv("MCP_SERVERS_FILE"),
                },
                "prompt": prompt,
                "events": events,
                "stream_events": streamed,
                "tool_calls": tool_calls,
                "answer": json_safe(getattr(result, "answer", "")),
                "error": json_safe(getattr(result, "error", None)),
                "thread_id": json_safe(getattr(result, "thread_id", None)),
                "note": (
                    "This is an event/tool trajectory. Exact replayable "
                    "LangGraph checkpoints require a configured checkpointer "
                    "and direct use of agent.graph."
                ),
            }
            return output
        finally:
            await agent.aclose()

    async def knowledge_demo(self) -> None:
        agent = await self.create_agent()
        try:
            # Demonstrates that the KnowledgeManager is exposed.
            docs = await agent.knowledge.list_documents()
            print(json.dumps(json_safe(docs), indent=2))
        finally:
            await agent.aclose()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="*", default=["what tools and skills are available?"])
    parser.add_argument(
        "--knowledge-list",
        action="store_true",
        help="List documents through CUGA's KnowledgeManager and exit.",
    )
    args = parser.parse_args()

    config = configure_environment()
    experiment = CugaExperiment(config)

    if args.knowledge_list:
        await experiment.knowledge_demo()
        return 0

    prompt = " ".join(args.prompt).strip()
    trace = await experiment.run(prompt)

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{trace['run_id']}.json"
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "run_id": trace["run_id"],
        "answer": trace["answer"],
        "tool_calls": trace["tool_calls"],
        "trace": str(path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
