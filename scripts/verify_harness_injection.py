"""Live verification that skills, policies, and memory are injected into a real
CUGA runtime from the wrapper's harness materialization.

Reads .env for model/endpoint; writes nothing secret. Each surface is checked
programmatically (not via LLM guesswork) and reported independently.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from agent_evolve.cuga_wrapper import (
    RuntimeSettings,
    _construct_agent,
    _require_autonomous_mode,
    materialize_harness,
    prepare_cuga_environment,
    resolve_skills_root,
)
from agent_evolve.cuga_wrapper.tools import build_tools

# --- Configuration -----------------------------------------------------------
WORKSPACE = Path("data/workspaces/verify-harness")
HARNESS = {
    "skills": {"greeting": "Always begin every answer with the phrase HELLO-WORLD."},
    "policies": {"be-concise": "Keep every answer to a single sentence."},
    "memory": {"favorite-color": "blue"},
}


async def main() -> int:
    prepare_cuga_environment()
    settings = RuntimeSettings.from_env()
    settings.configure_cuga_environment()
    os.environ["SKILLS_ROOT"] = resolve_skills_root()
    _require_autonomous_mode()

    tools = build_tools()
    workspace_dir = materialize_harness(HARNESS, WORKSPACE)
    agent = _construct_agent(HARNESS, tools, "You are a verification agent.", workspace_dir)

    report = {"workspace": workspace_dir}

    try:
        policies = await agent.policies.list()
        report["policies"] = [p["name"] for p in policies]
    except Exception as exc:  # noqa: BLE001
        report["policies_error"] = repr(exc)

    try:
        from cuga.backend.skills.loader import discover_skills

        report["skills"] = [e.name for e in discover_skills(workspace_dir)]
    except Exception as exc:  # noqa: BLE001
        report["skills_error"] = repr(exc)

    try:
        await agent.knowledge.ingest(str(Path(workspace_dir) / "memory" / "favorite-color.md"))
        report["memory_docs"] = await agent.knowledge.list_documents()
        search = await agent.knowledge.search("favorite color")
        report["memory_search"] = json.dumps(search, default=repr)[:500]
    except Exception as exc:  # noqa: BLE001
        report["memory_error"] = repr(exc)

    await agent.aclose()

    print(json.dumps(report, ensure_ascii=False, indent=2, default=repr))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
