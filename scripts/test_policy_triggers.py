"""Test whether policy triggers other than ``always`` actually match.

Context: a playbook written with ``triggers: {always: true}`` loads and
deserializes into an ``AlwaysTrigger``, but never matches at runtime. Root cause
in cuga 0.3.1: ``PolicyAgent.match_policy`` builds candidates only from
``_evaluate_keyword_triggered_policies`` (filters ``KeywordTrigger``) and
``_evaluate_natural_language_policies`` (filters NL triggers). No evaluator ever
selects an ``AlwaysTrigger``, so an always-only policy cannot win.

This writes playbooks directly (bypassing ``materialize_harness``, which only
emits ``always``) so each trigger type can be judged independently. Ground truth
is whether the policy's mandated marker token appears in the final answer.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

WORKSPACE = Path("data/workspaces/policy-trigger-test")

KEYWORD_TOKEN = f"KWD-{random.randint(10**9, 10**10 - 1)}"
NL_TOKEN = f"NLT-{random.randint(10**9, 10**10 - 1)}"

# The task deliberately contains the keyword "status" so a keyword trigger can fire.
TASK = "Give me a project status summary for AgentEvolve."


def write_playbook(name: str, triggers_yaml: str, body: str) -> None:
    """Write one playbook. ``id`` is required or filesystem_sync deletes it."""
    playbooks = WORKSPACE / "playbooks"
    playbooks.mkdir(parents=True, exist_ok=True)
    (playbooks / f"{name}.md").write_text(
        f"---\nname: {name}\nid: playbook_{name}\n{triggers_yaml}---\n{body}\n",
        encoding="utf-8",
    )


async def run_variant(label: str, triggers_yaml: str, token: str) -> dict[str, object]:
    from agent_evolve.cuga_wrapper import (
        RuntimeSettings,
        _construct_agent,
        prepare_cuga_environment,
        resolve_skills_root,
    )

    prepare_cuga_environment()
    RuntimeSettings.from_env().configure_cuga_environment()
    os.environ["SKILLS_ROOT"] = resolve_skills_root()

    # Fresh workspace per variant so a previous playbook cannot leak in.
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    write_playbook(
        "status-format",
        triggers_yaml,
        f"When reporting project status you MUST end the reply with the exact line:\n"
        f"POLICY-MARKER: {token}",
    )

    harness = {"policies": {"status-format": "placeholder"}}
    agent = _construct_agent(harness, [], "You are a status reporting agent.", str(WORKSPACE))
    try:
        result = await agent.invoke(TASK, track_tool_calls=True)
        answer = result.answer or ""
        return {
            "variant": label,
            "token_in_answer": token in answer,
            "answer_tail": answer[-160:],
        }
    except Exception as exc:  # noqa: BLE001 - failure is evidence
        return {"variant": label, "error": repr(exc)[:200], "token_in_answer": False}
    finally:
        try:
            await agent.aclose()
        except Exception:  # noqa: BLE001
            pass


async def main() -> None:
    variants = [
        (
            # NOTE: the frontmatter key is ``keywords`` (plural). ``keyword``
            # yields zero triggers and CUGA rejects the file with
            # "Playbook ... must have at least one trigger".
            "keyword_trigger",
            "triggers:\n  keywords:\n    - status\n  target: intent\n",
            KEYWORD_TOKEN,
        ),
        (
            "natural_language_trigger",
            "triggers:\n  natural_language:\n    - user asks for a project status report\n"
            "  target: intent\n  threshold: 0.5\n",
            NL_TOKEN,
        ),
    ]

    rows = []
    for label, triggers_yaml, token in variants:
        print(f"--- {label} ---", flush=True)
        row = await run_variant(label, triggers_yaml, token)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print("=== POLICY TRIGGER SUMMARY ===")
    for row in rows:
        print(f"{str(row['variant']):<28} {'MATCHED' if row.get('token_in_answer') else 'NOT-MATCHED'}")


if __name__ == "__main__":
    asyncio.run(main())
