"""Behavioral verification that injected harness artifacts CHANGE agent behavior.

Structural verification (``verify_harness_injection.py``) only proves CUGA loaded
the artifacts into its stores. For a self-evolving harness optimizer that is not
sufficient: an artifact that loads but never reaches the model cannot be
optimized against, and a structural-only check would report false success.

This test injects unguessable tokens into each artifact class and then asks a
question that can only be answered from the injected content, so the token
appearing in the answer is proof the artifact reached the model at runtime.

Each class is judged independently:
  * memory   -> a random fact retrievable only from the ingested document
  * policy   -> an always-on playbook prescribing a required output marker
  * skill    -> a discoverable skill whose body carries a required marker
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MEMORY_TOKEN = f"MEM-{random.randint(10**9, 10**10 - 1)}"
POLICY_TOKEN = f"POL-{random.randint(10**9, 10**10 - 1)}"
SKILL_TOKEN = f"SKL-{random.randint(10**9, 10**10 - 1)}"

# Each artifact carries a token the model cannot know from pretraining.
HARNESS = {
    "memory": {
        "project-clearance-code": (
            f"The AgentEvolve project clearance code is {MEMORY_TOKEN}. "
            "This code is required when reporting project status."
        )
    },
    "policies": {
        "status-format": (
            "When reporting project status, you MUST end your reply with the "
            f"exact line: POLICY-MARKER: {POLICY_TOKEN}"
        )
    },
    "skills": {
        "status-report": (
            "Use this skill when asked for a project status report.\n\n"
            f"Every status report must include the skill signature {SKILL_TOKEN}.\n"
        )
    },
}

TASK_INPUT = (
    "Report the AgentEvolve project status. "
    "Include the project clearance code from your memory, "
    "follow your status formatting policy exactly, "
    "and apply your status-report skill including its signature."
)


def main() -> None:
    from agent_evolve.core.trace import PayloadLevel
    from agent_evolve.cuga_wrapper import CugaWrapper, RuntimeSettings, TraceConfig

    trace_config = TraceConfig(
        enabled=True,
        output_root=Path("data/traces"),
        payload_level=PayloadLevel.CAUSAL_SUFFICIENT,
    )

    wrapper = CugaWrapper.from_cuga(RuntimeSettings.from_env(), trace_config=trace_config)
    result = wrapper.run_task("harness-behavioral-check", {"input": TASK_INPUT, **HARNESS})

    answer = str(result.get("final_output", ""))
    verdict = {
        "status": result.get("status"),
        "memory_token_in_answer": MEMORY_TOKEN in answer,
        "policy_token_in_answer": POLICY_TOKEN in answer,
        "skill_token_in_answer": SKILL_TOKEN in answer,
        "causal_trace_path": result.get("causal_trace_path"),
    }
    verdict["all_three_influenced_behavior"] = all(
        (
            verdict["memory_token_in_answer"],
            verdict["policy_token_in_answer"],
            verdict["skill_token_in_answer"],
        )
    )

    print("=== HARNESS BEHAVIORAL VERDICT ===")
    print(json.dumps(verdict, indent=2))
    print("=== ANSWER ===")
    print(answer[:2000])


if __name__ == "__main__":
    main()
