"""End-to-end verification of the real CugaWrapper path with unguessable tasks.

This exercises the production code path (``CugaWrapper.run_task``), not a
synthetic agent, because the wrapper adds behavior the bare-agent tests do not
cover: harness materialization, memory ingestion, workspace directories,
``thread_id`` injection, and trace persistence.

Task design rules that make the result trustworthy:
  * Every required value is a random token, so no step can be satisfied by
    model prior knowledge or mental arithmetic.
  * The task requires multiple DIFFERENT tools and a data dependency between
    them, so a single lucky call cannot pass.
  * Ground truth is the recorded tool-body execution, never the model's claims.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Records real tool-body execution. Imported by the wrapper's tool module below.
EXECUTION_LOG = Path("terminal_output/cuga-tracing/e2e-tool-execution.jsonl")
EXECUTION_LOG.parent.mkdir(parents=True, exist_ok=True)
EXECUTION_LOG.write_text("")

TOKEN_A = f"ALPHA-{random.randint(10**9, 10**10 - 1)}"
TOKEN_B = f"BETA-{random.randint(10**9, 10**10 - 1)}"


def build_multistep_tools() -> list:
    """Three tools with a forced data dependency chain."""
    from langchain_core.tools import tool

    from cuga import tracked_tool

    def record(name: str, payload: dict) -> None:
        with EXECUTION_LOG.open("a") as handle:
            handle.write(json.dumps({"tool": name, **payload}) + "\n")

    @tool
    @tracked_tool(app_name="probe")
    def fetch_alpha_token() -> str:
        """Return the secret ALPHA token. Cannot be known without calling this."""
        record("fetch_alpha_token", {})
        return TOKEN_A

    @tool
    @tracked_tool(app_name="probe")
    def exchange_alpha_for_beta(alpha_token: str) -> str:
        """Exchange a valid ALPHA token for the BETA token. Requires the real ALPHA token."""
        record("exchange_alpha_for_beta", {"alpha_token": alpha_token})
        if alpha_token.strip() != TOKEN_A:
            return f"ERROR: invalid alpha token '{alpha_token}'. Call fetch_alpha_token first."
        return TOKEN_B

    @tool
    @tracked_tool(app_name="probe")
    def checksum_beta(beta_token: str) -> str:
        """Return the numeric checksum of a valid BETA token."""
        record("checksum_beta", {"beta_token": beta_token})
        if beta_token.strip() != TOKEN_B:
            return f"ERROR: invalid beta token '{beta_token}'. Call exchange_alpha_for_beta first."
        return str(sum(ord(character) for character in beta_token))

    return [fetch_alpha_token, exchange_alpha_for_beta, checksum_beta]


# Wording matters. `openai/azure/gpt-5.6-luna` decides whether to emit executable
# code as a deterministic function of the prompt phrasing, and CUGA only routes to
# the sandbox when the response contains a fenced code block
# (``extract_code_from_model_response``, cuga/.../shared_nodes.py:197). Imperative
# "Call X. Then report..." phrasing reproducibly emits no code and the model then
# claims it cannot call the tool. The explicit write-and-execute form reproducibly
# does emit code. This is a property of the model, not of the wrapper, so it is
# corrected here in the verification instrument rather than in production code.
# See reference/cuga_example_wrapper/docs/cuga-integration-learnings.md.
TASK_INPUT = (
    "Write and execute Python code that performs this three-step chain:\n"
    "1. Call fetch_alpha_token() to obtain the ALPHA token.\n"
    "2. Pass that exact ALPHA token to exchange_alpha_for_beta() to obtain the BETA token.\n"
    "3. Pass that exact BETA token to checksum_beta() to obtain the checksum.\n"
    "Then report the exact values the tools returned: the ALPHA token, the BETA "
    "token, and the checksum, each on its own line. All three values are random "
    "and cannot be known without executing the tools."
)


def main() -> None:
    from agent_evolve.core.trace import PayloadLevel
    from agent_evolve.cuga_wrapper import CugaWrapper, RuntimeSettings, TraceConfig

    trace_config = TraceConfig(
        enabled=True,
        output_root=Path("data/traces"),
        payload_level=PayloadLevel.CAUSAL_SUFFICIENT,
        max_observation_bytes=1_048_576,
    )

    wrapper = CugaWrapper.from_cuga(RuntimeSettings.from_env(), trace_config=trace_config)

    result = wrapper.run_task(
        "e2e-multistep-chain",
        {
            "input": TASK_INPUT,
            # Pass the probe tools explicitly through the harness so the run
            # exercises runtime tool injection rather than the default toolset.
            "tools": build_multistep_tools(),
        },
    )

    executed = [
        json.loads(line) for line in EXECUTION_LOG.read_text().splitlines() if line.strip()
    ]
    executed_names = [row["tool"] for row in executed]
    answer = str(result.get("final_output", ""))
    events = list(result.get("events") or [])
    graph_nodes = [
        str(event.get("node"))
        for event in events
        if event.get("kind") == "graph_node_start" and event.get("node")
    ]

    verdict = {
        "status": result.get("status"),
        "tools_really_executed": executed_names,
        "distinct_tools_executed": sorted(set(executed_names)),
        "chain_completed": executed_names[:3]
        == ["fetch_alpha_token", "exchange_alpha_for_beta", "checksum_beta"],
        "alpha_in_answer": TOKEN_A in answer,
        "beta_in_answer": TOKEN_B in answer,
        "sdk_tool_call_events": len([e for e in events if e.get("kind") == "tool_call"]),
        "graph_node_start_events": len(graph_nodes),
        "graph_nodes_observed": sorted(set(graph_nodes)),
        "causal_trace_path": result.get("causal_trace_path"),
    }

    print("=== E2E VERDICT ===")
    print(json.dumps(verdict, indent=2))
    print("=== FINAL ANSWER ===")
    print(answer[:1500])

    trace_path = result.get("causal_trace_path")
    if trace_path:
        manifest = Path(str(trace_path)) / "manifest.json"
        if manifest.exists():
            print("=== TRACE MANIFEST ===")
            print(manifest.read_text())


if __name__ == "__main__":
    main()
