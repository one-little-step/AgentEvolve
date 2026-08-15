"""Produce ONE trace exercising every facility: tools + payloads + edges + topology."""
import json, random
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(".env")

LOG = Path("terminal_output/cuga-tracing/complete-demo-tools.jsonl")
LOG.parent.mkdir(parents=True, exist_ok=True); LOG.write_text("")
A = f"ALPHA-{random.randint(10**9,10**10-1)}"
B = f"BETA-{random.randint(10**9,10**10-1)}"

def build_tools():
    from langchain_core.tools import tool
    from cuga import tracked_tool
    def rec(n, p):
        with LOG.open("a") as f: f.write(json.dumps({"tool": n, **p}) + "\n")

    @tool
    @tracked_tool(app_name="demo")
    def fetch_alpha_token() -> str:
        """Return the ALPHA token."""
        rec("fetch_alpha_token", {}); return A

    @tool
    @tracked_tool(app_name="demo")
    def exchange_alpha_for_beta(alpha_token: str) -> str:
        """Exchange a valid ALPHA token for the BETA token."""
        rec("exchange_alpha_for_beta", {"in": alpha_token})
        return B if alpha_token.strip() == A else f"ERROR: bad alpha {alpha_token}"

    @tool
    @tracked_tool(app_name="demo")
    def checksum_beta(beta_token: str) -> str:
        """Return the numeric checksum of a valid BETA token."""
        rec("checksum_beta", {"in": beta_token})
        return str(sum(ord(c) for c in beta_token)) if beta_token.strip() == B else "ERROR: bad beta"

    return [fetch_alpha_token, exchange_alpha_for_beta, checksum_beta]

from agent_evolve.core.trace import PayloadLevel
from agent_evolve.cuga_wrapper import CugaWrapper, RuntimeSettings, TraceConfig

cfg = TraceConfig(
    enabled=True, output_root=Path("data/traces"),
    payload_level=PayloadLevel.RAW_OPT_IN, allow_raw_payloads=True,
    capture_node_payloads=True, max_observation_bytes=4_194_304,
)
wrapper = CugaWrapper.from_cuga(RuntimeSettings.from_env(), trace_config=cfg)
result = wrapper.run_task("complete-graph-demo", {
    "input": (
        "Write and execute Python code that calls fetch_alpha_token(), passes that exact "
        "returned value to exchange_alpha_for_beta(), then passes that exact returned value "
        "to checksum_beta(). Then report the three exact values the tools returned, one per line."
    ),
    "tools": build_tools(),
})
d = Path(str(result["causal_trace_path"]))
executed = [json.loads(l)["tool"] for l in LOG.read_text().splitlines() if l.strip()]
print("RESULT_START")
print("trace:", d)
print("tools really executed:", executed)
print("status:", result.get("status"))
print("sdk tool_call events:", sum(1 for e in (result.get("events") or []) if e.get("kind")=="tool_call"))
