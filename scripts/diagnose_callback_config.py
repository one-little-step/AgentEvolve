"""Isolate whether passing ``config`` with callbacks into ``CugaAgent.invoke``
breaks the run, and whether node-level callback events actually arrive.

Evidence-gathering only. Compares three variants against the SAME bare agent
surface so a single variable changes at a time:

  A. invoke(thread_id=..., track_tool_calls=True)                  [baseline]
  B. invoke(..., config={"configurable": {"thread_id": ...}, "callbacks": [h]})
  C. invoke(..., config={"callbacks": [h]})   # no thread_id inside config
"""
from __future__ import annotations

import asyncio
import json
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

EXECUTED: list[str] = []


def build_agent():
    from langchain_core.tools import tool

    from cuga import CugaAgent, tracked_tool

    @tool
    @tracked_tool(app_name="probe")
    def ping() -> str:
        """Return the secret ping token. Cannot be known without calling this."""
        EXECUTED.append("ping")
        return "PONG-4417"

    return CugaAgent(tools=[ping], special_instructions="Use the tools.", enable_knowledge=True)


def make_handler():
    from agent_evolve.cuga_wrapper import GraphEventCollector, build_graph_callback_handler

    collector = GraphEventCollector(max_events=10_000)
    return build_graph_callback_handler(collector), collector


def run_variant(label: str, use_config: bool, thread_in_config: bool) -> dict:
    EXECUTED.clear()
    agent = build_agent()
    handler, collector = make_handler()
    thread_id = f"probe-{label}"

    kwargs: dict = {"thread_id": thread_id, "track_tool_calls": True}
    if use_config:
        config: dict = {"callbacks": [handler]}
        if thread_in_config:
            config["configurable"] = {"thread_id": thread_id}
        kwargs["config"] = config

    out: dict = {"variant": label}
    try:
        result = asyncio.run(agent.invoke("Call ping and report the exact token.", **kwargs))
        out.update(
            answer_has_token="PONG-4417" in str(result.answer or ""),
            error=result.error,
            sdk_tool_calls=len(result.tool_calls or []),
            tool_body_ran=EXECUTED.count("ping"),
            node_events=sorted({str(e["node"]) for e in collector.events if e.get("node")}),
            node_event_count=len(collector.events),
        )
    except Exception as exc:  # noqa: BLE001 - this is the evidence
        out.update(exception=repr(exc), traceback=traceback.format_exc()[-1500:])
    finally:
        try:
            asyncio.run(agent.aclose())
        except Exception:  # noqa: BLE001
            pass

    # Post-invoke final state read (no re-execution).
    try:
        state = agent.graph.get_state({"configurable": {"thread_id": thread_id}})
        out["final_state_keys"] = sorted(str(k) for k in (state.values or {}).keys())[:12]
        out["final_state_next"] = list(state.next or ())
    except Exception as exc:  # noqa: BLE001
        out["final_state_error"] = repr(exc)
    return out


def main() -> None:
    os.environ.setdefault("SKILLS_ROOT", str(Path(".cuga/skills").resolve()))
    from agent_evolve.cuga_wrapper import RuntimeSettings

    RuntimeSettings.from_env().configure_cuga_environment()

    trials = int(os.environ.get("PROBE_TRIALS", "3"))
    tally: dict[str, list[int]] = {}

    for trial in range(trials):
        for label, use_config, thread_in_config in (
            ("A-baseline-no-config", False, False),
            ("B-config-with-callbacks", True, True),
        ):
            verdict = run_variant(f"{label}-t{trial}", use_config, thread_in_config)
            verdict["trial"] = trial
            print(f"=== {label} trial {trial} ===", flush=True)
            print(json.dumps(verdict, indent=2), flush=True)
            tally.setdefault(label, []).append(int(verdict.get("tool_body_ran") or 0))

    print("=== TOOL EXECUTION RATE (ground truth = tool body ran) ===", flush=True)
    for label, results in tally.items():
        ran = sum(1 for value in results if value > 0)
        print(f"{label}: {ran}/{len(results)} trials executed the tool", flush=True)


if __name__ == "__main__":
    main()
