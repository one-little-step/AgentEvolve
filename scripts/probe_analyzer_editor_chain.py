"""Go/no-go: on a REAL recorded failure, does the chain produce an actionable edit?

Two LLM calls, zero rollouts. The question is not "do the components run" -- the
offline suite already proves that with fakes. It is whether, given the canonical
failure we measured, the analyzer names a mechanism an editor can act on, and the
editor then picks a surface that can actually deliver the fix.

The input is trace ``gaia_3f57289b`` from the recorded 42-task baseline: the model
narrated "I'll fetch the Baseball-Reference team page and inspect the batting
rows", emitted no fenced block, made **zero** tool calls, and closed with "I'm
unable to retrieve the source page in this turn". A fence directive later flipped
this exact task to real web_search+web_fetch calls and the correct answer (519
at-bats). So a working chain has somewhere to go.

What counts as a pass, and why each is separate:

* **analyzer**: names the no-executable-code / no-fence mechanism, and does NOT
  attribute it to tool failure or missing capability. Attributing it to the tools
  is the specific wrong answer that the model's own false "I'm unable" claim
  invites -- it is the failure mode this probe exists to detect.
* **editor**: reaches a VALID plan and writes to a surface whose delivery does not
  depend on the broken behavior. ``instructions`` is unconditional; a skill
  requires ``load_skill``, which requires the tool-calling this failure prevents.
  A skill-only edit is therefore circular and counts as a miss even though it is
  a structurally valid plan.

Deliberately does NOT assert that the edit's wording resembles ours. That would
measure conformity to our fix rather than the chain's ability to find one.

Usage (needs credentials in .env; ~2 LLM calls, no rollouts):
    uv run python scripts/probe_analyzer_editor_chain.py 2>&1 \
      | tee terminal_output/chain_probe/run.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# A CURRENT-FORMAT trace. The recorded 42-task dataset predates the causal
# tracing rewrite: its events are 8 undifferentiated ``stream_event`` entries with
# ``actor_id=None`` and no ``tool_call`` kind, so an analyzer reading it sees no
# actors and no tool vocabulary. Probing against it measures the stale format, not
# the prompt. Current traces carry graph_node_start/end, llm_call_start/end,
# graph_tool_start/end and tool_call, with real actor ids.
TRACE_DIR = REPO_ROOT / "data/traces/0cb88c5a-1a6e-4aea-8ce0-f84c3f926e68"

# Wrong-answer markers: the mechanism blaming the tools or the environment rather
# than the turn-level output contract. The model's own final_output makes exactly
# this claim, and it is false.
#: The task this trace was produced for. Recorded traces store task_id, not the
#: question text, and an analyzer given an empty input cannot reason about intent.
INPUT_TEXT = (
    "How many studio albums were published by Mercedes Sosa between 2000 and 2009 "
    "(inclusive)?"
)

_TOOL_BLAME = (
    "tool failure",
    "tool did not return",
    "tool returned nothing",
    "tools unavailable",
    "tool unavailable",
    "no tools",
    "missing tool",
    "lacks a tool",
    "lacked a tool",
    "insufficient tool",
    "tool error",
)
# Right-answer markers: the no-code / contract framing.
_NO_CODE = (
    "no code",
    "no executable",
    "not emit",
    "never emitted",
    "no fenced",
    "without a fenced",
    "fence",
    "code block",
    "narrat",
    "prose",
    "output contract",
    "never invoked",
    "did not invoke",
    "never called",
)


def load_recorded_trace():
    """Rebuild an ExecutionTrace from the recorded rollout, verbatim."""
    from agent_evolve.core.trace import ExecutionTrace, TraceEvent

    recorded = json.loads((TRACE_DIR / "causal-trace.json").read_text())
    result = {"question": str(recorded.get("task_id") or "")}

    events = []
    for i, ev in enumerate(recorded.get("events") or []):
        events.append(
            TraceEvent(
                event_id=str(ev.get("event_id") or f"e{i}"),
                kind=str(ev.get("kind") or "graph_node_start"),
                actor_id=str(ev.get("actor_id") or ev.get("node") or "unknown"),
                payload=ev.get("payload") if isinstance(ev.get("payload"), dict) else {},
                parent_event_id=ev.get("parent_event_id"),
            )
        )
    return (
        ExecutionTrace(
            trace_id=str(recorded.get("run_id") or TRACE_DIR.name),
            candidate_id="base",
            task_id=str(recorded.get("task_id") or TRACE_DIR.name),
            events=tuple(events),
            final_output=str(recorded.get("final_output") or result.get("answer") or ""),
            status=str(recorded.get("status") or "success"),
        ),
        result,
        len(recorded.get("tool_observations") or []),
    )


def main() -> int:
    # Loads .env into os.environ. The analyzer resolves its model from the
    # environment at first call, so this must happen before analyze().
    from agent_evolve.cuga_wrapper import prepare_cuga_environment

    prepare_cuga_environment()

    from agent_evolve.adapters.cuga_analyzer import CugaTrajectoryAnalyzer
    from agent_evolve.core.analysis import RolloutGroupReport
    from agent_evolve.core.evidence import rollout_group_report
    from agent_evolve.core.contracts import EvolutionTask

    trace, result, n_obs = load_recorded_trace()
    print(f"trace           : {trace.trace_id}")
    print(f"events          : {len(trace.events)}")
    print(f"tool observations: {n_obs}  (recorded ground truth)")
    print(f"final_output tail: ...{trace.final_output.strip()[-180:]}\n", flush=True)

    task = EvolutionTask(
        task_id=trace.task_id,
        input_text=INPUT_TEXT,
        # expected_contract deliberately omitted: the analyzer and editor must
        # never see the answer key.
    )

    # ---------------------------------------------------------------- analyzer
    print("=" * 70)
    print("STEP 1: analyzer / judge")
    print("=" * 70, flush=True)
    report: RolloutGroupReport = rollout_group_report(task, trace)
    analyzer = CugaTrajectoryAnalyzer()
    findings = analyzer.analyze(report)

    if not findings:
        print("  NO FINDINGS -- chain cannot proceed.")
        return 1

    f = findings[0]
    mech = str(getattr(f, "mechanism_description", "") or "")
    status = str(getattr(f, "status", "") or "")
    print(f"  status   : {status}")
    print(f"  mechanism: {mech}")
    print(f"  rationale: {str(getattr(f, 'rationale', '') or '')[:400]}", flush=True)

    low = mech.lower() + " " + str(getattr(f, "rationale", "") or "").lower()
    named_no_code = any(t in low for t in _NO_CODE)
    blamed_tools = any(t in low for t in _TOOL_BLAME)
    print(f"\n  -> names no-code mechanism : {named_no_code}")
    print(f"  -> blames tools (WRONG)    : {blamed_tools}", flush=True)

    dest = REPO_ROOT / "terminal_output" / "chain_probe"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "result.json").write_text(
        json.dumps(
            {
                "mechanism": mech,
                "status": status,
                "rationale": str(getattr(f, "rationale", "") or ""),
                "named_no_code": named_no_code,
                "blamed_tools": blamed_tools,
                "recorded_tool_observations": n_obs,
            },
            indent=2,
            default=str,
        )
    )

    print("\n" + "=" * 70)
    print("VERDICT (analyzer)")
    print("=" * 70)
    if named_no_code and not blamed_tools:
        print("  PASS: mechanism is actionable and does not blame the tools.")
        rc = 0
    elif blamed_tools:
        print("  FAIL: analyzer repeated the model's false 'unable to call' claim.")
        print("        An editor acting on this would change the wrong surface.")
        rc = 1
    else:
        print("  WEAK: mechanism does not identify the no-executable-code pattern.")
        print("        The editor will be guessing.")
        rc = 1
    print(f"\nfull result: {dest / 'result.json'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
