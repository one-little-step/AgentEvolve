"""Diagnose WHY a GAIA rollout produces no tool call: unwilling, or unable?

The 10 non-answers in the 42-task baseline all narrate an intent to use a tool
("I'll fetch its page first...") and then apologise for missing data, while the
recorded trace shows **zero** tool observations. Two very different causes fit
that evidence and they demand opposite fixes:

* **UNWILLING** -- the model never emits a fenced Python block. CUGA's
  ``extract_code_from_model_response`` returns "", ``call_model`` takes the
  no-code branch (``shared_nodes.py:233``), and the graph never reaches the
  sandbox. The tools were registered and reachable; nothing was ever asked of
  them. Fix = prompt contract.
* **UNABLE** -- the model *does* emit a block, the sandbox runs it, and the call
  fails (tool missing from context, exception, auth, network). Fix = tool layer.

This script decides between them by instrumenting the exact boundary rather than
inferring from the final answer, which is the mistake that produced the
"insufficient tool results" theory: a model saying "I'm unable to call the tool"
is **not** evidence that calling failed. Our own learnings file records a live
case where that sentence was simply false.

Ground truth captured per turn:
  * ``raw_content``      -- what the model actually returned
  * ``has_fence``        -- did the response contain a ``` fence at all
  * ``extracted_code``   -- what CUGA extracted (the routing input)
  * ``routed_to_sandbox``-- the branch actually taken
  * ``tool_body_ran``    -- set by the tool functions themselves, never inferred

The last point matters: a model will happily narrate "Calling the tool now..."
in a turn where nothing ran, so the tool's own side effect is the only
trustworthy witness.

Usage (needs credentials in .env; costs real inference):
    uv run python scripts/diagnose_tool_invocation.py 2>&1 \
      | tee terminal_output/tool_diagnosis/run.log
"""
from __future__ import annotations


import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# The three GAIA questions whose recorded answers name a tool they never called.
# Verbatim from datasets/gaia/gaia_l1_validation__baseline__20260813_035541.
PROBE_TASKS = {
    "3f57289b": (
        "How many at bats did the Yankee with the most walks in the 1977 regular "
        "season have that same season?"
    ),
    "5188369a": (
        "In Series 9, Episode 11 of Doctor Who, the Doctor is trapped inside an "
        "expanding confession dial. What is the name of the vessel he is on?"
    ),
    "e142056d": (
        "A standard Rubik's cube has been broken into cubes making up its sides. "
        "The cubes are jumbled, and one is removed. There are 6 cubes with one "
        "colored face, 12 edge cubes with two colored faces, and 8 corner cubes "
        "with three colored faces. All blue cubes have been found. You are given "
        "a white cube. What is the probability that the removed cube was red?"
    ),
}

# Arm B: the wording our own learnings file records as one of only two reliable
# phrasings. If UNWILLING is the cause, this arm should flip the outcome with no
# change to the tool layer -- which is itself the proof.
FENCE_DIRECTIVE = (
    "\n\nStart now: make your very next message a single fenced Python block "
    "that awaits the tools you need. Narration without a fenced block executes "
    "nothing, so do not describe a plan before running it. Emit exactly ONE "
    "fenced Python block per turn and print the results."
)


def install_probe(record: list) -> None:
    """Patch CUGA's code-extraction boundary to record every routing decision.

    Wraps rather than replaces, so the real extraction logic (and therefore the
    real routing) is unchanged -- this observes, it does not alter behavior.
    """
    from cuga.backend.cuga_graph.nodes.cuga_agent_core.graph import shared_nodes

    original = shared_nodes.extract_code_from_model_response

    def traced(content, reasoning=None, *args, **kwargs):
        code = original(content, reasoning, *args, **kwargs)
        text = f"{content or ''}\n{reasoning or ''}"
        record.append(
            {
                "turn": len(record) + 1,
                "has_fence": "```" in text,
                "extracted_code_len": len(code or ""),
                "routed_to_sandbox": bool(code),
                "content_tail": (content or "").strip()[-400:],
                "extracted_code": (code or "")[:600],
            }
        )
        return code

    shared_nodes.extract_code_from_model_response = traced


def install_tool_witness(fired: list) -> None:
    """Make each tool record its OWN execution.

    Ground truth for "did the tool run" is the function body executing, never
    the model's narration and never an empty ``tool_calls`` list.

    ``functools.wraps`` is load-bearing, not tidiness: ``@tool`` derives its args
    schema from the signature, which ``inspect.signature`` can only recover by
    following ``__wrapped__`` through a ``*args`` wrapper. A bare wrapper makes
    every tool advertise an empty schema, and CUGA then rejects every call the
    model makes -- which looks exactly like a broken tool layer. The first run of
    this script had that bug and produced 7 different spurious signature errors.

    Idempotent: the module is cached across arms in one process, so re-wrapping
    would stack layers and re-break the signature.
    """
    import functools

    from agent_evolve.cuga_wrapper import tools as tools_mod

    if getattr(tools_mod, "_WITNESS_INSTALLED", False):
        tools_mod._WITNESS_SINK = fired  # type: ignore[attr-defined]
        return

    sink_holder = {"sink": fired}
    tools_mod._WITNESS_SINK = fired  # type: ignore[attr-defined]

    def make(fn, tool_name):
        @functools.wraps(fn)
        def witnessed(*args, **kwargs):
            getattr(tools_mod, "_WITNESS_SINK", sink_holder["sink"]).append(tool_name)
            return fn(*args, **kwargs)

        return witnessed

    wrapped = []
    for fn in tools_mod._RAW_TOOLS:
        w = make(fn, fn.__name__)
        setattr(tools_mod, fn.__name__, w)
        wrapped.append(w)
    # _RAW_TOOLS holds references captured at import, so rebuild it or the
    # witnesses never reach the agent -- the "instrumented callables must
    # actually reach the agent" trap from the learnings file.
    tools_mod._RAW_TOOLS = tuple(wrapped)
    tools_mod._WITNESS_INSTALLED = True  # type: ignore[attr-defined]


def run_arm(question: str, label: str) -> dict:
    from agent_evolve.cuga_wrapper import (
        CugaWrapper,
        RuntimeSettings,
        prepare_cuga_environment,
    )

    prepare_cuga_environment()
    turns: list = []
    fired: list = []
    install_probe(turns)
    install_tool_witness(fired)

    wrapper = CugaWrapper.from_cuga(RuntimeSettings.from_env())
    try:
        # run_task is the real rollout API; ``input`` carries the prompt exactly
        # as HarnessVersion.harness_config builds it, and ``tools`` is omitted so
        # _construct_agent falls back to build_tools() -- the same surface the
        # recorded baseline used.
        result = wrapper.run_task(
            f"diagnose-{label}",
            {"version": "vanilla", "input": question},
        )
        final = str(result.get("final_output") or "")
        status = str(result.get("status") or "")
        reported = result.get("tool_calls") or []
    except Exception as exc:  # noqa: BLE001 - the reason IS the finding
        final, status, reported = f"<exception> {exc!r}", "error", []

    return {
        "arm": label,
        "turns": len(turns),
        "turns_with_fence": sum(1 for t in turns if t["has_fence"]),
        "turns_routed_to_sandbox": sum(1 for t in turns if t["routed_to_sandbox"]),
        "tool_bodies_executed": list(fired),
        "sdk_reported_tool_calls": len(list(reported)) if reported else 0,
        "status": status,
        "final_output_tail": final.strip()[-300:],
        "turn_detail": turns,
    }


def main() -> int:
    out = {"started_at": datetime.now(timezone.utc).isoformat(), "probes": []}

    for task_id, question in PROBE_TASKS.items():
        for label, suffix in (("A_baseline", ""), ("B_fence_directive", FENCE_DIRECTIVE)):
            print(f"\n{'=' * 70}\n{task_id} :: {label}\n{'=' * 70}", flush=True)
            probe = run_arm(question + suffix, f"{task_id}-{label}")
            probe["arm"] = label
            probe["task_id"] = task_id
            out["probes"].append(probe)
            print(
                f"  turns={probe['turns']} "
                f"with_fence={probe['turns_with_fence']} "
                f"routed_to_sandbox={probe['turns_routed_to_sandbox']} "
                f"tools_executed={probe['tool_bodies_executed']} "
                f"sdk_reported={probe['sdk_reported_tool_calls']}",
                flush=True,
            )
            print(f"  final: {probe['final_output_tail'][:200]}", flush=True)

    dest = REPO_ROOT / "terminal_output" / "tool_diagnosis"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "diagnosis.json").write_text(json.dumps(out, indent=2))

    print(f"\n\n{'=' * 70}\nVERDICT\n{'=' * 70}")
    for arm_name in ("A_baseline", "B_fence_directive"):
        arm = [p for p in out["probes"] if p["arm"] == arm_name]
        fenced = sum(p["turns_with_fence"] for p in arm)
        routed = sum(p["turns_routed_to_sandbox"] for p in arm)
        executed = sum(len(p["tool_bodies_executed"]) for p in arm)
        print(
            f"{arm_name:20s} fenced_turns={fenced:3d} "
            f"routed_to_sandbox={routed:3d} tool_bodies_executed={executed:3d}"
        )
    print(
        "\nUNWILLING if fenced_turns==0 in arm A (model never asked);"
        "\nUNABLE   if routed_to_sandbox>0 but tool_bodies_executed==0."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
