"""Live end-to-end proof that a structured edit changes real CUGA behavior.

Everything here is real: real CUGA SDK, real model inference, real filesystem
workspaces, real causal traces, real core analyzer. The only thing constructed
for the test is the harness content itself.

WHY A DIFFERENTIAL TEST
-----------------------
A single live run cannot prove an edit works. If we injected one token and saw
it echoed, that could be luck, prompt sensitivity, or the model complying with
the task text rather than the artifact. Earlier sessions established that CUGA
tool execution is NOT reliably deterministic per prompt wording, so a single
observation proves little.

So we run the SAME task twice with the SAME wording, changing ONLY the skill
body via ``apply_structured_edits``:

    candidate A (base)   -> skill demands signature SKL-<tokenA>
    candidate B (edited) -> skill demands signature SKL-<tokenB>

Both tokens are random at runtime, so the model cannot know either from
pretraining, and neither appears in the task input. The decisive evidence is
*exclusivity*:

    tokenA in answer A and NOT in answer B, and
    tokenB in answer B and NOT in answer A

That pattern is only reachable if the edited artifact actually reached the
model on each run. Echoing, guessing, or ignoring the artifact all fail it.

The full adapter path is exercised: register_candidate -> materialize_candidate
-> apply_structured_edits -> run_full_rollout -> capture_trace -> analyze.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

OUT = Path("terminal_output/cuga-adapter/e2e")
OUT.mkdir(parents=True, exist_ok=True)

TOKEN_A = f"SKL-{random.randint(10**9, 10**10 - 1)}"
TOKEN_B = f"SKL-{random.randint(10**9, 10**10 - 1)}"

# The task never mentions a token, so the signature can only come from the
# skill artifact that CUGA loaded for that specific candidate.
TASK_INPUT = (
    "Report the AgentEvolve project status in two sentences. "
    "Apply your status-report skill exactly, including its required signature line."
)


def skill_body(token: str) -> str:
    return (
        "Use this skill whenever you are asked for a project status report.\n\n"
        "You MUST end every status report with this exact line:\n"
        f"SKILL-SIGNATURE: {token}\n"
    )


def summarize_trace(trace_dir: Path) -> dict[str, object]:
    causal = json.loads((trace_dir / "causal-trace.json").read_text())
    events = causal["events"]
    return {
        "trace_dir": str(trace_dir),
        "event_count": len(events),
        "events_with_parent": sum(1 for e in events if e.get("parent_event_id")),
        "distinct_actors": sorted({e["actor_id"] for e in events if e.get("actor_id")}),
        "payload_blobs": len(list((trace_dir / "payloads").glob("*.json")))
        if (trace_dir / "payloads").is_dir()
        else 0,
        "capabilities": causal.get("capabilities", {}),
        "status": causal.get("status"),
    }


def main() -> int:
    from agent_evolve.adapters.cuga_adapter import CugaAdapter
    from agent_evolve.core.analyzer import FakeAnalyzerJudge
    from agent_evolve.core.contracts import ArtifactEdit, EvolutionTask
    from agent_evolve.core.trace import PayloadLevel
    from agent_evolve.cuga_wrapper import CugaWrapper, RuntimeSettings, TraceConfig

    trace_config = TraceConfig(
        enabled=True,
        output_root=Path("data/traces"),
        payload_level=PayloadLevel.RAW_OPT_IN,
        allow_raw_payloads=True,
        capture_node_payloads=True,
        max_observation_bytes=4_194_304,
    )
    wrapper = CugaWrapper.from_cuga(RuntimeSettings.from_env(), trace_config=trace_config)
    adapter = CugaAdapter(wrapper)

    # Seed the base candidate exactly as a future SeedGenerator / RHO stage would.
    adapter.register_candidate("base", {"skills/status-report": skill_body(TOKEN_A)})

    print("=== SETUP ===")
    print(f"token A (base)   : {TOKEN_A}")
    print(f"token B (edited) : {TOKEN_B}")
    print(f"task input       : {TASK_INPUT}")

    print("\n=== ARTIFACT INVENTORY (base) ===")
    for d in adapter.artifact_inventory("base"):
        print(f"  {d.artifact_id}  kind={d.kind}  writable={d.writable}  {d.version_hash}")

    # ---------------- candidate A: unedited base ---------------- #
    workspace_a = adapter.materialize_candidate("base", "attempt-A")
    task_a = EvolutionTask(
        task_id="e2e-candidate-A",
        input_text=TASK_INPUT,
        expected_contract={"expected_substring": TOKEN_A},
    )
    print("\n=== ROLLOUT A (base skill, live CUGA) ===")
    result_a = adapter.run_full_rollout(workspace_a, task_a, "rollout-A")
    trace_a = adapter.capture_trace(result_a)
    answer_a = trace_a.final_output

    # ---------------- candidate B: edited skill ---------------- #
    workspace_b = adapter.materialize_candidate("base", "attempt-B")
    edited = adapter.apply_structured_edits(
        workspace_b,
        [
            ArtifactEdit(
                artifact_id="skills/status-report",
                operation="replace",
                payload={"content": skill_body(TOKEN_B)},
            )
        ],
    )
    assert TOKEN_B in edited["skills/status-report"]
    task_b = EvolutionTask(
        task_id="e2e-candidate-B",
        input_text=TASK_INPUT,
        expected_contract={"expected_substring": TOKEN_B},
    )
    print("=== ROLLOUT B (edited skill, live CUGA) ===")
    result_b = adapter.run_full_rollout(workspace_b, task_b, "rollout-B")
    trace_b = adapter.capture_trace(result_b)
    answer_b = trace_b.final_output

    # ---------------- on-disk workspace evidence ---------------- #
    ws_root = Path("data/workspaces")
    disk = {}
    for label, task_id, token in (("A", "e2e-candidate-A", TOKEN_A), ("B", "e2e-candidate-B", TOKEN_B)):
        skill_file = ws_root / task_id / "skills" / "status-report" / "SKILL.md"
        if skill_file.is_file():
            body = skill_file.read_text()
            disk[label] = {
                "path": str(skill_file),
                "contains_own_token": token in body,
                "contains_other_token": (TOKEN_B if label == "A" else TOKEN_A) in body,
            }
        else:
            disk[label] = {"path": str(skill_file), "missing": True}

    # ---------------- verdict ---------------- #
    verdict = {
        "answer_A_has_tokenA": TOKEN_A in answer_a,
        "answer_A_has_tokenB": TOKEN_B in answer_a,
        "answer_B_has_tokenA": TOKEN_A in answer_b,
        "answer_B_has_tokenB": TOKEN_B in answer_b,
    }
    verdict["edit_changed_behavior_exclusively"] = (
        verdict["answer_A_has_tokenA"]
        and not verdict["answer_A_has_tokenB"]
        and verdict["answer_B_has_tokenB"]
        and not verdict["answer_B_has_tokenA"]
    )

    summary_a = summarize_trace(Path(str(result_a["trace"]["causal_trace_path"])))
    summary_b = summarize_trace(Path(str(result_b["trace"]["causal_trace_path"])))

    analysis_a = FakeAnalyzerJudge().analyze(task_a, trace_a)
    analysis_b = FakeAnalyzerJudge().analyze(task_b, trace_b)

    def blame(analysis):
        return {
            "score": analysis.score,
            "mechanism": analysis.mechanism,
            "nodes": [
                {"actor_id": n.actor_id, "blame": round(n.blame, 4)}
                for n in analysis.blame_graph.nodes
            ],
            "has_unknown_placeholder": any(
                n.actor_id == "unknown" for n in analysis.blame_graph.nodes
            ),
        }

    report = {
        "tokens": {"A": TOKEN_A, "B": TOKEN_B},
        "task_input": TASK_INPUT,
        "harness_sent_A": {
            k: v for k, v in result_a["trace"].items() if k in ("active_artifacts",)
        },
        "verdict": verdict,
        "answers": {"A": answer_a, "B": answer_b},
        "on_disk_workspace": disk,
        "trace_A": summary_a,
        "trace_B": summary_b,
        "blame_A": blame(analysis_a),
        "blame_B": blame(analysis_b),
        "adapter_core_mapping": {
            "A_events": len(trace_a.events),
            "A_actors": sorted({e.actor_id for e in trace_a.events if e.actor_id}),
            "A_parent_edges": sum(1 for e in trace_a.events if e.parent_event_id),
            "B_events": len(trace_b.events),
            "B_actors": sorted({e.actor_id for e in trace_b.events if e.actor_id}),
            "B_parent_edges": sum(1 for e in trace_b.events if e.parent_event_id),
        },
    }

    # Payload-leak audit on the real traces reaching the core.
    ref_keys = {"state_before_ref", "state_after_ref", "messages_ref", "response_ref"}
    leaks = []
    for label, trace in (("A", trace_a), ("B", trace_b)):
        for event in trace.events:
            for key, value in dict(event.payload).items():
                text = value if isinstance(value, str) else json.dumps(value)
                if key in ref_keys and len(text) != 64:
                    leaks.append(f"{label}:{event.event_id}:{key}:len={len(text)}")
                elif key not in ref_keys and key != "tool_call" and len(text) > 64:
                    leaks.append(f"{label}:{event.event_id}:{key}:oversized={len(text)}")
    report["payload_leak_violations"] = leaks

    print("\n=== ANSWER A ===")
    print(answer_a)
    print("\n=== ANSWER B ===")
    print(answer_b)
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))
    print("\n=== FULL REPORT ===")
    print(json.dumps(report, indent=2, default=str))

    (OUT / "e2e-report.json").write_text(json.dumps(report, indent=2, default=str))
    for label, trace_dir in (("A", summary_a["trace_dir"]), ("B", summary_b["trace_dir"])):
        dest = OUT / f"trace-{label}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(trace_dir, dest)
        print(f"copied trace {label} -> {dest}")

    ok = verdict["edit_changed_behavior_exclusively"] and not leaks
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
