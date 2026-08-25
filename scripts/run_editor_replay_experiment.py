"""run_editor_replay_experiment.py - REAL D5 editor over LIVE-TAIL replay.

PAID (small): one real CugaEditorAgent turn (multi-tool, provider calls)
plus one live tail invocation after the taped prefix. Guard: AE_LIVE_GO=1.

Closes the "scripted editor" caveat from Phase 6: the diagnosis is handed
over as a CausalAnalysis exactly as the analyzer would produce it, and the
MUTATION (skill content) is authored by the REAL editor agent through its
own tool flow (read evidence -> stage edits -> submit_edit_plan). The
staged writes are then applied to the harness and re-driven through the
taped prefix with a live tail, measuring adoption.

Usage: AE_LIVE_GO=1 python scripts/run_editor_replay_experiment.py [--resume 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_TASK_ID = "complex-20260825-193759"

F1_MECHANISM = (
    "File-authoring steps fail in the cuga_lite sandbox on Windows: no "
    "write_file-style tool exists and bash heredocs die under cmd.exe "
    "('<<' was unexpected at this time; 'cat'/'grep' not recognized), so "
    "the agent burns a cycle on a doomed strategy before recovering with "
    "python -c hex-encoded writes."
)


def _build_request(prompt_text: str):
    from agent_evolve.adapters.cuga_editor_tools import EditMemory
    from agent_evolve.core.analyzer import CausalAnalysis
    from agent_evolve.core.blame import BlameGraph, BlameNode
    from agent_evolve.core.contracts import EvolutionTask
    from agent_evolve.core.editor import CandidateWorkspace, EditorRequest

    analysis = CausalAnalysis(
        mechanism=F1_MECHANISM,
        severity=0.8,
        score=0.8,
        blame_graph=BlameGraph(nodes=(
            BlameNode(
                actor_id="sandbox",
                blame=0.9,
                artifacts=("report.md",),
            ),
        )),
        counterfactual_evidence=(
            "write attempt used a cmd.exe heredoc and failed: '<< was "
            "unexpected at this time.'; 'cat'/'grep' not recognized; the "
            "run only recovered after pivoting to python -c with "
            "bytes.fromhex - a skill teaching that pattern up front should "
            "remove the doomed first attempt entirely.",
        ),
        analyzer_model_id="ox-alpha-free (historical analyzer run)",
    )
    return EditorRequest(
        base_workspace=CandidateWorkspace(
            attempt_id="attempt-replay-1",
            version="v-baseline",
            path=ROOT,
            parent_version="",
        ),
        task=EvolutionTask(
            task_id=SOURCE_TASK_ID,
            input_text=prompt_text,
        ),
        analysis=analysis,
        issue_id="F1-windows-file-authoring",
        write_set=("skills/windows-file-authoring",),
        current_artifacts={},
        creatable_prefixes=("skills/",),
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=int, default=3)
    args = parser.parse_args(argv[1:])

    import os

    if "--go" not in argv and os.environ.get("AE_LIVE_GO") != "1":
        print("PAID experiment (editor turn + live tail): "
              "set AE_LIVE_GO=1 or pass --go.")
        return 1

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    # --- real editor turn -------------------------------------------------
    from agent_evolve.adapters.cuga_editor import (
        CugaEditorAgent,
        prepare_editor_environment,
    )
    from agent_evolve.adapters.cuga_editor_tools import EditMemory

    prepare_editor_environment()

    prompt_text, _notes = _task_constants()
    request = _build_request(prompt_text)

    editor = CugaEditorAgent(
        adapter={"kind": "replay-experiment-driver"},
        memory=EditMemory(),
    )
    print("running REAL CugaEditorAgent.propose_edit ...")
    response = editor.propose_edit(request)

    print("--- editor outcome ---")
    print("outcome:", editor.last_outcome)
    print("tools_called:", editor.last_tools_called)
    print("sdk_tool_calls:",
          [getattr(tc, "name", str(tc))[:40]
           for tc in editor.last_sdk_tool_calls])
    print("rationale:", str(response.rationale)[:400])

    skills: dict[str, str] = {}
    for artifact_id, content in (response.writes or {}).items():
        name = artifact_id.split("/", 1)[-1] if "/" in artifact_id else artifact_id
        skills[name] = str(content)
        print(f"staged write {artifact_id} -> skill '{name}' ({len(content)} chars)")
        print("  preview:", content[:200].replace("\n", " "))

    out_dir = ROOT / "terminal_output" / "node-replay" / "editor-experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "editor-plan.json").write_text(json.dumps({
        "issue_id": request.issue_id,
        "outcome": str(editor.last_outcome),
        "tools_called": list(editor.last_tools_called),
        "sdk_tool_calls": [getattr(tc, "name", str(tc))
                           for tc in editor.last_sdk_tool_calls],
        "rationale": str(response.rationale),
        "writes": dict(response.writes or {}),
        "risks": response.risks,
        "expected_effects": response.expected_effects,
    }, indent=2, default=str), encoding="utf-8")

    if not skills:
        print("editor produced no applicable writes; nothing to re-drive.")
        return 1

    # --- LIVE-TAIL with the editor-authored skill -------------------------
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tail_experiment", ROOT / "scripts" / "run_live_tail_experiment.py")
    tail_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(tail_mod)

    report = tail_mod._run_arm(
        "mutated", args.resume, extra_skills=skills, label="editor-authored")

    print("--- live-tail with editor-authored skill ---")
    print(json.dumps(report, indent=2, default=str))

    summary = {
        "editor": {
            "outcome": str(editor.last_outcome),
            "tools_called": list(editor.last_tools_called),
            "skills_authored": sorted(skills),
        },
        "live_tail": report,
        "baseline_control_ref": "tail-experiment/control/report.json (v2)",
    }
    (out_dir / "experiment-summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"summary written: {out_dir / 'experiment-summary.json'}")
    return 0


def _task_constants() -> tuple[str, str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_live_complex_query", ROOT / "scripts" / "run_live_complex_query.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PROMPT, module.NOTES


if __name__ == "__main__":
    sys.exit(main(sys.argv))
