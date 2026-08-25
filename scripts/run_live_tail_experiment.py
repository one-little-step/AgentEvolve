"""run_live_tail_experiment.py - Phase 6 LIVE-TAIL editor experiment on F1.

PAID (small): the live tail after resume boundary N makes real provider
calls. Guarded by AE_LIVE_GO=1 / --go.

Plays the D5 editor-experiment flow with the editor role scripted
deterministically (the diagnosis is the F1 finding already banked in the
design doc; the skill mutation is hand-authored editor output):

  resume  N=3 of 4 recorded boundaries - taped prefix ends right after the
          heredoc failure evidence has entered state; call 4 (the recovery
          decision + report authoring) goes LIVE.
  control arm    gate stays ON; re-verifies the prefix while the tail runs
                 live. Measures unmutated tail behaviour.
  mutated arm    MUTATION mode (gate deliberately open - an injected skill
                 legitimately changes prompts from call one). Measures whether
                 the live tail ADOPTS the authored Windows file-authoring
                 skill.

Usage:  AE_LIVE_GO=1 python scripts/run_live_tail_experiment.py [--resume 3]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_TRACE = (
    ROOT / "terminal_output" / "live-run-prep" / "traces"
    / "3306905e-668f-41a3-adb0-e7a0ba33e332"
)
SOURCE_TASK_ID = "complex-20260825-193759"
DATETIME_LIKE = __import__("re").compile(
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z)?")

SKILL_WINDOWS_FILE_AUTHORING = """\
## Authoring files on Windows sandboxes

The host shell is Windows cmd.exe. Heredocs (`<<`), `cat`, and `grep` are
UNAVAILABLE and fail with quoting errors. To create a file:

1. Build the content as a Python string, then hex-encode it:
   `payload = content.encode("utf-8").hex()`
2. Write it in ONE step:
   `python -c "open('report.md','wb').write(bytes.fromhex('<payload>'))"`
3. ALWAYS verify by reading back and checking every required header exists.
"""


def _load_task_constants() -> tuple[str, str]:
    spec = importlib.util.spec_from_file_location(
        "run_live_complex_query", ROOT / "scripts" / "run_live_complex_query.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PROMPT, module.NOTES


def _run_arm(arm: str, resume: int, extra_skills: dict[str, str] | None = None,
             label: str | None = None) -> dict:
    from agent_evolve.cuga_wrapper import (
        CugaWrapper,
        RuntimeSettings,
        TraceConfig,
        load_dotenv,
        prepare_cuga_environment,
    )
    from agent_evolve.cuga_wrapper.tape_replay import HybridTapeModel

    prompt, notes = _load_task_constants()
    harness_config: dict[str, object] = {
        "input": prompt,
        "memory": {"meeting-notes": notes},
    }
    if extra_skills:
        harness_config["skills"] = dict(extra_skills)
    elif arm == "mutated":
        harness_config["skills"] = {
            "windows-file-authoring": SKILL_WINDOWS_FILE_AUTHORING,
        }

    prepare_cuga_environment()
    settings = RuntimeSettings.from_env()
    settings.configure_cuga_environment()

    model = HybridTapeModel.from_trace(
        SOURCE_TRACE,
        cutoff=resume,
        scrub_patterns=(DATETIME_LIKE,),
        # MUTATION opens the prefix gates deliberately (RQ4); control keeps
        # them on so the taped prefix stays verified while its tail is live.
        gate_enabled=(arm == "control"),
    )

    def live_factory():
        from cuga.backend.llm.models import LLMManager
        from cuga.config import settings as cuga_settings

        manager = LLMManager()
        manager.clear_pre_instantiated_model()
        return manager.get_model(cuga_settings.agent.code.model)

    model._live_factory = live_factory

    from cuga.backend.llm.models import LLMManager

    manager = LLMManager()
    manager.clear_models()
    manager.set_llm(model)

    out_root = ROOT / "terminal_output" / "node-replay" / "tail-experiment"
    arm_dir_name = label or arm
    trace_config = TraceConfig(enabled=True,
                               output_root=out_root / f"{arm_dir_name}-traces")
    wrapper = CugaWrapper.from_cuga(settings, trace_config)
    wrapper._runtime._workspace_root = out_root / f"{arm_dir_name}-workspaces"

    started = time.time()
    result = wrapper.run_task(SOURCE_TASK_ID, harness_config)
    elapsed = time.time() - started

    trace_dir = result.get("causal_trace_path")
    report = {
        "arm": arm,
        "resume_boundary": resume,
        "status": result.get("status"),
        "elapsed_s": round(elapsed, 1),
        "taped_calls": model.pointer - model.live_calls if model else None,
        "live_calls": model.live_calls if model else None,
        "trace_dir": str(trace_dir) if trace_dir else None,
        "final_output_chars":
            len(str(result.get("final_output", ""))) if result else 0,
    }
    if trace_dir:
        blob_text = "".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in (Path(str(trace_dir)) / "payloads").glob("*.json"))
        report["f1_evidence_heredoc_failure"] = "<< was unexpected" in blob_text
        report["skill_adoption_hex_authoring"] = (
            "bytes.fromhex" in blob_text or "fromhex" in blob_text)

    arm_dir = out_root / arm_dir_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    final = str(result.get("final_output", "")) if result else ""
    (arm_dir / "final_output.txt").write_text(final, encoding="utf-8")
    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=int, default=3)
    args = parser.parse_args(argv[1:])

    if "--go" not in argv and __import__("os").environ.get("AE_LIVE_GO") != "1":
        print("PAID experiment: set AE_LIVE_GO=1 or pass --go.")
        return 1

    print(f"LIVE-TAIL experiment: resume={args.resume}, arms=control+mutated")
    reports = []
    for arm in ("control", "mutated"):
        print(f"--- arm: {arm} ---")
        reports.append(_run_arm(arm, args.resume))
        print(json.dumps(reports[-1], indent=2, default=str))

    summary_path = (ROOT / "terminal_output" / "node-replay" / "tail-experiment"
                    / "experiment-summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "resume_boundary": args.resume,
        "arms": reports,
    }, indent=2, default=str), encoding="utf-8")
    print(f"summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
