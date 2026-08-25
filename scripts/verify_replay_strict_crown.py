"""verify_replay_strict_crown.py - Phase 5 STRICT replay crown (R5 amended).

Re-drives a reference trace's exact task through CUGA with every LLM response
served from tape (TapeModel injected via the documented ``LLMManager.set_llm``
override). ZERO provider spend: all model traffic is local tape serving; the
only live work is deterministic sandbox execution plus the local Ollama
embedder behind ``knowledge_search``.

Verdict levels (design R5 amendment):
  verified_raw         every checkpoint byte-equal with NO scrubbing applied
  verified_normalized  every checkpoint equal under the declared volatility
                       scrub registry (wall-clock stamps, task id)
  diverged             some checkpoint fails even after scrubbing

Usage:
    python scripts/verify_replay_strict_crown.py [source_trace_dir]

Writes crown-report.json next to the replay trace and exits non-zero unless
a verified_* verdict is reached.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "terminal_output" / "live-run-prep" / "traces"
    / "3306905e-668f-41a3-adb0-e7a0ba33e332"
)
SOURCE_TASK_ID = "complex-20260825-193759"

# Volatility registry (R5 amendment): patterns scrubbed identically on BOTH
# sides before comparison. Data here, not buried logic.
DATETIME_LIKE = re.compile(
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z)?")
SCRUBS = [DATETIME_LIKE]


def _load_task_constants() -> tuple[str, str]:
    spec = importlib.util.spec_from_file_location(
        "run_live_complex_query", ROOT / "scripts" / "run_live_complex_query.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PROMPT, module.NOTES


def _scrub(text: str, extra_literals: list[str]) -> str:
    for pattern in SCRUBS:
        text = pattern.sub("<volatile>", text)
    for literal in extra_literals:
        text = text.replace(literal, "<task>")
    return text


def main(argv: list[str]) -> int:
    source_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_SOURCE
    if not source_dir.exists():
        print(f"source trace not found: {source_dir}")
        return 2

    started = time.time()
    from agent_evolve.cuga_wrapper import (
        CugaWrapper,
        RuntimeSettings,
        TraceConfig,
        load_dotenv,
        prepare_cuga_environment,
    )
    from agent_evolve.cuga_wrapper.tape_replay import (
        TapeCallSequenceExhausted,
        TapeDivergence,
        load_tape_model,
    )
    from agent_evolve.core.tape import TapeIndex

    load_dotenv(ROOT / ".env")
    prompt, notes = _load_task_constants()
    harness_config = {"input": prompt, "memory": {"meeting-notes": notes}}
    extra_literals = [SOURCE_TASK_ID, source_dir.name]

    # --- tape -----------------------------------------------------------
    tape = load_tape_model(source_dir, scrub_patterns=tuple(SCRUBS))
    expected_calls = len(tape.entries)
    print(f"tape loaded: {expected_calls} recorded call(s) from {source_dir.name}")

    # --- environment ----------------------------------------------------
    prepare_cuga_environment()
    settings = RuntimeSettings.from_env()
    settings.configure_cuga_environment()

    from cuga.backend.llm.models import LLMManager

    manager = LLMManager()
    manager.clear_models()
    manager.set_llm(tape)

    out_root = ROOT / "terminal_output" / "node-replay"
    trace_config = TraceConfig(enabled=True, output_root=out_root / "replay-traces")
    wrapper = CugaWrapper.from_cuga(settings, trace_config)
    # R7 isolation: fresh workspace root, never the original workspace.
    wrapper._runtime._workspace_root = out_root / "replay-workspaces"

    # --- re-drive -------------------------------------------------------
    failure: str | None = None
    try:
        result = wrapper.run_task(SOURCE_TASK_ID, harness_config)
    except (TapeDivergence, TapeCallSequenceExhausted) as exc:
        result = None
        failure = f"{type(exc).__name__}: {exc}"
        print(f"RUNTIME DIVERGENCE: {failure}")

    elapsed = time.time() - started
    replay_dir = None
    if result is not None:
        replay_dir = result.get("causal_trace_path")
        print(f"replay status: {result.get('status')} elapsed: {elapsed:.1f}s")

    # --- checkpoints -----------------------------------------------------
    report: dict[str, object] = {
        "source": str(source_dir),
        "replay": str(replay_dir) if replay_dir else None,
        "expected_calls": expected_calls,
        "scrub_registry": [p.pattern for p in SCRUBS] + [
            f"literal:{lit}" for lit in extra_literals],
        "runtime_failure": failure,
    }

    runtime_ok = failure is None and result is not None \
        and str(result.get("status")) == "success"
    consumed = getattr(tape, "pointer", None)
    report["calls_consumed"] = consumed
    report["all_calls_consumed"] = bool(runtime_ok and consumed == expected_calls)

    gates_raw = gates_scrubbed = None
    nodes_aligned = None
    final_raw = final_scrubbed = None

    if replay_dir:
        src_index = TapeIndex.load(source_dir)
        rep_index = TapeIndex.load(Path(replay_dir))

        src_bounds = src_index.llm_boundaries
        rep_bounds = rep_index.llm_boundaries
        if len(src_bounds) != len(rep_bounds):
            gates_raw = gates_scrubbed = 0
            report["gate_note"] = (
                f"boundary count mismatch: {len(src_bounds)} vs {len(rep_bounds)}")
        else:
            gates_raw = gates_scrubbed = 0
            for sb, rb in zip(src_bounds, rep_bounds):
                a = src_index.resolve(sb.messages_ref).decode("utf-8")
                b = rep_index.resolve(rb.messages_ref).decode("utf-8")
                if a == b:
                    gates_raw += 1
                if _scrub(a, extra_literals) == _scrub(b, extra_literals):
                    gates_scrubbed += 1

        src_nodes = [(n.node, n.step) for n in src_index.node_starts]
        rep_nodes = [(n.node, n.step) for n in rep_index.node_starts]
        nodes_aligned = src_nodes == rep_nodes
        report["node_sequence_source"] = src_nodes
        report["node_sequence_replay"] = rep_nodes

        src_doc = json.loads(
            (source_dir / "causal-trace.json").read_text(encoding="utf-8"))
        src_final = str(src_doc.get("final_output", ""))
        rep_final = str(result.get("final_output", "")) if result else ""
        final_raw = src_final == rep_final
        final_scrubbed = _scrub(src_final, extra_literals) == \
            _scrub(rep_final, extra_literals)

    report["gates_total"] = expected_calls
    report["gates_raw_passed"] = gates_raw
    report["gates_scrubbed_passed"] = gates_scrubbed
    report["node_sequence_aligned"] = nodes_aligned
    report["final_output_raw_equal"] = final_raw
    report["final_output_scrubbed_equal"] = final_scrubbed

    if not runtime_ok:
        verdict = "diverged"
    elif (report["all_calls_consumed"] and nodes_aligned
          and final_raw and gates_raw == expected_calls):
        verdict = "verified_raw"
    elif (report["all_calls_consumed"] and nodes_aligned
          and final_scrubbed and gates_scrubbed == expected_calls):
        verdict = "verified_normalized"
    else:
        verdict = "diverged"
    report["verdict"] = verdict

    report_path = None
    if replay_dir:
        report_path = Path(str(replay_dir)) / "crown-report.json"
        report_path.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("---")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("node_sequence_source", "node_sequence_replay")},
                     indent=2, default=str))
    print(f"verdict: {verdict}  ({time.time() - started:.1f}s total)")
    if report_path:
        print(f"report written: {report_path}")
    return 0 if verdict.startswith("verified") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
