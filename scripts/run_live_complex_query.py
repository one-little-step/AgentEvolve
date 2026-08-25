"""Run one complex LIVE query through CugaWrapper -- the inference harness itself.

Purpose (?live-complex-01): exercise the harness that will be evolved, on a
task that needs multiple tool calls AND real reasoning, with every trace
facility enabled, so the persisted trace can be inspected before building
node-level replay on top of it.

COST GUARD: this performs paid model calls. It refuses to run unless
AE_LIVE_GO=1 is set in the environment or --go is passed.

Task design (offline-safe, no browser/network):
* a meeting-notes memory document with TWO planted numeric inconsistencies;
* deliverables that force: knowledge retrieval, cross-section arithmetic,
  file authoring via thread-scoped filesystem tools, read-back verification.

Output:
* stdout -> tee to terminal_output/live-run-prep/live-complex-query.log
* full causal trace under terminal_output/live-run-prep/traces/<run_id>/
* an INDEX.md written next to the trace for human browsing.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

NOTES = """# Program sync - raw notes

## Platform section
- Retrieval service p95 latency at 4.1s; target 2s. Owner: planner team.
- 3 incidents this month traced to stale skill caches.
- Headcount: 12 engineers on the platform team as of this quarter.
- Code agent sandbox timeout raised to 900s after batch failures.
- Knowledge store flock contention observed during parallel rollouts.

## Finance section
- Budget approved for Q3: 120000 USD total.
  - Infra: 40000 USD
  - Tooling: 25000 USD
  - Contractor support: 30000 USD
  - Contingency: 10000 USD
- Q3 platform hiring plan assumes 14 engineers on the platform team.
- Spend review scheduled end of September.

## General
- Evaluation harness migrated to Windows laptop; baseline re-established.
- Positivity judge live-verified; abstention fired unprompted once.
- Action item: reconcile budget figures before finance review.
- Action item: confirm platform headcount with HR before requisitions.
"""

PROMPT = f"""You have access to an ingested memory document 'meeting-notes'.
Complete ALL FOUR steps using your tools where applicable:

1. RISK THEMES: identify the three most frequent risk/incident themes across
   ALL sections of the notes, with a count of supporting note lines each.
2. CONSISTENCY AUDIT: find every NUMERIC inconsistency between the Platform
   section and the Finance section. For each: quote both conflicting values,
   state which one is wrong, and give the corrected value with reasoning.
3. AUTHOR REPORT: create a file named report.md in your workspace root with
   exactly these section headers: '# Risk Themes', '# Inconsistencies',
   '# Summary' (summary max 5 lines).
4. VERIFY: read report.md back from disk and list which of the three required
   headers are present.

Do not narrate results without acting: use the filesystem tools to actually
create and re-read the file."""


def _guard() -> bool:
    if "--go" in sys.argv or os.environ.get("AE_LIVE_GO") == "1":
        return True
    print("REFUSING: this probe performs PAID model calls.")
    print("Re-run with AE_LIVE_GO=1 (env) or --go (flag).")
    return False


def _write_index(trace_dir: Path, result: dict[str, object]) -> None:
    lines = ["# Live complex-query trace index", ""]
    causal = json.loads((trace_dir / "causal-trace.json").read_text(encoding="utf-8"))
    caps = causal.get("capabilities") or {}
    lines.append("## Capabilities")
    for name, cap in sorted(caps.items()):
        status = cap.get("status") if isinstance(cap, dict) else str(cap)
        lines.append(f"- {name}: {status}")
    lines += ["", "## Events (sequence | kind | actor | payload refs)"]
    events_path = trace_dir / "events.jsonl"
    if events_path.exists():
        for i, line in enumerate(events_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            ev = json.loads(line)
            payload = ev.get("payload") or {}
            refs = [
                v for k, v in payload.items()
                if isinstance(v, str) and (k.endswith("_ref") or k.endswith("_refs"))
            ]
            actor = ev.get("actor_id") or "-"
            lines.append(
                f"{i:04d} | {ev.get('kind')} | {actor} | "
                f"keys={sorted(k for k in payload)} refs={refs}"
            )
    else:
        lines.append("(no events.jsonl)")
    payloads = trace_dir / "payloads"
    lines += ["", "## Payload blobs"]
    if payloads.is_dir():
        for blob in sorted(payloads.iterdir()):
            lines.append(f"- {blob.name} ({blob.stat().st_size} bytes)")
    else:
        lines.append("(none)")
    out = trace_dir / "INDEX.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"index written: {out}")


def main() -> int:
    if not _guard():
        return 1
    load_dotenv(ROOT / ".env")

    from agent_evolve.cuga_wrapper import (
        CugaWrapper,
        RuntimeSettings,
        TraceConfig,
        prepare_cuga_environment,
    )

    prepare_cuga_environment()
    settings = RuntimeSettings.from_env()
    settings.configure_cuga_environment()

    out_root = ROOT / "terminal_output" / "live-run-prep"
    trace_config = TraceConfig(enabled=True, output_root=out_root / "traces")
    wrapper = CugaWrapper.from_cuga(settings, trace_config)

    task_id = f"complex-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"model={settings.model} task_id={task_id}")
    started = time.time()
    result = wrapper.run_task(
        task_id,
        {
            "input": PROMPT,
            "memory": {"meeting-notes": NOTES},
        },
    )
    elapsed = time.time() - started

    print("---")
    print(f"status: {result.get('status')}  elapsed: {elapsed:.1f}s")
    # The wrapper's result dict omits the runtime error string; it only lands
    # in the persisted causal trace. Read it back so failures are visible.
    trace_path = result.get("causal_trace_path")
    if result.get("status") != "success" and trace_path:
        causal = json.loads((Path(str(trace_path)) / "causal-trace.json").read_text(encoding="utf-8"))
        err = causal.get("error")
        if err:
            print(f"error: {err}")
    output = str(result.get("final_output", ""))
    print(f"final_output ({len(output)} chars):")
    print(output[:600] + ("..." if len(output) > 600 else ""))
    print(f"causal_trace_path: {trace_path}")
    if trace_path:
        _write_index(Path(str(trace_path)), result)
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
