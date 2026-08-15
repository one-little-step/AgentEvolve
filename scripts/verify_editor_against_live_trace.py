"""One real editor-agent invocation over the reference live trace.

This answers the design's highest tracked risk (§13): does the model actually
call the editor tools? A model that never calls submit_edit_plan makes the
editor inert, and that must be visible immediately rather than after a full
experiment.

Reports: tools called, outcome classification, staged edits, parents read, and
whether the contamination guard fired. Makes ONE live inference.

Usage:
    uv run python scripts/verify_editor_against_live_trace.py \
        2>&1 | tee terminal_output/cuga-editor/live/editor-run.log
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_evolve.adapters.cuga_adapter import CugaAdapter  # noqa: E402
from agent_evolve.adapters.cuga_editor import (  # noqa: E402
    CugaEditorAgent,
    EditorDeclined,
)
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    CandidateWorkspace,
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)
from agent_evolve.core.editor import EditorRequest, ParentContext  # noqa: E402
from agent_evolve.core.memory import EditMemory  # noqa: E402
from agent_evolve.cuga_wrapper import (  # noqa: E402
    CugaWrapper,
    InMemoryRuntime,
    RuntimeSettings,
    prepare_cuga_environment,
)

# Load .env and normalize blank CUGA_CONFIGURATIONS_DIR BEFORE the SDK is
# imported: an empty value makes CUGA resolve model config as a relative
# path and fail at import, before any agent exists.
prepare_cuga_environment()

TRACE_DIR = ROOT / "data/traces/5d434903-bc26-4dc4-9229-8d886d2c6781"
REPORT_DIR = ROOT / "terminal_output/cuga-editor/live"


def load_reference_trace() -> ExecutionTrace:
    """Map the persisted causal trace into an ExecutionTrace."""
    causal = json.loads((TRACE_DIR / "causal-trace.json").read_text())
    events = tuple(
        TraceEvent(
            event_id=str(e["event_id"]),
            kind=str(e["kind"]),
            actor_id=(str(e["actor_id"]) if e.get("actor_id") else None),
            parent_event_id=(
                str(e["parent_event_id"]) if e.get("parent_event_id") else None
            ),
            payload=dict(e.get("payload") or {}),
        )
        for e in causal["events"]
    )
    return ExecutionTrace(
        trace_id="live-reference",
        candidate_id="cand-primary",
        task_id="reference-task",
        events=events,
        final_output="",
        status="completed",
    )


def build_request(adapter: CugaAdapter) -> EditorRequest:
    task = EvolutionTask(
        task_id="reference-task",
        input_text=(
            "Fetch the alpha token, exchange it for a beta token, then report "
            "the beta checksum."
        ),
    )
    analysis = CausalAnalysis(
        mechanism="the agent reported a final answer without verifying the checksum",
        severity=0.9,
        score=0.0,
        blame_graph=BlameGraph(
            nodes=(
                BlameNode(actor_id="call_model", blame=0.7,
                          artifacts=("skills/token-workflow",)),
                BlameNode(actor_id="FinalAnswerAgent", blame=0.3, artifacts=()),
            )
        ),
    )
    return EditorRequest(
        base_workspace=CandidateWorkspace(
            "live-att-1", "v-primary", Path("."), "v0"
        ),
        task=task,
        analysis=analysis,
        issue_id="live-issue-1",
        write_set=("skills/token-workflow",),
        current_artifacts={
            "skills/token-workflow": (
                "# Token workflow\n\n"
                "1. Call fetch_alpha_token.\n"
                "2. Call exchange_alpha_for_beta.\n"
                "3. Report the result.\n"
            )
        },
        creatable_prefix=CugaAdapter.creatable_prefix,
        parents=(
            ParentContext("cand-primary", "v-primary", True, {"reference-task": 0.0}),
            ParentContext("cand-donor", "v-donor", False, {"reference-task": 0.8}),
        ),
    )


def main() -> int:
    if not TRACE_DIR.is_dir():
        print(f"FAIL: reference trace missing at {TRACE_DIR}")
        return 1
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    adapter = CugaAdapter(
        wrapper=CugaWrapper(
            InMemoryRuntime(),
            RuntimeSettings(model=os.getenv("CUGA_MODEL", "openai/azure/gpt-5.6-luna")),
        )
    )
    adapter.register_candidate(
        "v-primary",
        {"skills/token-workflow": "# Token workflow\n\n1. Fetch.\n2. Report.\n"},
    )
    adapter.register_candidate(
        "v-donor",
        {
            "skills/token-workflow": (
                "# Token workflow\n\n"
                "1. Fetch alpha.\n2. Exchange for beta.\n"
                "3. Verify the checksum before reporting.\n"
            )
        },
    )

    editor = CugaEditorAgent(
        adapter=adapter, memory=EditMemory(), trace=load_reference_trace()
    )
    request = build_request(adapter)

    outcome = "unknown"
    edits: list[dict[str, object]] = []
    rationale = ""
    error = ""
    try:
        response = editor.propose_edit(request)
        outcome = editor.last_outcome.value
        rationale = response.rationale
        edits = [
            {
                "artifact_id": e.artifact_id,
                "operation": e.operation,
                "content_length": len(str(e.payload.get("content", ""))),
            }
            for e in response.edits
        ]
    except EditorDeclined as exc:
        outcome = exc.outcome.value
        error = str(exc)

    report = {
        "outcome": outcome,
        "tools_called": list(editor.last_tools_called),
        "distinct_tools_called": sorted(set(editor.last_tools_called)),
        "tool_call_count": len(editor.last_tools_called),
        "parents_read": list(editor.last_parents_read),
        "edits": edits,
        "rationale": rationale,
        "error": error,
    }
    (REPORT_DIR / "editor-report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print()
    if outcome == "valid":
        print("PASS: the editor produced a plan from real evidence.")
        return 0
    if outcome == "no_op":
        print("INCONCLUSIVE: the editor declined explicitly. Read the rationale.")
        return 0
    if outcome == "no_tool_call":
        print(
            "FAIL: the agent never finalized. This is the tracked "
            "tool-invocation risk (design §13), not a code defect."
        )
        return 1
    print(f"FAIL: editor unavailable: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
