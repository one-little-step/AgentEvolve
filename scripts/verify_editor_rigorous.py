"""Rigorous live verification of the CUGA editor agent.

The first live verification (``verify_editor_against_live_trace.py``) proved the
editor can produce one valid plan from one parent with empty history. That left
the questions that actually determine edit quality unanswered:

* Are the editor's four skills reachable, or does CUGA load stale global ones?
* Does the agent consult edit history when history exists?
* Does it use a donor parent when a donor is genuinely better?
* Does it create a new artifact when no existing artifact covers the failure,
  and does that creation survive the adapter and reach CUGA's skill loader?
* Which of the 16 tools are reachable at all, and which are never touched?

Each scenario supplies its own evidence and its own quality checks, because a
check that fits a refinement ("did the new content mention verification?") is
meaningless for a creation. Ground truth is tool-body execution and the
artifact actually produced, never the agent's prose.

Usage:
    uv run python scripts/verify_editor_rigorous.py [--scenario NAME]

Scenarios: history, crossover, creation, all (default).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_evolve.adapters.cuga_adapter import CugaAdapter  # noqa: E402
from agent_evolve.adapters.cuga_editor import (  # noqa: E402
    CugaEditorAgent,
    EditorDeclined,
)
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    ArtifactEdit,
    CandidateWorkspace,
    EvolutionTask,
    ExecutionTrace,
    MemoryRecord,
    RedactionReport,
    TraceEvent,
)
from agent_evolve.core.editor import EditorRequest, ParentContext  # noqa: E402
from agent_evolve.core.storage import JSONFileStorage  # noqa: E402
from agent_evolve.core.memory import (  # noqa: E402
    AttemptStatus,
    EditAttempt,
    EditMemory,
)
from agent_evolve.cuga_wrapper import (  # noqa: E402
    CugaWrapper,
    InMemoryRuntime,
    RuntimeSettings,
    materialize_harness,
    prepare_cuga_environment,
)

prepare_cuga_environment()

REPORT_DIR = ROOT / "terminal_output/cuga-editor/live"

ALL_TOOLS = (
    "get_mechanism",
    "list_blamed_actors",
    "get_task_input",
    "list_trace_actors",
    "read_trace_events",
    "list_artifacts",
    "read_artifact",
    "stage_replace",
    "stage_create",
    "list_staged",
    "unstage",
    "search_edit_history",
    "get_attempt_outcome",
    "list_parents",
    "read_parent_artifact",
    "submit_edit_plan",
)

# --------------------------------------------------------------------- #
# Scenario A/B evidence: refinement and crossover on a token workflow
# --------------------------------------------------------------------- #
# The primary's skill omits verification; the donor's contains it. A crossover
# is objectively the better move, so donor use is measurable rather than
# a matter of taste.
PRIMARY_SKILL = """# Token workflow

1. Fetch the alpha token.
2. Exchange it for a beta token.
3. Report the result.
"""

DONOR_SKILL = """# Token workflow

1. Fetch the alpha token.
2. Exchange it for a beta token.
3. Compute the beta checksum with the checksum tool.
4. Compare the computed checksum against the exchange response.
5. Report the verified checksum only after the comparison succeeds.
"""

TOKEN_TASK = EvolutionTask(
    task_id="task-token",
    input_text=(
        "Fetch the alpha token, exchange it for a beta token, then report "
        "the verified beta checksum."
    ),
    expected_contract={"expected_substring": "verified-checksum-OK"},
)


def token_trace() -> ExecutionTrace:
    events = (
        TraceEvent(
            event_id="graph:1",
            kind="llm_call",
            actor_id="call_model",
            parent_event_id=None,
            payload={"messages_ref": "a" * 64},
        ),
        TraceEvent(
            event_id="graph:2",
            kind="tool_call",
            actor_id="sandbox",
            parent_event_id="graph:1",
            payload={"name": "fetch_alpha", "result": "alpha ok"},
        ),
        TraceEvent(
            event_id="graph:3",
            kind="tool_call",
            actor_id="sandbox",
            parent_event_id="graph:2",
            payload={"name": "exchange", "result": "beta issued, checksum unverified"},
        ),
        TraceEvent(
            event_id="graph:4",
            kind="llm_call",
            actor_id="FinalAnswerAgent",
            parent_event_id="graph:3",
            payload={"messages_ref": "b" * 64},
        ),
    )
    return ExecutionTrace(
        trace_id="rigorous-verification",
        candidate_id="primary-v0",
        task_id="task-token",
        events=events,
        final_output="reported without verifying",
        status="completed",
    )


def token_analysis() -> CausalAnalysis:
    return CausalAnalysis(
        mechanism=(
            "the agent reported the beta checksum without verifying it against "
            "the exchange response"
        ),
        severity=0.9,
        score=0.0,
        blame_graph=BlameGraph(
            nodes=(
                BlameNode(
                    actor_id="call_model",
                    blame=0.7,
                    artifacts=("skills/token-workflow",),
                ),
                BlameNode(actor_id="FinalAnswerAgent", blame=0.3, artifacts=()),
            )
        ),
    )


# --------------------------------------------------------------------- #
# Scenario C evidence: a capability that is absent rather than wrong
# --------------------------------------------------------------------- #
# Creation is the evidenced move here for two reasons the editor can verify
# from its own tools, not because the scenario forbids anything else:
#
#   1. The blamed actor is attributed NO artifact, so there is nothing the
#      blame graph points at to refine.
#   2. The single writable artifact covers authentication only. Its content is
#      not wrong; it is silent on the failing concern.
#
# A refine into the auth skill therefore remains reachable. If the agent takes
# it, that is a reportable finding about creation discoverability, not a
# scenario that rigged the outcome.
AUTH_ONLY_SKILL = """# Service authentication

1. Fetch the alpha token from the auth endpoint.
2. Exchange it for a beta token.
3. Attach the beta token to every subsequent request.
"""

PAGINATION_TASK = EvolutionTask(
    task_id="task-records",
    input_text=(
        "Reconcile every customer record from the records API and report the "
        "total record count."
    ),
    expected_contract={"expected_substring": "total-count-OK"},
)


def pagination_trace() -> ExecutionTrace:
    events = (
        TraceEvent(
            event_id="graph:1",
            kind="llm_call",
            actor_id="call_model",
            parent_event_id=None,
            payload={"messages_ref": "c" * 64},
        ),
        TraceEvent(
            event_id="graph:2",
            kind="tool_call",
            actor_id="sandbox",
            parent_event_id="graph:1",
            payload={"name": "auth_exchange", "result": "beta token issued"},
        ),
        TraceEvent(
            event_id="graph:3",
            kind="tool_call",
            actor_id="sandbox",
            parent_event_id="graph:2",
            payload={
                "name": "list_records",
                "result": (
                    "returned 50 records; response included "
                    "next_page_token=p2 and has_more=true"
                ),
            },
        ),
        TraceEvent(
            event_id="graph:4",
            kind="llm_call",
            actor_id="FinalAnswerAgent",
            parent_event_id="graph:3",
            payload={"messages_ref": "d" * 64},
        ),
    )
    return ExecutionTrace(
        trace_id="rigorous-creation",
        candidate_id="primary-v0",
        task_id="task-records",
        events=events,
        final_output="reported 50 records",
        status="completed",
    )


def pagination_analysis() -> CausalAnalysis:
    return CausalAnalysis(
        mechanism=(
            "the agent stopped after the first page of records and never "
            "followed next_page_token, so it reported a partial count. No "
            "artifact in the harness describes pagination; the existing skill "
            "covers authentication only and its content is not incorrect."
        ),
        severity=0.85,
        score=0.0,
        blame_graph=BlameGraph(
            nodes=(
                # Attributed to no artifact: the gap is an absent capability,
                # not a wrong one.
                BlameNode(actor_id="call_model", blame=0.8, artifacts=()),
                BlameNode(actor_id="FinalAnswerAgent", blame=0.2, artifacts=()),
            )
        ),
    )


# --------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------- #
def build_memory(issue_id: str, storage_root: Path) -> EditMemory:
    """Populate real edit history, including one failed strategy to avoid.

    Uses real JSONFileStorage rather than an in-memory stub, so records pass
    through the backend's recursive sanitizer exactly as in production.
    """
    memory = EditMemory(storage=JSONFileStorage(storage_root))
    failed = EditAttempt(
        attempt_id="att-prior-001",
        candidate_id="primary-v0",
        issue_id=issue_id,
        artifact_ids=("skills/token-workflow",),
        operation="replace",
        sanitized_reasoning=(
            "Added the sentence 'be careful to verify' to the skill. Rejected: "
            "the wording was advisory, the agent still skipped verification."
        ),
        sanitized_diff={"summary": "appended one advisory sentence"},
        evidence_refs=("rigorous-verification",),
        history_refs=(),
        status=AttemptStatus.REJECTED,
    )
    memory.record(failed, artifact_group="skills", lineage="primary-v0")
    memory.append(
        MemoryRecord(
            memory_record_id="mem-001",
            attempt_id="att-prior-001",
            artifact_ids=("skills/token-workflow",),
            issue_fingerprint=issue_id,
            outcome=AttemptStatus.REJECTED,
            summary=(
                "Advisory wording ('be careful to verify') did not change "
                "behavior. A prescriptive numbered step is required."
            ),
            evidence_refs=("rigorous-verification",),
            redaction_report=RedactionReport(rule_hits=(), truncations=0),
        )
    )
    return memory


# --------------------------------------------------------------------- #
# Scenario specs
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scenario:
    """One live editor run and the checks that make its result meaningful."""

    name: str
    issue_id: str
    task: EvolutionTask
    primary_artifacts: Mapping[str, str]
    trace_factory: Callable[[], ExecutionTrace]
    analysis_factory: Callable[[], CausalAnalysis]
    donor_artifacts: Mapping[str, str] | None = None
    donor_scores: Mapping[str, float] = field(default_factory=dict)
    with_history: bool = False
    # result-dict -> named boolean quality checks
    checks: Callable[[dict], dict[str, bool]] = lambda result: {}


def _refine_checks(result: dict) -> dict[str, bool]:
    """Quality signals for a refinement, judged on the produced artifact."""
    content = result["_primary_content"].lower()
    return {
        "edit_targets_primary_artifact": any(
            e["artifact_id"] == "skills/token-workflow" for e in result["_edits"]
        ),
        "edit_mentions_verification": any(
            word in content for word in ("verif", "compare", "checksum")
        ),
        "edit_is_prescriptive_not_advisory": (
            "be careful" not in content and bool(content.strip())
        ),
        "edit_preserved_existing_steps": (
            "alpha" in content and "exchange" in content
        ),
    }


def _creation_checks(result: dict) -> dict[str, bool]:
    """Quality signals for a creation, judged on the produced artifact."""
    created = [e for e in result["_edits"] if e["operation"] == "create"]
    prefix = "skills/generated-"
    body = " ".join(e["content"].lower() for e in created)
    ids = [e["artifact_id"] for e in created]
    return {
        "staged_a_creation": bool(created),
        "created_id_is_namespaced": bool(ids) and all(
            i.startswith(prefix) and len(i) > len(prefix) for i in ids
        ),
        "respected_per_attempt_cap": len(created) <= 2,
        "created_content_addresses_pagination": any(
            term in body
            for term in ("page", "next_page", "paginat", "has_more", "cursor")
        ),
        "did_not_rewrite_unrelated_auth_skill": not any(
            e["operation"] == "replace"
            and e["artifact_id"] == "skills/service-auth"
            for e in result["_edits"]
        ),
    }


SCENARIOS: dict[str, Scenario] = {
    "history": Scenario(
        name="history",
        issue_id="issue-token-verification",
        task=TOKEN_TASK,
        primary_artifacts={"skills/token-workflow": PRIMARY_SKILL},
        trace_factory=token_trace,
        analysis_factory=token_analysis,
        with_history=True,
        checks=_refine_checks,
    ),
    "crossover": Scenario(
        name="crossover",
        issue_id="issue-token-verification",
        task=TOKEN_TASK,
        primary_artifacts={"skills/token-workflow": PRIMARY_SKILL},
        trace_factory=token_trace,
        analysis_factory=token_analysis,
        donor_artifacts={"skills/token-workflow": DONOR_SKILL},
        donor_scores={"task-token": 1.0},
        with_history=True,
        checks=_refine_checks,
    ),
    "creation": Scenario(
        name="creation",
        issue_id="issue-records-pagination",
        task=PAGINATION_TASK,
        primary_artifacts={"skills/service-auth": AUTH_ONLY_SKILL},
        trace_factory=pagination_trace,
        analysis_factory=pagination_analysis,
        with_history=False,
        checks=_creation_checks,
    ),
}


# --------------------------------------------------------------------- #
# Downstream survival of a created artifact
# --------------------------------------------------------------------- #
def verify_created_artifact_reaches_cuga(
    adapter: CugaAdapter,
    task: EvolutionTask,
    edits: list[dict],
) -> dict[str, object]:
    """Prove a created artifact survives the adapter and CUGA materialization.

    A staged creation the adapter rejects, or one that never lands as a
    loadable SKILL.md, would be a creation that reports success while reaching
    no agent -- the same failure class as bug 3 in the handoff.
    """
    created = [e for e in edits if e["operation"] == "create"]
    if not created:
        return {"attempted": False}

    out: dict[str, object] = {"attempted": True}
    workspace = adapter.materialize_candidate("primary-v0", "att-create-check")
    try:
        adapter.apply_structured_edits(
            workspace,
            [
                ArtifactEdit(
                    artifact_id=e["artifact_id"],
                    operation="create",
                    payload={"content": e["content"]},
                )
                for e in created
            ],
        )
    except Exception as exc:  # noqa: BLE001
        out["adapter_accepted"] = False
        out["adapter_error"] = str(exc)
        return out
    out["adapter_accepted"] = True

    inventory_ids = {d.artifact_id for d in adapter.artifact_inventory(workspace.version)}
    out["in_adapter_inventory"] = all(
        e["artifact_id"] in inventory_ids for e in created
    )
    out["counted_as_created"] = adapter.created_artifact_count(workspace.version)

    # White-box on purpose: _harness_config is the exact mapping CUGA receives,
    # so checking it is checking what the agent would actually load.
    harness = adapter._harness_config(workspace.version, task)
    skills = harness.get("skills") or {}
    out["in_harness_skills_group"] = all(
        e["artifact_id"].split("/", 1)[1] in skills for e in created
    )

    materialized = Path(tempfile.mkdtemp(prefix="ae-create-check-"))
    materialize_harness(harness, materialized)
    files: dict[str, object] = {}
    for e in created:
        name = e["artifact_id"].split("/", 1)[1]
        skill_file = materialized / "skills" / name / "SKILL.md"
        text = skill_file.read_text(encoding="utf-8") if skill_file.is_file() else ""
        files[e["artifact_id"]] = {
            "skill_md_exists": skill_file.is_file(),
            # A 'description: None' line is CUGA's silent skill-rejection
            # signature (bug 8), so assert the derived description survived.
            "description_is_populated": (
                "description:" in text and "description: None" not in text
            ),
        }
    out["materialized"] = files
    return out


# --------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------- #
def run_scenario(spec: Scenario) -> dict:
    settings = RuntimeSettings(
        model=os.getenv("CUGA_MODEL", "openai/azure/gpt-5.6-luna")
    )
    adapter = CugaAdapter(wrapper=CugaWrapper(InMemoryRuntime(), settings))
    adapter.register_candidate("primary-v0", dict(spec.primary_artifacts))
    if spec.donor_artifacts:
        adapter.register_candidate("donor-v0", dict(spec.donor_artifacts))

    parents = [
        ParentContext(
            candidate_id="primary",
            version="primary-v0",
            is_primary=True,
            score_summary={spec.task.task_id: 0.0},
        )
    ]
    if spec.donor_artifacts:
        parents.append(
            ParentContext(
                candidate_id="donor",
                version="donor-v0",
                is_primary=False,
                score_summary=dict(spec.donor_scores),
            )
        )

    memory = (
        build_memory(spec.issue_id, Path(tempfile.mkdtemp(prefix="ae-editmem-")))
        if spec.with_history
        else EditMemory()
    )
    editor = CugaEditorAgent(
        adapter=adapter, memory=memory, trace=spec.trace_factory()
    )

    workspace = CandidateWorkspace(
        attempt_id="att-rigorous",
        version="primary-v0",
        path=Path("."),
        parent_version="primary-v0",
    )
    write_set = tuple(spec.primary_artifacts)
    request = EditorRequest(
        base_workspace=workspace,
        task=spec.task,
        analysis=spec.analysis_factory(),
        issue_id=spec.issue_id,
        write_set=write_set,
        current_artifacts=dict(spec.primary_artifacts),
        parents=tuple(parents),
        creatable_prefix=adapter.creatable_prefix,
        pool_created_count=0,
    )

    outcome = "valid"
    error = ""
    edits: list[dict] = []
    rationale = ""
    try:
        response = editor.propose_edit(request)
        rationale = response.rationale
        edits = [
            {
                "artifact_id": e.artifact_id,
                "operation": e.operation,
                "content": str(e.payload.get("content", "")),
            }
            for e in response.edits
        ]
    except EditorDeclined as declined:
        outcome = declined.outcome.value
        error = str(declined)

    called = list(editor.last_tools_called)
    distinct = sorted(set(called))
    primary_content = next(
        (e["content"] for e in edits if e["artifact_id"] in write_set), ""
    )
    result: dict = {
        "scenario": spec.name,
        "outcome": outcome,
        "error": error,
        "tool_call_count": len(called),
        "distinct_tools_called": distinct,
        "tools_never_reached": [t for t in ALL_TOOLS if t not in distinct],
        "sdk_reported_tool_calls": len(editor.last_sdk_tool_calls),
        "consulted_history": any(
            t in distinct for t in ("search_edit_history", "get_attempt_outcome")
        ),
        "read_donor": "read_parent_artifact" in distinct,
        "parents_read": list(editor.last_parents_read),
        "edits": [
            {
                "artifact_id": e["artifact_id"],
                "operation": e["operation"],
                "content_length": len(e["content"]),
            }
            for e in edits
        ],
        "_edits": edits,
        "_primary_content": primary_content,
        "rationale": rationale,
    }
    result["checks"] = spec.checks(result)
    if spec.name == "creation":
        result["created_artifact_downstream"] = verify_created_artifact_reaches_cuga(
            adapter, spec.task, edits
        )
    # Keep produced content in the report, but out of the check inputs.
    result["new_content"] = {e["artifact_id"]: e["content"] for e in edits}
    del result["_edits"]
    del result["_primary_content"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all")
    args = parser.parse_args()

    chosen = (
        list(SCENARIOS) if args.scenario == "all" else args.scenario.split(",")
    )
    unknown = [name for name in chosen if name not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenario(s): {unknown}; known: {list(SCENARIOS)}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name in chosen:
        result = run_scenario(SCENARIOS[name])
        results.append(result)
        print(json.dumps(result, indent=2))

    suffix = "" if args.scenario == "all" else f"-{args.scenario.replace(',', '_')}"
    (REPORT_DIR / f"rigorous-report{suffix}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    reached: set[str] = set()
    for r in results:
        reached.update(r["distinct_tools_called"])
    print("\n=== AGGREGATE ===")
    print(f"tools reached across scenarios: {len(reached)}/{len(ALL_TOOLS)}")
    print(f"never reached: {sorted(set(ALL_TOOLS) - reached)}")
    for r in results:
        failed = [k for k, v in r["checks"].items() if not v]
        print(
            f"{r['scenario']:10s} outcome={r['outcome']:12s} "
            f"tools={r['tool_call_count']:2d} "
            f"history={r['consulted_history']} donor={r['read_donor']} "
            f"checks_failed={failed or 'none'}"
        )
    ok = all(
        r["outcome"] == "valid" and all(r["checks"].values()) for r in results
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
